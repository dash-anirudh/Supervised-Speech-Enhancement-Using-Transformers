import os
os.environ["CUDA_VISIBLE_DEVICES"]='3'
import itertools
import argparse
import time
import logging
import torch
import pysepm
import shutil
import timeit
import torch.nn.functional as F
import torch.multiprocessing as mp
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import dataloader_conv
from tensorboard_conv import TensorboardWriter
from torchinfo import summary
from conv2 import EnhancementModel
import Losses
from valid_length import valid_length_time
from torch_pesq import PesqLoss

seed = 0  #Fixing a seed for reproducibility
# random.seed(seed) #Ensures constant values for random.random or random.shuffle
os.environ['PYTHONHASHSEED'] = str(seed) #Fixes the hash seed in Python's environment variables to control hash randomization
# np.random.seed(seed) #Fixes seed for np.random.rand() or np.random.choice() 
torch.manual_seed(seed) #Seeds the PyTorch random number generator for CPU tensors
torch.cuda.manual_seed(seed) #Seeds the random number generator on the current GPU
torch.cuda.manual_seed_all(seed) #Ensures that all GPUs (if you are using multiple GPUs) have the same random seed.
torch.backends.cudnn.deterministic = True

#Use command python filename.py -h to see the help commands
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--log_interval", type=int, default=50)
parser.add_argument("--decay_epoch", type=int, default=30, help="epoch from which to start lr decay")
parser.add_argument("--init_lr", type=float, default=1e-3, help="initial learning rate")
parser.add_argument("--cut_len", type=int, default=32160, help="cut length, default is 2 seconds in denoise "
                                                                 "and dereverberation")
parser.add_argument("--sup_data_dir", type=str, default='/data/sivaganesh/pv/tvcn_torch/tvcnGAN/ValentiniData/train/',
                    help="dir of training dataset")
parser.add_argument("--test_data_dir", type=str, default='/data/sivaganesh/pv/tvcn_torch/tvcnGAN/ValentiniData/test/',
                    help="dir of testing dataset")
parser.add_argument("--save_model_dir", type=str, default='checkpoints_conv_3_1e-3/',
                    help="dir of saved model")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO) #Debug wont be shown i.e., use case logging.info("Info Message")

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost" #defines the IP address or hostname of the master node, "localhost" means the master node is the same machine
    os.environ["MASTER_PORT"] = "13100" #defines the port number on the master node that the processes will use to communicate. 
    init_process_group(backend="nccl", rank=rank, world_size=world_size) 

class Trainer:
    def __init__(self, train_ds, test_ds, gpu_id: int, args=None):
        #If apc_reps are to be used, fixed values of n_fft, hop, winlen
        self.n_fft = 512
        self.hop = 160
        self.winlen = 400
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.args = args
        self.tensorboard = TensorboardWriter(args.save_model_dir)
        self.model = EnhancementModel().cuda()

        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
        #shutil requires source, destination so files named tvcn.py anf train.py will be copied into directory (as destination is directory)
        shutil.copy('conv2.py', args.save_model_dir)
        shutil.copy('train_conv.py', args.save_model_dir)
        # self.sigma=2**15
        
        #Input to model is time domain signal
        summary(self.model, (args.batch_size,args.cut_len))
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-6)
        
        self.model = DDP(self.model, device_ids=[gpu_id])
        self.gpu_id = gpu_id
        
        self.pesq = PesqLoss(1, sample_rate = 16000).cuda()
        self.mask_loss = Losses.MaskLoss()
        self.cepstral_loss = Losses.CepstralLoss()
        self.sissnr_loss = Losses.Si_SSNR()

    def train_step(self, epoch, batch):
        #batch->clean, noisy, pitch, noisy, pitch
        clean_sup = batch[0].to(self.gpu_id)
        noisy_sup = batch[1].to(self.gpu_id)

        clean_spec_cmplx_sup = torch.stft(clean_sup, n_fft=self.n_fft, hop_length=self.hop, win_length=self.winlen, window=torch.hann_window(self.winlen).to(clean_sup.device), return_complex=True)
        clean_spec_sup = torch.abs(clean_spec_cmplx_sup)
        
        noisy_spec_cmplx_sup = torch.stft(noisy_sup, n_fft=self.n_fft, hop_length=self.hop, win_length=self.winlen, window=torch.hann_window(self.winlen).to(clean_sup.device), return_complex=True)
        noisy_spec_sup = torch.abs(noisy_spec_cmplx_sup)
        
        self.optimizer.zero_grad()
        est_audio_sup, est_spec_cmplx_sup, est_cmplx_mask = self.model(noisy_sup)
        est_spec_sup = torch.abs(est_spec_cmplx_sup)
        est_mask_sup = torch.abs(est_cmplx_mask)
        
        m_loss = 10*self.mask_loss(clean_spec_sup, noisy_spec_sup, est_mask_sup)
        
        est_audio_sup = est_audio_sup.squeeze(1)
        p_loss = torch.mean(self.pesq(clean_sup, est_audio_sup))
        
        s_loss = 0.2*self.sissnr_loss(clean_sup, est_audio_sup)
        
        loss = m_loss + p_loss + s_loss 
        loss.backward()
        self.optimizer.step()
        #.item() to make loss a float instead of tensor
        return loss.item()#, [m_loss.item(), recon_loss.item()]

    @torch.no_grad()
    def test_metrics(self):
        self.model.eval()
        print('********Starting metrics evaluation on val dataset**********')
        total_pesq = 0.0
        count=0
        sr = 16000
        with torch.no_grad():
            for k, (clean_wav, noisy_wav, _) in tqdm(enumerate(self.test_ds)):
                noisy_wav = noisy_wav.to(self.gpu_id)
                valid_length = valid_length_time(noisy_wav.shape[-1])
                if(valid_length>noisy_wav.shape[-1]):
                    noisy_wav = F.pad(noisy_wav, (0, valid_length_time(noisy_wav.shape[-1]) - noisy_wav.shape[-1]))
                est_wav, _, _ = self.model(noisy_wav)
                est_wav = est_wav.squeeze()  # [sig_len_recover,]
                clean_wav = clean_wav.squeeze()
                L = min(est_wav.shape[-1], clean_wav.shape[-1])
                est_wav = est_wav[:L]  # keep length same (output label)
                clean_wav = clean_wav[:L]
                est_sp = est_wav.cpu().numpy()
                cln_raw = clean_wav.numpy()
                pe = pysepm.pesq(cln_raw, est_sp, sr)[-1]
                total_pesq += pe
                count += 1
        return total_pesq / count

    def train(self):
        scheduler_G = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5)
        g_loss = [];
        patience = 50
        prev_test_metric = 0
        torch.autograd.set_detect_anomaly(True)
        for epoch in range(args.epochs):
            beg_t = timeit.default_timer()
            avg_loss = 0.
            self.model.train()
            for idx, batch in enumerate(self.train_ds):
                step = idx + 1
                loss = self.train_step(epoch, batch)
                # loss, gen_losses = self.train_step(epoch, batch)
                self.tensorboard.log_training([loss], epoch*len(self.train_ds)+step)
                template = 'GPU: {}, Epoch {}, Step {}, loss: {:.3f}'
                if (step % args.log_interval) == 0:
                    logging.info(template.format(self.gpu_id, epoch, step, loss))
                avg_loss += loss
            test_pesq = self.test_metrics()
            self.tensorboard.log_evaluation(test_pesq, epoch)
            if test_pesq > prev_test_metric:
                prev_test_metric = test_pesq
                patience = 50
            else:
                patience -= 1
            g_loss.append(avg_loss/step)
            end_t = timeit.default_timer()
            print('Epoch:{} Patience_stauts:{} train_time:{:.2f} loss:{:.3f} the PESQ: {:.2f}'.format(epoch, patience, end_t-beg_t, avg_loss/step, test_pesq))
            path = os.path.join(args.save_model_dir, 'tvcn_epoch_' + str(epoch) + '_' + str(test_pesq)[:5])
            if self.gpu_id == 0:
                #Save parameter values of each model at path
                torch.save(self.model.module.state_dict(), path)
            scheduler_G.step(avg_loss/step)
            if patience==0:
                break

def main(rank: int, world_size: int, args):
    ddp_setup(rank, world_size)
    if rank == 0:
        print(args)
        available_gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        print(available_gpus)
    train_ds, test_ds = dataloader_conv.load_data(args.sup_data_dir, args.test_data_dir, args.batch_size, 2, args.cut_len)
    trainer = Trainer(train_ds, test_ds, rank, args)
    trainer.train()
    destroy_process_group()


if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args), nprocs=world_size)