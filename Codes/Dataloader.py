import os
import numpy as np
import torch.utils.data
import torchaudio
import random
import glob
from scipy.io import wavfile
from utils import *
from natsort import natsorted
from itertools import cycle
import soundfile as sf
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import pysepm

from torch.utils.data.distributed import DistributedSampler

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, cut_len = 32160, fs=16000,  n_samples=None):
        self.clean_dir = os.path.join(data_dir, 'clean')
        self.noisy_dir = os.path.join(data_dir, 'noisy')       
        self.noisy_wavs = glob.glob(self.noisy_dir+'/*.wav')
        self.clean_wavs = glob.glob(self.clean_dir+'/*.wav')
        self.noisy_wavs.sort()
        self.clean_wavs.sort()
        self.clean_wavs = self.clean_wavs[::-1]
        self.noisy_wavs = self.noisy_wavs[::-1]
        self.fs = fs
        self.n_samples = len(self.noisy_wavs)
        self.cut_len = cut_len
        if n_samples:
            self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        noisy_ds, _ = torchaudio.load(self.noisy_wavs[idx])
        clean_ds, _ = torchaudio.load(self.clean_wavs[idx])
        
        clean_ds = clean_ds.squeeze()
        noisy_ds = noisy_ds.squeeze()
        
        lens = len(clean_ds)//256 + 1
        
        length = len(clean_ds)
        if length < self.cut_len:
            units = self.cut_len // length
            clean_ds_final = []
            noisy_ds_final = []
            for i in range(units):
                clean_ds_final.append(clean_ds)
                noisy_ds_final.append(noisy_ds)
            clean_ds_final.append(clean_ds[: self.cut_len%length])
            noisy_ds_final.append(noisy_ds[: self.cut_len%length])
            clean_ds = torch.cat(clean_ds_final, dim=-1)
            noisy_ds = torch.cat(noisy_ds_final, dim=-1)
        else:
            # randomly cut 2 seconds segment
            wav_start = random.randint(0, length - self.cut_len)
            noisy_ds = noisy_ds[wav_start:wav_start + self.cut_len]
            clean_ds = clean_ds[wav_start:wav_start + self.cut_len]
            length = self.cut_len
        return clean_ds, noisy_ds, lens
        

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, train_dir, cut_len=32160, n_samples=None):
        self.cut_len = cut_len
        self.clean_dir = os.path.join(train_dir, 'clean/')
        self.noisy_dir = os.path.join(train_dir, 'noisy/')
        self.pitch_dir = os.path.join(train_dir, 'noisy_pitch/')
        self.clean_wav_names = glob.glob(self.clean_dir+'*.wav')
        self.noisy_wav_names = glob.glob(self.noisy_dir+'*.wav')
        self.samples = len(self.noisy_wav_names)
        if n_samples:
            self.samples = n_samples

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        clean_ds, _ = torchaudio.load(self.clean_wav_names[idx])
        noisy_ds, _ = torchaudio.load(self.noisy_wav_names[idx])
        
        clean_ds = clean_ds.squeeze()
        noisy_ds = noisy_ds.squeeze()
        
        pitch = torch.zeros((len(noisy_ds)), dtype=torch.int16)
        pitch_file = os.path.join(self.pitch_dir, self.noisy_wav_names[idx].split('/')[-1].split('.')[0]+'.bin')
        pitch = torch.from_numpy(np.fromfile(pitch_file, dtype=np.int16))

        length = len(clean_ds)
        if length < self.cut_len:
            units = self.cut_len // length
            clean_ds_final = []
            noisy_ds_final = []
            pitch_final = []
            for i in range(units):
                clean_ds_final.append(clean_ds)
                noisy_ds_final.append(noisy_ds)
                pitch_final.append(pitch)
            clean_ds_final.append(clean_ds[: self.cut_len%length])
            noisy_ds_final.append(noisy_ds[: self.cut_len%length])
            pitch_final.append(pitch[: self.cut_len%length])
            clean_ds = torch.cat(clean_ds_final, dim=-1)
            noisy_ds = torch.cat(noisy_ds_final, dim=-1)
            pitch = torch.cat(pitch_final, dim=-1)
        else:
            # randomly cut 2 seconds segment
            wav_start = random.randint(0, length - self.cut_len)
            noisy_ds = noisy_ds[wav_start:wav_start + self.cut_len]
            clean_ds = clean_ds[wav_start:wav_start + self.cut_len]
            pitch = pitch[wav_start:wav_start + self.cut_len]
            length = self.cut_len
        return clean_ds, noisy_ds, pitch

def load_data(train_dir, test_dir, batch_size, n_cpu, cut_len, test_samples=200):
    torchaudio.set_audio_backend("sox_io")         # in linux

    train_ds = TrainDataset(train_dir, cut_len)
    test_ds = TestDataset(test_dir, cut_len, n_samples=test_samples)

    train_dataset = torch.utils.data.DataLoader(dataset=train_ds, batch_size=batch_size, pin_memory=True, shuffle=False,
                                                sampler=DistributedSampler(train_ds), 
                                                drop_last=True, num_workers=n_cpu)
    test_dataset = torch.utils.data.DataLoader(dataset=test_ds, batch_size=1, pin_memory=True, shuffle=False,
                                                sampler=DistributedSampler(test_ds),
                                               drop_last=False, num_workers=n_cpu)

    return train_dataset, test_dataset
    
if __name__=="__main__":
    dataset = TrainDataset('/data/sivaganesh/pv/tvcn_torch/tvcnGAN/ValentiniData/train/')
    pesq_clean = []
    pesq_noisy = []
    for i in range(10):
        cl1, ns1, pi = dataset.__getitem__(i)
        print(cl1.shape, pi.shape, ns1.shape)
