import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

# Stride and Kernel take 1D inputs

def valid_length_time(input_length, stride = (1,1,1,1), kernel = (3,3,3,3), window_length = 400, hop_length = 160):  
  centered_time_bins = 1+(input_length)//hop_length
  if(len(stride)!=len(kernel)):
    print('Pass stride and kernel size in the same dimension')
    return None
  number_of_layers = len(stride)
  A = -1*kernel[0]
  B = stride[0]
  for idx in range(number_of_layers-1):
    A += B*(1 - kernel[idx+1])
    B *= stride[idx+1]
  if((centered_time_bins + A) % B ==0):
    required_time_bins = centered_time_bins
  else:
    required_time_bins = centered_time_bins + B - (centered_time_bins + A) % B
  
  return (required_time_bins-1)*hop_length


class EnhancementModel(nn.Module):
    def __init__(self, hop=160, n_fft=512, win=400):
        super(EnhancementModel, self).__init__()
        self.hop = hop
        self.n_fft = n_fft
        self.win = win 

        # Encoder Convolutional Layers
        self.conv1 = nn.Conv2d(2, 16, kernel_size=(3, 4), stride=(1, 2), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 16, kernel_size=(3, 4), stride=(1, 2), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(16)
        
        self.conv3 = nn.Conv2d(16, 16, kernel_size=(3, 4), stride=(1, 2), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(16)
        
        self.conv4 = nn.Conv2d(16, 16, kernel_size=(3, 4), stride=(1, 2), padding=(1, 1))
        self.bn4 = nn.BatchNorm2d(16)
        
        # self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 4), stride=(1, 2), padding=(1, 1))
        # self.bn3 = nn.BatchNorm2d(64)
        
        # self.conv4 = nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        # self.bn4 = nn.BatchNorm2d(128)
        
        # self.conv5 = nn.Conv2d(128, 256, kernel_size=(3, 4), stride=(1, 2), padding=(1, 1))
        # self.bn5 = nn.BatchNorm2d(256)

        # Transformer Encoder Layers for Bottleneck
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=32*101, nhead=101, batch_first=True)
        # self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=1)

        # Decoder Convolutional Layers
        # self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=(3, 4), stride=(1, 2), padding=(1, 1))
        # self.bn_deconv1 = nn.BatchNorm2d(128)
        
        # self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        # self.bn_deconv2 = nn.BatchNorm2d(64)
        
        # self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=(3, 4), stride=(1, 2), padding=(1, 1))
        # self.bn_deconv3 = nn.BatchNorm2d(32)
        
        self.deconv2 = nn.ConvTranspose2d(16, 16, kernel_size=(3, 4), stride=(1, 2), padding=(1, 1))
        self.bn_deconv2 = nn.BatchNorm2d(16)
        
        self.deconv3 = nn.ConvTranspose2d(16, 16, kernel_size=(3, 4), stride=(1, 2), padding=(1, 1))
        self.bn_deconv3 = nn.BatchNorm2d(16)
        
        self.deconv4 = nn.ConvTranspose2d(16, 16, kernel_size=(3, 4), stride=(1, 2), padding=(1, 1))
        self.bn_deconv4 = nn.BatchNorm2d(16)
        
        self.deconv5 = nn.ConvTranspose2d(16, 2, kernel_size=(3, 4), stride=(1, 2), padding=(1, 1))
        self.bn_deconv5 = nn.BatchNorm2d(2)

        self.activation = nn.LeakyReLU()

    def forward(self, x):
        # STFT Transformation
        x_spec = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop, win_length=self.win, window=torch.hann_window(self.win).to(x.device), return_complex=True)
        x_spec = x_spec.unsqueeze(1)
        x_real = x_spec.real
        x_imag = x_spec.imag
        x_input_spec = torch.cat((x_real, x_imag), dim=1)
        x = x_input_spec.permute(0, 1, 3, 2)
        x = x[..., :-1]

        # Encoding Path
        x_conv1 = self.activation(self.bn1(self.conv1(x)))
        x_conv2 = self.activation(self.bn2(self.conv2(x_conv1)))
        x_conv3 = self.activation(self.bn3(self.conv3(x_conv2)))
        x_conv4 = self.activation(self.bn4(self.conv4(x_conv3)))
        # x_conv5 = self.activation(self.bn5(self.conv5(x_conv4)))
        
        # Flatten for Transformer
        batch_size, channels, height, width = x_conv4.size()
        x_flat = x_conv4.view(batch_size, channels * height, width).permute(0, 2, 1)  # Shape: [batch_size, width, channels * height]

        # Transformer Encoder for Bottleneck
        # print(x_flat.shape)
        x_transformed = self.transformer_encoder_layer(x_flat)
        x_transformed = x_transformed.permute(0, 2, 1).view(batch_size, channels, height, width)
        # print(x_transformed.shape)
        # print(x_conv3.shape)

        # Decoding Path with Skip Connections
        # x_deconv1 = self.activation(self.bn_deconv1(self.deconv1(x_transformed))) #+ x_conv4
        x_deconv2 = self.activation(self.bn_deconv2(self.deconv2(x_transformed)))# + x_conv3
        # print(x_deconv2.shape)
        x_deconv3 = self.activation(self.bn_deconv3(self.deconv3(x_deconv2))) + x_conv2
        x_deconv4 = self.activation(self.bn_deconv4(self.deconv4(x_deconv3))) + x_conv1
        x_deconv5 = torch.tanh(self.bn_deconv5(self.deconv5(x_deconv4)))

        # Reshape and Masking
        x_deconv5 = x_deconv5.permute(0, 1, 3, 2)
        mask = F.pad(x_deconv5, (0, 0, 0, 1))
        enhanced_spec = mask * x_input_spec

        # Complex Mask Application
        mask_real = mask[:, 0, :, :]
        mask_imag = mask[:, 1, :, :]
        mask_spec = torch.complex(mask_real, mask_imag).squeeze(1)
        enhanced_real = enhanced_spec[:, 0, :, :]
        enhanced_imag = enhanced_spec[:, 1, :, :]
        enhanced_cmplx_spec = torch.complex(enhanced_real, enhanced_imag).squeeze(1)

        # ISTFT
        enhanced_audio = torch.istft(enhanced_cmplx_spec, n_fft=self.n_fft, hop_length=self.hop, win_length=self.win, window=torch.hann_window(self.win).to(x.device), return_complex=False)
        return enhanced_audio, enhanced_cmplx_spec, mask_spec

if __name__ == '__main__':
    model = EnhancementModel()
    print(summary(model, input_size=(8, valid_length_time(32160))))
