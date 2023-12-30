import torch
import torch.nn as nn
import torch.nn.functional as F

class Unet(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(Unet, self).__init__()
        hidden_channels = [64, 128, 256, 512]
        self.encoder = Encoder(input_channel, hidden_channels)
        self.decoder = Decoder(hidden_channels, output_channel)

    def forward(self, x):
        # [b, c, f, t]
        z = self.encoder(x)
        y = self.decoder(z)
        return y


class Encoder(nn.Module):
    def __init__(self, input_channel, hidden_channels):
        super(Encoder, self).__init__()
        self.block1 = BlockNet(input_channel, hidden_channels[0], 3, 1, 1)
        self.block2 = nn.Sequential(
            BlockNet(hidden_channels[0], hidden_channels[1], 3, 1, 1),
            nn.MaxPool2d(2, 2)
        )
        self.block3 = nn.Sequential(
            BlockNet(hidden_channels[1], hidden_channels[2], 3, 1, 1),
            nn.MaxPool2d(2, 2)
        )
        self.block4 = nn.Sequential(
            BlockNet(hidden_channels[2], hidden_channels[3], 3, 1, 1),
            nn.MaxPool2d(2, 2)
        )

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        return x1, x2, x3, x4

class Decoder(nn.Module):
    def __init__(self, hidden_channels, output_channel):
        super(Decoder, self).__init__()
        self.block1 = BlockNet(hidden_channels[3]+hidden_channels[2], hidden_channels[2], 3, 1, 1)
        self.block2 = BlockNet(hidden_channels[2]+hidden_channels[1], hidden_channels[1], 3, 1, 1)
        self.block3 = BlockNet(hidden_channels[1]+hidden_channels[0], hidden_channels[0], 3, 1, 1)
        # self.final = nn.Conv2d(hidden_channels[0], output_channel, 1)
        self.final = nn.Sequential(
            nn.Conv2d(hidden_channels[0], output_channel, 1),
            nn.ReLU(),  # mag
        )

    def forward(self, x):
        x1, x2, x3, x4 = x
        
        h = F.interpolate(x4, x3.shape[2:], mode='bilinear', align_corners=True)
        h = self.block1(torch.cat((x3, h), 1))
        
        h = F.interpolate(h, x2.shape[2:], mode='bilinear', align_corners=True)
        h = self.block2(torch.cat((x2, h), 1))
        
        h = F.interpolate(h, x1.shape[2:], mode='bilinear', align_corners=True)
        h = self.block3(torch.cat((x1, h), 1))
        
        y = self.final(h)
       
        return y
                

class BlockNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, block_num=1):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.convs = nn.Sequential(*[
            BlockConv(out_channels) for i in range(block_num)
        ])
    
    def forward(self, x):
        x = self.transform(x)
        y = self.convs(x)
        return y

class BlockConv(nn.Module):

    def __init__(self, out_channels):
        super().__init__()
        self.patch_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        )
    
    def forward(self, x):
        return x + self.patch_conv(x)

class GeneratorLoss(nn.Module):
    def __init__(self, k, freqs, alpha=0, beta=0):
        super().__init__()
        self.mseLoss = nn.MSELoss()
        self.alpha = alpha
        self.beta=beta
        self.k = k
        self.freqs=freqs
        self.eps=1e-7
    
    def forward(self, mask, y_target, batch_x):
        # batch_x = [b, 1, f, t]
        # y_target [b, k, f, t]
        y_pred = mask * batch_x     # mag [b,k,f,t]
        per_loss = self.mseLoss(y_pred, y_target)
        freq_loss = self.cal_freqs_loss(y_pred, y_target)
        time_loss = self.cal_time_loss(y_pred, y_target)
        return per_loss+self.alpha*freq_loss+self.beta*time_loss

    def cal_freqs_loss(self, y_pred, y_target):
        sc_pred, ss_pred = self.cal_centroid_spread(y_pred)
        sc_target, ss_target = self.cal_centroid_spread(y_target) 
        return self.mseLoss(sc_pred, sc_target)+self.mseLoss(ss_pred, ss_target)  # [b, k2, t]
    
    def cal_time_loss(self, y_pred, y_target):
        flux_pred = self.cal_flux(y_pred)
        flux_target = self.cal_flux(y_target)
        return self.mseLoss(flux_pred, flux_target)

    def cal_centroid_spread(self, mag):
        # mag [b, c, f, t]
        b, _, _, _ = mag.shape
        fz = torch.sum(self.freqs[:b]*mag, dim=2)   # [b,c,t]
        fm = torch.sum(mag, dim=2)+self.eps
        centroid = fz / fm 
        fz = torch.sum(((self.freqs[:b]-centroid.unsqueeze(2))**2)*mag, dim=2)  # [b,c,f,t]-[b,c,t]
        spread = torch.sqrt(fz/fm+self.eps)
        return centroid, spread
    
    def cal_flux(self, mag):
        mag = 10*torch.log10(mag+self.eps)
        diff = mag[:,:,:,1:] - mag[:,:,:,:-1]
        fz = torch.sum(diff**2, dim=2)
        flux = torch.sqrt(fz+self.eps)/mag.shape[2]  # f
        return flux

if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    k = 5
    n_fft=2048
    sr = 44100
    b, k2, f, t = 2, k*2, n_fft//2+1, 435
    import numpy as np
    freqs = np.arange(n_fft//2+1) # [f]
    freqs = np.tile(freqs, [b, k2, t, 1]).swapaxes(2, 3)
    freqs = torch.tensor(freqs).to(device)
    criterion = GeneratorLoss(k, freqs, alpha=0.1, beta=0.1)
    mask=torch.rand((b,k2,f,t)).to(device)
    y_target=10*torch.rand((b,k2,f,t)).to(device)
    y_label=None 
    batch_x=10*torch.rand((b,2,f,t)).to(device)
    loss = criterion(mask, y_target, y_label, batch_x)
    print(loss)

