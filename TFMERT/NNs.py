import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoModel
import torch
from torch import nn
import torchaudio.transforms as T
from datasets import load_dataset
from transformers import (AdamW, get_linear_schedule_with_warmup)

def load_model_generator(max_steps, args, config_dict, mode='train'):
    device_ids = list(map(int, args.device.split(',')))
    for i in range(len(device_ids)):
        device_ids[i] = i
    k = len(config_dict['K'])
    model = DrumModel(k).cuda()
    model = nn.DataParallel(model.cuda(), device_ids)
    if mode == 'train':
        # get freqs
        f = config_dict['n_fft']//2+1
        freqs = torch.arange(f)/f # [f]
        freqs = torch.tile(freqs, [args.batch_size, k, 435, 1]).swapdims(2, 3).cuda()
        freqs.requires_grad = False
        # end freqs
        criterion = GeneratorLoss(k, freqs, config_dict['G_alpha'], config_dict['G_beta'])
        
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=max_steps * 0.05,
                                                    num_training_steps=max_steps)

        if args.warm_start == 1:
            last_weights = torch.load(args.checkpoint)
            model.load_state_dict(last_weights['G'])
        #     optimizer.load_state_dict(last_weights['G_optim']) 
        #     scheduler.load_state_dict(last_weights['G_schedule']) 
        return model, criterion, optimizer, scheduler
    else:
        weights = torch.load(args.checkpoint)
        model.load_state_dict(weights['G'])
        return model


class Decoder(nn.Module):
    def __init__(self, hidden_channels, output_channel):
        super(Decoder, self).__init__()
        self.block1 = BlockNet(hidden_channels[3], hidden_channels[2], 5, 1, 2)
        self.block2 = BlockNet(hidden_channels[2], hidden_channels[1], 5, 1, 2)
        self.block3 = BlockNet(hidden_channels[1], hidden_channels[0], 5, 1, 2)
        self.final = nn.Sequential(
            nn.Conv2d(hidden_channels[0], output_channel, 1),
            nn.ReLU(),  # mag
        )

    def forward(self, x):  # [43, 16]
        h = F.interpolate(x, (128, 64), mode='bilinear', align_corners=True)
        h = self.block1(h)
        
        h = F.interpolate(h, (512, 256), mode='bilinear', align_corners=True)
        h = self.block2(h)
        
        h = F.interpolate(h, (1025, 435), mode='bilinear', align_corners=True)
        h = self.block3(h)
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

class DrumModel(nn.Module):
    def __init__(self, drum_num):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("./model/MERT-v0", trust_remote_code=True)
        hidden_channels = [64, 256, 512, 768]
        self.decoder = Decoder(hidden_channels, drum_num)
    
    def forward(self, input_values):
        hidden_state = self.encoder(input_values)[0]
        return self.decoder(hidden_state.transpose(-1, -2).reshape(-1, 768, 43, 16))

    def frozen_encoder(self):
        for (name, param) in self.encoder.named_parameters():
            param.requires_grad = False

    def unfrozen_encoder(self):
        for (name, param) in self.encoder.named_parameters():
            param.requires_grad = True


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
