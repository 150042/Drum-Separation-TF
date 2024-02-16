import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup

from bandsplitrnn import BandSplitRNN


def load_model_generator(max_steps, args, config_dict, mode='train'):
    device_ids = list(map(int, args.device.split(',')))
    for i in range(len(device_ids)):
        device_ids[i] = i
    k = len(config_dict['K'])
    model_cfg = {
        "sr": 44100,
        "n_fft": 2048,
        "bandsplits": [
            (1000, 100),
            (4000, 250),
            (8000, 500),
            (16000, 1000),
            (20000, 2000),
        ],
        "complex_as_channel": True,
        "is_mono": False,
        "bottleneck_layer": 'rnn',
        "t_timesteps": 435,
        "fc_dim": 128,
        "rnn_dim": 256,
        "rnn_type": "LSTM",
        "bidirectional": True,
        "num_layers": 12,
        "mlp_dim": 512,
        "return_mask": False,
        "n_channel": k
    }
    model = BandSplitRNN(**model_cfg).cuda()
    # model = nn.DataParallel(model.cuda(), device_ids)
    if mode == 'train':
        # get freqs
        f = config_dict['n_fft'] // 2 + 1
        freqs = torch.arange(f) / f  # [f]
        freqs = torch.tile(freqs, [args.batch_size, k, 435, 1]).swapdims(2, 3).cuda()
        freqs.requires_grad = False
        # end freqs
        criterion = GeneratorLoss(k, freqs, config_dict['G_alpha'], config_dict['G_beta'])

        optimizer = AdamW(model.parameters(), lr=args.lr)
        scheduler = None
        # scheduler = get_linear_schedule_with_warmup(optimizer,
        #                                             num_warmup_steps=max_steps * 0.05,
        #                                             num_training_steps=max_steps)
        
    
        if args.warm_start == 1:
            print(args.checkpoint)
            last_weights = torch.load(args.checkpoint)
            model.load_state_dict(last_weights['G'])
        #     optimizer.load_state_dict(last_weights['G_optim'])
        #     scheduler.load_state_dict(last_weights['G_schedule'])
        return model, criterion, optimizer, scheduler
    else:
        # weights = torch.load('./weights/{}.pth'.format(args.nntype))
        weights = torch.load(args.checkpoint)
        model.load_state_dict(weights['G'])
        return model


class GeneratorLoss(nn.Module):
    def __init__(self, k, freqs, alpha=0, beta=0):
        super().__init__()
        self.mseLoss = nn.MSELoss()
        self.L1Loss = nn.L1Loss()
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.freqs = freqs
        self.eps = 1e-7

    def forward(self, pred, tgt):
        # batch_x = [b, 1, f, t]
        # y_target [b, k, f, t]
        loss_r = self.L1Loss(pred.real, tgt.real)
        loss_l = self.L1Loss(pred.imag, tgt.imag)
        per_loss = loss_r + loss_l
        freq_loss = self.cal_freqs_loss(torch.abs(pred), torch.abs(tgt))
        time_loss = self.cal_time_loss(torch.abs(pred), torch.abs(tgt))
        return per_loss+self.alpha*freq_loss+self.beta*time_loss

    def cal_freqs_loss(self, y_pred, y_target):
        sc_pred, ss_pred = self.cal_centroid_spread(y_pred)
        sc_target, ss_target = self.cal_centroid_spread(y_target)
        return self.L1Loss(sc_pred, sc_target) + self.L1Loss(ss_pred, ss_target)  # [b, k2, t]

    def cal_time_loss(self, y_pred, y_target):
        flux_pred = self.cal_flux(y_pred)
        flux_target = self.cal_flux(y_target)
        return self.L1Loss(flux_pred, flux_target)

    def cal_centroid_spread(self, mag):
        # mag [b, c, f, t]
        b, _, _, _ = mag.shape
        fz = torch.sum(self.freqs[:b] * mag, dim=2)  # [b,c,t]
        fm = torch.sum(mag, dim=2) + self.eps
        centroid = fz / fm
        fz = torch.sum(((self.freqs[:b] - centroid.unsqueeze(2)) ** 2) * mag, dim=2)  # [b,c,f,t]-[b,c,t]
        spread = torch.sqrt(fz / fm + self.eps)
        # spread = fz 
        return centroid, spread

    def cal_flux(self, mag):
        mag = 10 * torch.log10(mag + self.eps)
        diff = mag[:, :, :, 1:] - mag[:, :, :, :-1]
        fz = torch.sum(diff ** 2, dim=2)
        flux = torch.sqrt(fz + self.eps) / mag.shape[2]  # f
        # flux = fz / mag.shape[2]  # f
        return flux