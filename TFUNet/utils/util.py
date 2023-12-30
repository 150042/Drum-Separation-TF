from utils.NNs import *
import torch.nn as nn
import torch.optim as optim
import torch
from utils.History import History
import matplotlib.pyplot as plt
import datetime
import os
import librosa
import numpy as np
from mir_eval.separation import bss_eval_sources

def load_model_generator(args, config_dict, mode='train'):
    device_ids = list(map(int, args.device.split(',')))
    for i in range(len(device_ids)):
        device_ids[i] = i
    k = len(config_dict['K'])
    model = Unet(1, k)
    model = nn.DataParallel(model.cuda(), device_ids)
    if mode == 'train':
        # get freqs
        f = config_dict['n_fft']//2+1
        freqs = torch.arange(f)/f # [f]
        freqs = torch.tile(freqs, [args.batch_size, k, 435, 1]).swapdims(2, 3).cuda()
        freqs.requires_grad = False
        # end freqs
        criterion = GeneratorLoss(k, freqs, config_dict['G_alpha'], config_dict['G_beta'])

        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
       
        if args.warm_start == 1:
            # save context
            last_weights = torch.load(args.checkpoint)
            last_weights['G_optim']['param_groups'][0]['lr'] *= 2
            model.load_state_dict(last_weights['G'])
            optimizer.load_state_dict(last_weights['G_optim']) 
        return model, criterion, optimizer
    else:
        weights = torch.load(args.checkpoint)
        model.load_state_dict(weights['G'])
        return model

def load_history(args):
    history = History(args.epochs, args.nntype)
    if args.warm_start == 1:
        last_weights = torch.load(args.checkpoint)
        history.load_context(last_weights)
    return history

def train_one_epoch_wo(G, G_loss, G_optim, train_dataloader):
    G.train()
    train_loss, train_size = 0.0, 0
    for batch_x, batch_y, _, _ in train_dataloader:
        batch_x = batch_x.cuda()   
        batch_y = batch_y.cuda()
        y_pred = G(batch_x)
        loss = G_loss(y_pred, batch_y, batch_x)
        G_optim.zero_grad()
        loss.backward()
        G_optim.step()
        train_loss += loss*batch_x.shape[0]
        train_size += batch_x.shape[0]
    return G, G_optim, train_loss/train_size

def test_one_epoch(G, G_loss, test_dataloader, K):
    G.eval()
    sdr_source = {}
    for k in K:
        sdr_source[k] = []
    cnt = 0
    test_loss, test_size = 0.0, 0
    with torch.no_grad():
        for batch_x, batch_y, phase, label_y in test_dataloader:
            batch_x = batch_x.cuda()   # [b, 1, f, t]
            batch_y = batch_y.cuda()
            # print(batch_x.shape, batch_y.shape)
            mask = G(batch_x)  # [b, k, f, t]
            loss = G_loss(mask, batch_y, batch_x)
            test_loss += loss*batch_x.shape[0]
            test_size += batch_x.shape[0]
            b, k, f, t = mask.shape
            mag_pred = mask * batch_x   # mag [b,k,f,t]
            phase = phase.cuda()
            spec = mag_pred*torch.cos(phase)+mag_pred*torch.sin(phase)*(1j)     # [b, k, f, t]            
            y_all = librosa.istft(spec.cpu().numpy(), length=label_y.shape[-1])     # [b, k, len]
            label_y = label_y.numpy()
            for pred, ref in zip(y_all, label_y):
                # pred, ref  [k, T]
                ref_data = []
                pred_data = []
                ref_id = []
                flag = 0  
                for i in range(len(K)):
                    if ref[i].max() == 0:
                        continue
                    ref_id.append(i)
                    # read ref
                    ref_data.append(ref[i])
                    pred_data.append(pred[i])
                    if np.sum(pred[i]!=0) == 0:
                        flag = 1
                if flag == 1:
                    return -1, 0, {'KD': 0, 'SD': 0, 'HH': 0}
                kk = [K[i] for i in ref_id]  
                # sdr, sir, sar
                sdr, sir, sar, popt = bss_eval_sources(np.asarray(ref_data), np.asarray(pred_data), compute_permutation=False)
                for i in range(len(ref_id)):
                    sdr_source[kk[i]].append(sdr[i])
                cnt += 1

    for k in sdr_source.keys():
        sdr_source[k] = np.median(np.asarray(sdr_source[k]))
    return test_loss/test_size, cnt, sdr_source



def plot_process_losses(process_losses, title):
    sub_titiles = ['train_loss', 'train_G_loss', 'train_D_loss', 'g_acc', 't_acc', 'f_acc']
    plt.figure()
    x = range(len(process_losses))
    if len(process_losses[0]) == 1: # Unet
        plt.plot(x, process_losses, label=sub_titiles[0])
    else:
        plt.suptitle(title)
        for i in range(len(sub_titiles)):
            plt.subplot(2, 3, i+1)
            plt.title(sub_titiles[i])
            plt.plot(x, process_losses[i], label=sub_titiles[i])
        plt.tight_layout()
   
    cur_time = datetime.datetime.now()
    if not os.path.exists('./plots'):
        os.mkdir('./plots')
    plt.savefig('plots/{}_{}.png'.format(title, cur_time.strftime('%Y-%m-%d %H:%M:%S')))
    plt.close()
