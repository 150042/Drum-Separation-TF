import torch.nn as nn
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
import datetime
import os
import librosa
import numpy as np


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
