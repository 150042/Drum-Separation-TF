import os
from torch.utils.data import Dataset, DataLoader, RandomSampler
import numpy as np
import librosa
import torch

def load_dataloader(config_dict, args, mode='train'):
    if mode == 'train':
        data_dataset = DrumDataset(config_dict, mode, args.debug)
        train_sampler = RandomSampler(data_dataset)
        data_dataloader = DataLoader(data_dataset, sampler=train_sampler, batch_size=args.batch_size,
                              num_workers=4, pin_memory=True)
    else:
        data_dataset = DrumDataset(config_dict, mode, debug=0)
        data_dataloader = DataLoader(data_dataset, args.batch_size, num_workers=4, shuffle=False)
    return data_dataloader


class DrumDataset(Dataset):
    def __init__(self, config_dict, mode, debug=0):
        super().__init__()
        self.mode = mode
        mix_path = os.path.join(config_dict[f'{mode}_path'], 'mix')
        clean_path = os.path.join(config_dict[f'{mode}_path'], 'clean')
        with open(os.path.join(config_dict[f'{mode}_path'], "reflect.csv"), "r") as fd:
            content = fd.read().split("\n")
        self.reflect = dict()
        for line in content:
            if len(line) < 2 : # EOF
                break
            if '\t' in line:
                x = line.split('\t')
            else:
                x = line.split(',')
            self.reflect[x[0]] = x[1:]
            
        mix_wavnames = sorted(os.listdir(mix_path))
        clean_wavnames = sorted(os.listdir(clean_path))
        if debug:
            tot_num = len(mix_wavnames)
            step = tot_num//debug
            mix_wavnames = mix_wavnames[::step]
            mix_wavnames = mix_wavnames[:debug]
            clean_wavnames = list()
            for wav_name in mix_wavnames:
                clean_wavnames.extend(self.reflect[wav_name])
        self.mix_y = dict()
        self.mix_data = dict()
        self.phase_data = dict()
        for wav_name in mix_wavnames:
            cpath = os.path.join(mix_path, wav_name)
            mag, phase, y = read_wav_file(cpath, config_dict)
            self.mix_y[wav_name] = y.reshape(-1) 
            self.mix_data[wav_name] = mag 
            if self.mode != 'train':    
                self.phase_data[wav_name] = phase

        self.clean_data = dict()
        self.clean_y = dict()
        for wav_name in clean_wavnames:
            cpath = os.path.join(clean_path, wav_name)
            mag, phase, y = read_wav_file(cpath, config_dict)
            self.clean_data[wav_name] = mag
            if self.mode != 'train':
                self.clean_y[wav_name] = y
        self.mix_wavnames = mix_wavnames
          
    def __len__(self):
        return len(self.mix_wavnames)

    def __getitem__(self, idx):
        mix_y = (self.mix_y[self.mix_wavnames[idx]]).astype(np.float32)   # [tlen]
        mix_data = (self.mix_data[self.mix_wavnames[idx]]).astype(np.float32)    # [1, f, t]
        clean_data = []
        clean_y = []
        for wav_name in self.reflect[self.mix_wavnames[idx]]:
            clean_data.append(self.clean_data[wav_name])
            if self.mode != 'train':
                clean_y.append(self.clean_y[wav_name])
        clean_data = np.asarray(clean_data, dtype=np.float32).squeeze(1) # train[k, f, t]
        if self.mode != 'train':
            clean_y = np.asarray(clean_y, dtype=np.float32).squeeze(1)  # [k, T]
            phase_data = self.phase_data[self.mix_wavnames[idx]]
            phase_data = np.asarray(phase_data, dtype=np.float32)  # [1, f, t]
            return mix_y, clean_data, mix_data, phase_data, clean_y
        return mix_y, clean_data, mix_data

def read_wav_file(cpath, config_dict):
    """
    y: (1/2, 5*sr)
    S: 1025*435
    """
    sr, n_fft = config_dict['sr'], config_dict['n_fft']
    y, _ = librosa.load(cpath, sr=sr, mono=False)
    if len(y.shape) == 1:
        y = np.vstack([y, y])
    y = y[0].reshape(1, -1)
    y_pad = np.concatenate([y, np.zeros((y.shape[0], n_fft))], axis=1)
    S = librosa.stft(y_pad, n_fft=n_fft)
    return np.abs(S), np.angle(S), y
    




