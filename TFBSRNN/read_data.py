import os
import numpy as np
import librosa
from torch.utils.data import Dataset, RandomSampler, DataLoader
import torch
from tqdm import tqdm

def read_wav_file(cpath, config_dict):
    """
    y: (2, 5*sr)
    S: 1025*435
    """
    sr, n_fft = config_dict['sr'], config_dict['n_fft']
    y, _ = librosa.load(cpath, sr=sr, mono=False)
    # if len(y) > 0 and len(config_dict["K"]) > 3:
    #     y = y[0]
    if len(y.shape) > 1:
        y = y[0]
    y = y.reshape(1, -1)
    y_pad = np.concatenate([y, np.zeros((y.shape[0], n_fft))], axis=1)
    S = librosa.stft(y_pad, n_fft=n_fft)
    # return np.abs(S), np.angle(S), y
    return S, S, y


def load_dataloader(config_dict, args, mode='train'):
    if mode == 'train':
        if args.dataset=='syn52idmt':
            config_dict['flag'] = 1
        data_dataset = DrumDataset(config_dict, mode, args.debug)
        train_sampler = RandomSampler(data_dataset)
        data_dataloader = DataLoader(data_dataset, sampler=train_sampler, batch_size=args.batch_size,
                                     num_workers=4, pin_memory=True)
    else:
        test_debug = 0
        if len(config_dict["K"]) > 3:
            test_debug = 200
        if args.dataset=='idmt2syn5':
            test_debug = 200
            config_dict['flag'] = 1
        if args.dataset=='syn52idmt':
            test_debug = 0
        data_dataset = DrumDataset(config_dict, mode, debug=test_debug)
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
            if len(line) < 2:  
                break
            if '\t' in line:
                x = line.split('\t')
            else:
                x = line.split(',')
            self.reflect[x[0]] = x[1:]
        
        if 'flag' in config_dict.keys():
            tmp_list = []
            for wav_name in os.listdir(mix_path):
                cle = self.reflect[wav_name]
                if len(cle)>3 and (cle[3] != '0.wav' or cle[4] != '0.wav'):
                    continue
                self.reflect[wav_name] = cle[:3]
                tmp_list.append(wav_name)
            mix_wavnames = sorted(tmp_list)
            debug = min(debug, len(mix_wavnames))
        else:
            mix_wavnames = sorted(os.listdir(mix_path))
        # mix_wavnames = sorted(os.listdir(mix_path))
        clean_wavnames = sorted(os.listdir(clean_path))
        if debug > 0:
            tot_num = len(mix_wavnames)
            step = tot_num // debug
            mix_wavnames = mix_wavnames[::step]
            mix_wavnames = mix_wavnames[:debug]
            # if 'flag' in config_dict.keys():
            #     tmp_list = []
            #     for wav_name in mix_wavnames:
            #         cle = self.reflect[wav_name]
            #         if len(cle)>3 and (cle[3] != '0.wav' or cle[4] != '0.wav'):
            #             continue
            #         self.reflect[wav_name] = cle[:3]
            #         tmp_list.append(wav_name)
            #     mix_wavnames = tmp_list
            clean_wavnames = list()
            for wav_name in mix_wavnames:
                clean_wavnames.extend(self.reflect[wav_name])
        self.mix_y = dict()
        self.mix_data = dict()
        self.phase_data = dict()

        for wav_name in tqdm(mix_wavnames, desc=f'loading {mode} mix data'):
            cpath = os.path.join(mix_path, wav_name)
            mag, phase, y = read_wav_file(cpath, config_dict)
            self.mix_y[wav_name] = y.reshape(-1)
            self.mix_data[wav_name] = mag
            self.phase_data[wav_name] = phase

        self.clean_data = dict()
        self.clean_y = dict()
        for wav_name in tqdm(clean_wavnames, desc=f'loading {mode} tgt data'):
            cpath = os.path.join(clean_path, wav_name)
            mag, phase, y = read_wav_file(cpath, config_dict)
            self.clean_data[wav_name] = mag
            self.clean_y[wav_name] = y
        self.mix_wavnames = mix_wavnames

    def __len__(self):
        return len(self.mix_wavnames)

    def __getitem__(self, idx):
        mix_data = self.mix_data[self.mix_wavnames[idx]]
        clean_data = []
        clean_y = []
        for wav_name in self.reflect[self.mix_wavnames[idx]]:
            clean_data.append(self.clean_data[wav_name])
            clean_y.append(self.clean_y[wav_name])
        mix_data = torch.as_tensor(mix_data, dtype=torch.cfloat)
        clean_data = np.asarray(clean_data, dtype=np.complex_).squeeze(1)  # train[k, f, t]
        clean_data = torch.as_tensor(clean_data, dtype=torch.cfloat)
        return mix_data, clean_data
