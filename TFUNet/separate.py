import json
import datetime
import argparse
import torch
import numpy as np
import random
import logging
import os
import gc
import traceback
from tqdm import tqdm
from utils.util import *
import soundfile as sf
import librosa
import shutil

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def read_wav_file_pad(cpath, config_dict):
    sr, n_fft = config_dict['sr'], config_dict['n_fft']
    y, _ = librosa.load(cpath, sr=sr, mono=False)
    assert y.shape[-1] <= 5*sr, 'tlen < 5*sr'
    y_pad = np.concatenate([y, np.zeros((y.shape[0], 5*sr-y.shape[-1]+n_fft))], axis=1)
    S = librosa.stft(y_pad, n_fft=n_fft)
    return np.abs(S), np.angle(S), y.shape[-1]


def main(args, info_title, config_dict):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
    # 1. load model
    G = load_model_generator(args, config_dict, mode='test')  
    logging.info("load model over!")
    
    # 2. load mix song
    mix_path = os.path.join(config_dict['test_path'], 'mix')
    mix_wavnames = sorted(os.listdir(mix_path))
    if args.debug > 0:
        # mix_wavnames = mix_wavnames[:args.debug]
        mix_wavnames = mix_wavnames[::10]
        # pass
    elif args.debug < 0:
        mix_wavnames = ['{}.wav'.format(-args.debug)]
    
    # 3. separate
    G.eval()
    k, sr = len(config_dict['K']), config_dict['sr']
    result_path = config_dict['result_path']+'_'+info_title
    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    os.mkdir(result_path)
    for wav_name in tqdm(mix_wavnames, desc='separating'):
        cpath = os.path.join(mix_path, wav_name)
        mag, phase, num = read_wav_file_pad(cpath, config_dict)
        with torch.no_grad():
            batch_x = torch.tensor(mag[np.newaxis,:,:,:], dtype=torch.float32).cuda() #[1,2,f,t]
            mask = G(batch_x)   # [b, k*2, f, t] 
            b, k2, f, t = mask.shape
            mask = mask.view(b, k, -1, f, t) # mask [b, k, 2, f, t]
            mag_pred = mask * batch_x.unsqueeze(dim=1) # mag [b,k,2,f,t]
            mag_pred = mag_pred.squeeze(dim=0) # [k, 2, f, t]
        mag_pred = mag_pred.cpu().numpy()
        phase = phase[np.newaxis,:,:,:] # [1,2,f,t]
        spec = mag_pred*np.cos(phase)+mag_pred*np.sin(phase)*(1j)
        y_all = librosa.istft(spec, length=num) # [k,2, len]
        mix_name = wav_name[:-4]
        save_path = os.path.join(result_path, mix_name)
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.mkdir(save_path)
        shutil.copy(cpath, os.path.join(save_path, wav_name))
        for drum_type, y in zip(config_dict['K'], y_all):
            sf.write(os.path.join(save_path, drum_type+".wav"), y.T, sr)
        
    logging.info('separating over...') 



def get_args():
    parser = argparse.ArgumentParser(description='dss')
    # debug
    parser.add_argument('-debug', type=int, default=0, help='debug')
    parser.add_argument('-gan_alpha', type=float, default=0.1, help='gan_alpha')
    parser.add_argument('-checkpoint', type=str)
    
    # hyperparam
    parser.add_argument('-a', type=float, default=0.0, help='batch_size')
    parser.add_argument('-b', type=float, default=0.0, help='epochs')
    parser.add_argument('-nntype', type=str)
    parser.add_argument('-device', type=str, default='2,3', help='cuda')
    return parser.parse_args()

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    # configuration
    with open('./config.json', 'r') as fd:
        config_dict = json.load(fd)
      
    st_time = datetime.datetime.now()
    args = get_args()
    config_dict["G_alpha"] = args.a
    config_dict["G_beta"] = args.b
    # info_title = args.nntype
    info_title = f'{args.nntype}_Ga{config_dict["G_alpha"]}_Gb{config_dict["G_beta"]}'


    if not os.path.exists("logs/"):
        os.mkdir("logs")
    log_file = "logs/test_{}.log".format(info_title)
    logging.basicConfig(filename=log_file, level=logging.INFO, handlers=None,
        filemode='a', format='%(asctime)s>> %(message)s', datefmt='%Y%m%d-%H:%M:%S')
   
    set_random_seed(config_dict['seed'])
    
    try:
        main(args, info_title, config_dict)
    except Exception as e:
        logging.error("Error!\n{}\n".format(e))
        logging.error(traceback.format_exc())
        exit(0)

    ed_time = datetime.datetime.now()
    logging.info('separate time: {}'.format(ed_time - st_time))


