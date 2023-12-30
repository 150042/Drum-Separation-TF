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
from utils.read_data import load_dataloader
from utils.util import *


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    
def main(args, info_title, config_dict):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
    # 1. load wav_name 
    train_dataloader = load_dataloader(config_dict, args, mode='train')
    test_dataloader = load_dataloader(config_dict, args, mode='test')
    logging.info("load data over!")
    
    # 2. load model
    G, G_loss, G_optim = load_model_generator(args, config_dict, mode='train')  
    history = load_history(args)
    logging.info("load model over!")
    
    # 3. training
    process_losses = []
    drum_types = config_dict['K']
    mx_sdr = 0.0
    weight_file = None
    for epoch in tqdm(range(history.start_epoch, history.epochs), desc='train {}'.format(args.nntype)):
        # train
        G, G_optim, train_loss = train_one_epoch_wo(G, G_loss, G_optim, train_dataloader)   
        log_msg = "epoch: {:d}\t train_loss: {:.3f}\t".format(epoch, train_loss)
        process_losses.append([train_loss.item()])
        
        # save checkpoint
        if (epoch+1) % 10 == 0:
            # add context to history
            new_context = {'G': G.state_dict(), 'G_optim': G_optim.state_dict(), 'process_losses': process_losses}
            history.add_context(new_context)
            # valid
            # test_dataloader = load_dataloader(config_dict, args, mode='test')
            test_loss, cnt, sdr = test_one_epoch(G, G_loss, test_dataloader, drum_types)
            w = 0.0
            if cnt > 0:
                n = 0
                for k, v in sdr.items():
                    n += 1
                    w += v
                w /= n

            if mx_sdr < w:
                mx_sdr = w 
                if not os.path.exists('./weights'):
                    os.mkdir('./weights')
                if weight_file is not None:
                    os.remove(weight_file)
                weight_file = f'./weights/{info_title}_epoch{epoch}.pth'
                history.save_context(weight_file)
            
            log_msg += "cnt: {:d}\t avg_sdr: {:.2f}\t".format(cnt, w)
            if sdr is not None:
                for k, v in sdr.items():
                    log_msg += " {:s}_sdr: {:.2f}\t".format(k, v)

        logging.info(log_msg)

    
    # 4. plot
    plot_process_losses(process_losses, info_title)
    logging.info('trainning over...') 

def get_args():
    parser = argparse.ArgumentParser(description='dss')
    parser.add_argument('-debug', type=int, default=0, help='debug')
   
    parser.add_argument('-warm_start', type=int, default=1)
    parser.add_argument('-checkpoint', type=str)
    
    parser.add_argument('-nntype', type=str, default='Unet')
    parser.add_argument('-device', type=str, default='2,3', help='cuda')

    parser.add_argument('-a', type=float, default=0.0, help='batch_size')
    parser.add_argument('-b', type=float, default=0.0, help='epochs')
    parser.add_argument('-batch_size', type=int, default=4, help='batch_size')
    parser.add_argument('-epochs', type=int, default=100, help='epochs')
    parser.add_argument('-lr', type=float, default=1e-3, help='lr')
    parser.add_argument('-half_lr', type=int, default=5)
    parser.add_argument('-early_stop', type=int, default=10, help='early_stop')
    parser.add_argument('-weight_decay', type=float, default=0.01, help='weight_decay')
    
    return parser.parse_args()

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    with open('config.json', 'r') as fd:
        config_dict = json.load(fd)
      
    st_time = datetime.datetime.now()
    args = get_args()
    config_dict["G_alpha"] = args.a
    config_dict["G_beta"] = args.b
    info_title = f'debug{args.debug}_{args.nntype}_Ga{config_dict["G_alpha"]}_Gb{config_dict["G_beta"]}'

    if not os.path.exists("logs/"):
        os.mkdir("logs")
    log_file = "logs/{}.log".format(info_title)
    logging.basicConfig(filename=log_file, level=logging.INFO, handlers=None,
        filemode='a', format='%(asctime)s>> %(message)s', datefmt='%Y%m%d-%H:%M:%S')
    logging.info('args: {}'.format(args))
   
    set_random_seed(config_dict['seed'])
    
    # train
    try:
        main(args, info_title, config_dict)
    except Exception as e:
        logging.error("Error!\n{}\n".format(e))
        logging.error(traceback.format_exc())
        exit(0)

    ed_time = datetime.datetime.now()
    logging.info('training time: {}'.format(ed_time - st_time))


