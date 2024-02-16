import argparse
import datetime
import gc
import json
import random
import traceback
from tqdm import tqdm
import torch
import librosa
from mir_eval.separation import bss_eval_sources

import logging
import numpy as np
import os

from history import load_history
from read_data import load_dataloader
from NNs import load_model_generator
from util import plot_process_losses

log = logging.getLogger(__name__)


def valid_main(model, test_dataloader, K):
    # valid
    model.eval()
    sdr_source = {}
    for k in K:
        sdr_source[k] = []
    cnt = 0
    test_size = 0
    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            mix_mag = batch[0].cuda()
            clean_spec = batch[1].cuda()

            mix_mag = torch.repeat_interleave(mix_mag, len(K), dim=1)
            pred_spec = model(mix_mag)
            b, k, f, t = pred_spec.shape
            test_size += b  # [b, k, f, t]
            y_all = librosa.istft(pred_spec.cpu().numpy(), length=44100*5)  # [b, k, len]
            clean_y = librosa.istft(clean_spec.cpu().numpy(), length=44100*5)  # [b, k, len]
            for pred, ref in zip(y_all, clean_y):
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
                    if np.sum(pred[i] != 0) == 0:
                        flag = 1
                if flag == 1:
                    return 0, None
                kk = [K[i] for i in ref_id]
           
                sdr, sir, sar, popt = bss_eval_sources(np.asarray(ref_data), np.asarray(pred_data),
                                                       compute_permutation=False)
                for i in range(len(ref_id)):
                    sdr_source[kk[i]].append(sdr[i])
                cnt += 1

    for k in sdr_source.keys():
        if sdr_source[k] == []:
            sdr_source[k] = 0.0
        else:
            sdr_source[k] = np.median(np.asarray(sdr_source[k]))
    return cnt, sdr_source


def get_args():
    parser = argparse.ArgumentParser(description='dss')
    parser.add_argument('-debug', type=int, default=0, help='debug')

    parser.add_argument('-warm_start', type=int, default=1)
    parser.add_argument('-checkpoint', type=str)

    parser.add_argument('-nntype', type=str, default='Unet')
    parser.add_argument('-dataset', type=str, default='IDMT')
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

def main(args, info_title, config_dict):

    train_dataloader = load_dataloader(config_dict, args, "train")
    test_dataloader = load_dataloader(config_dict, args, "test")
    logging.info("load data over!")

    # 2. load model
    history = load_history(args)
    max_steps = history.epochs * len(train_dataloader)
    model, loss_function, optimizer, scheduler = load_model_generator(max_steps, args, config_dict, mode='train')
    logging.info("load model over!")

    process_losses = []
    drum_types = config_dict['K']
    mx_sdr = 0.0
    weight_file = None
    model.zero_grad()
    global_step = 0
    for epoch in range(history.start_epoch, history.epochs):
        tr_num = 0
        train_loss = 0
        model.train()
        for step, batch in enumerate(tqdm(train_dataloader, desc=f'epoch {epoch}')):
            mix_mag = batch[0].cuda()
            tgt = batch[1].cuda()
            mix_mag = torch.repeat_interleave(mix_mag, len(drum_types), dim=1)
            pred = model(mix_mag)
            loss = loss_function(pred, tgt)
            if torch.cuda.device_count() > 1:
                loss = loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            tr_num += 1
            train_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            # scheduler.step()
            global_step += 1

        train_loss /= tr_num
        process_losses.append([train_loss])
        log_msg = "epoch: {:d}\t lr: {:f}\t train_loss: {:.3f}\t".format(epoch,
                                                                       optimizer.state_dict()['param_groups'][0]['lr'],
                                                                       train_loss
                                                                       )

        if (epoch + 1) % 1 == 0:
            new_context = {'G': model.state_dict(), 'G_optim': optimizer.state_dict(), 'G_schedule': None, 'process_losses': process_losses}
            # new_context = {'G': model.state_dict(), 'G_optim': optimizer.state_dict(),
            #                 'G_schedule': scheduler.state_dict(), 'process_losses': process_losses}
            history.add_context(new_context)
   
            cnt, sdr = valid_main(model, test_dataloader, drum_types)
            log_msg += "cnt: {:d}\t".format(cnt)
            w = 0.0
            if cnt > 0:
                n = 0
                for k, v in sdr.items():
                    n += 1
                    w += v
                w /= n
                log_msg += " avg_sdr: {:.2f}\t".format(w)
                for k, v in sdr.items():
                    log_msg += " {:s}_sdr: {:.2f}\t".format(k, v)

            # valid
            if mx_sdr < w:
                mx_sdr = w
                os.makedirs('./weights', exist_ok=True)
                if weight_file is not None:
                    os.remove(weight_file)
                weight_file = f'./weights/{info_title}_epoch{epoch}.pth'
                history.save_context(weight_file)

        logging.info(log_msg)

        gc.collect()
        torch.cuda.empty_cache()

    logging.info('training over...')


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    st_time = datetime.datetime.now()
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    with open(f'config_{args.dataset}.json', 'r') as fd:
        config_dict = json.load(fd)
    config_dict["G_alpha"] = args.a
    config_dict["G_beta"] = args.b
    os.makedirs("logs", exist_ok=True)
    info_title = f'debug{args.debug}_{args.nntype}_dataset{args.dataset}_Ga{config_dict["G_alpha"]}_Gb{config_dict["G_beta"]}'
    log_file = os.path.join("logs", f'{info_title}.log')

    logging.basicConfig(filename=log_file, level=logging.INFO, handlers=None,
        filemode='a', format='%(asctime)s>> %(message)s', datefmt='%Y%m%d-%H:%M:%S')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logging.getLogger('').addHandler(console_handler)
    logging.info('cmd: {}'.format(args))

   
    set_random_seed(config_dict['seed'])

    try:
        main(args, info_title, config_dict)
    except Exception as e:
        logging.error("Error\n{}\n".format(e))
        logging.error(traceback.format_exc())
        exit(0)

    ed_time = datetime.datetime.now()
    logging.info('time: {}'.format(ed_time - st_time))

