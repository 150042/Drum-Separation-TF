import numpy as np
import torch


def load_history(args):
    history = History(args.epochs, args.nntype)
    if args.warm_start == 1:
        last_weights = torch.load(args.checkpoint)
        history.load_context(last_weights)
    return history


class History(object):
    def __init__(self, epochs, nntype):
        self.epochs = epochs
        self.nntype = nntype
        self.start_epoch = 0
        self.end_epoch = 0
        self.process_losses = []
        self.G = None 
        self.G_optim = None
        self.G_schedule = None

    def load_context(self, last_weights):
        self.G = last_weights['G']
        self.G_optim = last_weights['G_optim']
        self.G_schedule = last_weights['G_schedule']
        self.start_epoch = int(last_weights.get('start_epoch', 1))
        self.end_epoch = self.start_epoch
        self.epochs += self.start_epoch
        self.process_losses = last_weights['process_losses'][:self.start_epoch]
    
    def add_context(self, new_context):
        self.G = new_context['G']
        self.G_optim = new_context['G_optim']
        self.G_schedule = new_context['G_schedule']
        self.process_losses.append(new_context['process_losses'])
        self.end_epoch += 1

    def save_context(self, save_file):
        package = {
            'start_epoch': self.end_epoch,
            'G': self.G, 'G_optim': self.G_optim, 'G_schedule': self.G_schedule,
            'process_losses':  self.process_losses,
        }
        torch.save(package, save_file)
        



