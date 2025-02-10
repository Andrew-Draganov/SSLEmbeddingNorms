import yaml
from pathlib import Path

import os
import torch
import torch.nn as nn
import numpy as np

from CLA.training.models import SimCLR

def cosine_scaling(step, total_steps, lr_max, lr_min):
    """ cosine annealing lr """
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))

def get_lr(step, data_size, epochs, lr_max, lr_min, warmup_length=10):
    total_steps = data_size * epochs
    ten_epochs = warmup_length * data_size
    if step > ten_epochs: # Cosine learning rate after first ten epochs
        return cosine_scaling(step, total_steps, lr_max, lr_min)
    return lr_max * step / ten_epochs # Soft warmup for first ten epochs

class AttributeDict(dict):
    __slots__ = ()
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

def load_config(config_path, dataset, log_dir, model_str, chdir=True):
    conf = yaml.safe_load(Path(config_path).read_text())
    args = AttributeDict(conf)

    parent_dir = os.getcwd()
    args.data_dir = os.path.join(parent_dir, 'data')
    args.dataset = dataset

    log_dir = os.path.join(parent_dir, 'outputs', dataset, model_str, log_dir)
    args.log_dir = log_dir
    os.makedirs(log_dir, exist_ok=True)
    if chdir:
        os.chdir(log_dir)
    return args

def overwrite_config_vars(args, overrides):
    if overrides is None:
        return args

    for k, v in overrides.items():
        args[k] = v

    return args


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


@torch.no_grad()
def cut_weights(model, cut_ratio):
    for parameter in model.parameters():
        parameter.data = parameter.data / cut_ratio
