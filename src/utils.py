import logging
import os
import sys
import time
import torch
import numpy as np


def count_parameters(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters())

def set_output_dir(save, path='checkpoints'):
    # save = '%s-%s' % (
    #     save, time.strftime("%Y%m%d-%H%M%S")
    # )
    save = '%s' % (
        save
    )
    save = os.path.join(path, save)

    if os.path.exists(save):
        num = 1
        alter_save = '%s-%03d' % (save, num)
        while os.path.exists(alter_save):
            num += 1
            alter_save = '%s-%03d' % (save, num)
            assert num <= 999, (num, alter_save)
        save = alter_save

    os.makedirs(save)

    return save


def set_logging(save=None, level=logging.INFO):
    log_format = '%(asctime)s [%(levelname)s] %(message)s'
    date_format = '%Y/%m/%d %H:%M:%S'
    logging.basicConfig(
        stream=sys.stdout, level=level,
        format=log_format, datefmt=date_format
    )
    if save is not None:
        fh = logging.FileHandler(os.path.join(save, 'log.txt'))
        fh.setFormatter(logging.Formatter(
            fmt=log_format,
            datefmt=date_format,
        ))
        logging.getLogger().addHandler(fh)


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
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