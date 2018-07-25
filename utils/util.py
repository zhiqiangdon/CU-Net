# Xi Peng, Feb 2017
import os, sys
import numpy as np
from PIL import Image
from collections import OrderedDict
import torch

class TrainHistory():
    """store statuses from the 1st to current epoch"""
    def __init__(self):
        self.epoch = []
        self.lr = []
        self.loss = []
        self.pckh = []
        self.best_pckh = 0.
        self.is_best = True

    def update(self, epoch, lr, loss, pckh):
        # lr, epoch, loss, rmse (OrderedDict)
        # epoch = OrderedDict([('epoch',1)] )
        # loss = OrderedDict( [('train_loss',0.1),('val_loss',0.2)] )
        self.epoch.append(epoch)
        self.lr.append(lr)
        self.loss.append(loss)
        self.pckh.append(pckh)

        self.is_best = pckh['val_pckh'] > self.best_pckh
        self.best_pckh = max(pckh['val_pckh'], self.best_pckh)

    def state_dict(self):
        dest = OrderedDict()
        dest['epoch'] = self.epoch
        dest['lr'] = self.lr
        dest['loss'] = self.loss
        dest['pckh'] = self.pckh
        dest['best_pckh'] = self.best_pckh
        dest['is_best'] = self.is_best
        return dest

    def load_state_dict(self, state_dict):
        self.epoch = state_dict['epoch']
        self.lr = state_dict['lr']
        self.loss = state_dict['loss']
        self.pckh = state_dict['pckh']
        self.best_pckh = state_dict['best_pckh']
        self.is_best = state_dict['is_best']

class TrainHistoryFace():
    """store statuses from the 1st to current epoch"""
    def __init__(self):
        self.epoch = []
        self.lr = []
        self.loss = []
        self.rmse = []
        self.best_rmse = 1.
        self.is_best = True

    def update(self, epoch, lr, loss, rmse):
        # lr, epoch, loss, rmse (OrderedDict)
        # epoch = OrderedDict([('epoch',1)] )
        # loss = OrderedDict( [('train_loss',0.1),('val_loss',0.2)] )
        self.epoch.append(epoch)
        self.lr.append(lr)
        self.loss.append(loss)
        self.rmse.append(rmse)

        self.is_best = rmse['val_rmse'] < self.best_rmse
        self.best_rmse = min(rmse['val_rmse'], self.best_rmse)

    def state_dict(self):
        dest = OrderedDict()
        dest['epoch'] = self.epoch
        dest['lr'] = self.lr
        dest['loss'] = self.loss
        dest['rmse'] = self.rmse
        dest['best_rmse'] = self.best_rmse
        dest['is_best'] = self.is_best
        return dest

    def load_state_dict(self, state_dict):
        self.epoch = state_dict['epoch']
        self.lr = state_dict['lr']
        self.loss = state_dict['loss']
        self.rmse = state_dict['rmse']
        self.best_rmse = state_dict['best_rmse']
        self.is_best = state_dict['is_best']


class AverageMeter():
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

def adjust_lr(opt, optimizer, epoch):
    if epoch < 101:
        for param_group in optimizer.param_groups:
                print(param_group['lr'])
        return
    elif epoch == 101:
         opt.lr = opt.lr * 0.2
    elif epoch == 141:
         opt.lr = opt.lr * 0.5
    elif epoch == 161:
         opt.lr = opt.lr * 0.5
    for param_group in optimizer.param_groups:
        param_group['lr'] = opt.lr
        print(param_group['lr'])

def AdjustLR(opt, optimizer, epoch):
    if epoch < 30:
        for param_group in optimizer.param_groups:
                print(param_group['lr'])
        return
    elif epoch == 30:
         opt.lr = opt.lr * 0.2
    elif epoch == 60:
         opt.lr = opt.lr * 0.5
    elif epoch == 90:
         opt.lr = opt.lr * 0.5
    for param_group in optimizer.param_groups:
        param_group['lr'] = opt.lr
        print(param_group['lr'])


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def get_n_trainable_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    number = sum([np.prod(p.size()) for p in model_parameters])
    return number

def get_n_conv_params(model):
    pp=0
    for name, param in model.named_parameters():
        # nn = 1
        if 'conv' in name:
            # for s in list(param.size()):
            #     nn = nn*s
            pp += param.data.nelement()
    return pp

