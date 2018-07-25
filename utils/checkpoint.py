# Xi Peng, May 2017
import os, shutil
import torch
from torch.nn.parameter import Parameter
import scipy.io

class Checkpoint():
    def __init__(self):
        # self.opt = opt
        self.save_prefix = ''
        self.load_prefix = ''

    def save_checkpoint(self, net, optimizer, train_history, preds):
        lr_prefix = ('lr-%.15f' % train_history.lr[-1]['lr']).rstrip('0').rstrip('.')
        save_path = self.save_prefix + lr_prefix + ('-%d.pth.tar' % train_history.epoch[-1]['epoch'])
        save_pred_path = self.save_prefix + lr_prefix + ('-%d-preds.mat' % train_history.epoch[-1]['epoch'])
        checkpoint = { 'train_history': train_history.state_dict(), 
                       'state_dict': net.state_dict(),
                        'optimizer': optimizer.state_dict()}
        torch.save( checkpoint, save_path )
        print("=> saving '{}'".format(save_path))
        preds = preds.numpy()
        scipy.io.savemat(save_pred_path, mdict={'preds': preds})
        print("=> saving '{}'".format(save_pred_path))
        if train_history.is_best:
            save_path2 = self.save_prefix + lr_prefix + ('-%d-model-best.pth.tar' % train_history.epoch[-1]['epoch'])
            print("=> saving best checkpoint '{}'".format(save_path2))
            shutil.copyfile(save_path, save_path2)
            save_pred_path2 = self.save_prefix + lr_prefix + ('-%d-preds-best.mat' % train_history.epoch[-1]['epoch'])
            print("=> saving best predictions '{}'".format(save_pred_path2))
            shutil.copyfile(save_pred_path, save_pred_path2)

    def save_preds(self, preds):
        # print preds.size()
        preds = preds.numpy()
        # lr_prefix = ('lr-%.15f' % train_history.lr[-1]['lr']).rstrip('0').rstrip('.')
        save_pred_path = self.save_prefix + '-%d-preds.mat'
        scipy.io.savemat(save_pred_path, mdict={'preds': preds})

    def load_checkpoint(self, net, optimizer, train_history):
        # if not self.opt.load_checkpoint:
        #     return
        save_path = self.load_prefix + '.pth.tar'
        # self.save_path = self.save_path + '-'
        if os.path.isfile(save_path):
            print("=> loading checkpoint '{}'".format(save_path))
            checkpoint = torch.load(save_path)

            train_history.load_state_dict( checkpoint['train_history'] )
            # net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            state_dict = checkpoint['state_dict']

            net_dict = net.state_dict()
            for name, param in state_dict.items():
              # name = name[7:] #????????????????
               if name not in net_dict:
                   print("=> not load weights '{}'".format(name))
                   continue
               if isinstance(param, Parameter):
                   param = param.data
               net_dict[name].copy_(param)
            #     print("load weights '{}'".format(name))
            # print( "=> loaded checkpoint '{}'\t=> epoch:{}"
            #       .format(save_name, self.opt.resume_epoch) )
        else:
            print("=> no checkpoint found at '{}'".format(save_path))
  
