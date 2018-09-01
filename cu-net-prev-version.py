# Zhiqiang Tang, May 2017
#import sys, warnings, traceback, torch
#def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
#    sys.stderr.write(warnings.formatwarning(message, category, filename, lineno, line))
#    traceback.print_stack(sys._getframe(2))
#warnings.showwarning = warn_with_traceback; warnings.simplefilter('always', UserWarning)
#torch.utils.backcompat.broadcast_warning.enabled = True
#torch.utils.backcompat.keepdim_warning.enabled = True

import os, time
from PIL import Image, ImageDraw
import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

from options.train_options import TrainOptions
from data.mpii_for_mpii_22 import MPII
from models.cu_net_prev_version import create_cu_net  
from utils.util import AverageMeter, adjust_lr
from utils.util import TrainHistory, get_n_params, get_n_trainable_params, get_n_conv_params
from utils.visualizer import Visualizer
from utils.checkpoint import Checkpoint
from utils.logger import Logger
from pylib import HumanAcc, HumanPts, HumanAug
import pylib.Evaluation_prev_version as Evaluation

cudnn.benchmark = True
joint_flip_index = np.array([[1, 4], [0, 5],
                             [12, 13], [11, 14], [10, 15], [2, 3]])


def main():
    opt = TrainOptions().parse()
    train_history = TrainHistory()
    checkpoint = Checkpoint()
    visualizer = Visualizer(opt)
    exp_dir = os.path.join(opt.exp_dir, opt.exp_id)
    log_name = opt.vis_env + 'log.txt'
    visualizer.log_name = os.path.join(exp_dir, log_name)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
    # if opt.dataset == 'mpii':
    class_num = 16
    layer_num = opt.layer_num
    net = create_cu_net(neck_size=4, growth_rate=32, init_chan_num=128,
                            class_num=class_num, layer_num=layer_num,
                            order=1, loss_num=layer_num)
    # num1 = get_n_params(net)
    # num2 = get_n_trainable_params(net)
    # num3 = get_n_conv_params(net)
    # print 'number of params: ', num1
    # print 'number of trainalbe params: ', num2
    # print 'number of conv params: ', num3
    # torch.save(net.state_dict(), 'test-model-size.pth.tar')
    # exit()
    # device = torch.device("cuda:0")
    # net = net.to(device)
    net = torch.nn.DataParallel(net).cuda()
    optimizer = torch.optim.RMSprop(net.parameters(), lr=opt.lr, alpha=0.99,
                                    eps=1e-8, momentum=0, weight_decay=0)
    """optionally resume from a checkpoint"""
    if opt.resume_prefix != '':
        # if 'pth' in opt.resume_prefix:
        #     trunc_index = opt.resume_prefix.index('pth')
        #     opt.resume_prefix = opt.resume_prefix[0:trunc_index - 1]
        # checkpoint.save_prefix = os.path.join(exp_dir, opt.resume_prefix)
        checkpoint.save_prefix = exp_dir + '/'
        checkpoint.load_prefix = os.path.join(exp_dir, opt.resume_prefix)[0:-1]
        checkpoint.load_checkpoint(net, optimizer, train_history)
        opt.lr = optimizer.param_groups[0]['lr']
        resume_log = True
    else:
        checkpoint.save_prefix = exp_dir + '/'
        resume_log = False
    print 'save prefix: ', checkpoint.save_prefix
    # model = {'state_dict': net.state_dict()}
    # save_path = checkpoint.save_prefix + 'test-model-size.pth.tar'
    # torch.save(model, save_path)
    # exit()
    """load data"""
    train_loader = torch.utils.data.DataLoader(
        MPII('dataset/mpii-hr-lsp-normalizer.json', '/bigdata1/zt53/data', is_train=True),
        batch_size=opt.bs, shuffle=True,
        num_workers=opt.nThreads, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        MPII('dataset/mpii-hr-lsp-normalizer.json', '/bigdata1/zt53/data', is_train=False),
        batch_size=opt.bs, shuffle=False,
        num_workers=opt.nThreads, pin_memory=True)

    """optimizer"""
    # optimizer = torch.optim.SGD( net.parameters(), lr=opt.lr,
    #                             momentum=opt.momentum,
    #                             weight_decay=opt.weight_decay )
    # optimizer = torch.optim.RMSprop(net.parameters(), lr=opt.lr, alpha=0.99,
    #                                 eps=1e-8, momentum=0, weight_decay=0)
    print type(optimizer)
    # idx = range(0, 16)
    # idx = [e for e in idx if e not in (6, 7, 8, 9, 12, 13)]
    idx = [0, 1, 2, 3, 4, 5, 10, 11, 14, 15]
    logger = Logger(os.path.join(opt.exp_dir, opt.exp_id, 'training-summary.txt'),
                                 title='training-summary', resume=resume_log)
    logger.set_names(['Epoch', 'LR', 'Train Loss', 'Val Loss', 'Train Acc', 'Val Acc'])
    if not opt.is_train:
        visualizer.log_path = os.path.join(opt.exp_dir, opt.exp_id, 'val_log.txt')
        val_loss, val_pckh, predictions = validate(val_loader, net,
                                                   train_history.epoch[-1]['epoch'],
                                                   visualizer, idx, joint_flip_index,
                                                   num_classes)
        checkpoint.save_preds(predictions)
        return

    """training and validation"""
    start_epoch = 0
    if opt.resume_prefix != '':
        start_epoch = train_history.epoch[-1]['epoch'] + 1
    for epoch in range(start_epoch, opt.nEpochs):
        adjust_lr(opt, optimizer, epoch)
        # # train for one epoch
        train_loss, train_pckh = train(train_loader, net, optimizer,
                                       epoch, visualizer, idx, opt)

        # evaluate on validation set
        val_loss, val_pckh, predictions = validate(val_loader, net, epoch,
                                                   visualizer, idx, joint_flip_index,
                                                   num_classes)
        # visualizer.display_imgpts(imgs, pred_pts, 4)
        # exit()
        # update training history
        e = OrderedDict([('epoch', epoch)])
        lr = OrderedDict([('lr', optimizer.param_groups[0]['lr'])])
        loss = OrderedDict([('train_loss', train_loss), ('val_loss', val_loss)])
        pckh = OrderedDict([('val_pckh', val_pckh)])
        train_history.update(e, lr, loss, pckh)
        checkpoint.save_checkpoint(net, optimizer, train_history, predictions)
        # visualizer.plot_train_history(train_history)
        logger.append([epoch, optimizer.param_groups[0]['lr'], train_loss,
                       val_loss, train_pckh, val_pckh])
    logger.close()

    # exit()
    # if train_history.is_best:
    #     visualizer.display_imgpts(imgs, pred_pts, 4)


def train(train_loader, net, optimizer, epoch, visualizer, idx, opt):
    # batch_time = AverageMeter()
    # data_time = AverageMeter()
    losses = AverageMeter()
    pckhs = AverageMeter()
    pckhs_origin_res = AverageMeter()
    # switch to train mode
    net.train()

    # end = time.time()
    for i, (img, heatmap, c, s, r, grnd_pts,
            normalizer) in enumerate(train_loader):
        # """measure data loading time"""
        # data_time.update(time.time() - end)

        # input and groundtruth
        img_var = torch.autograd.Variable(img)
        heatmap = heatmap.cuda(async=True)
        target_var = torch.autograd.Variable(heatmap)

        # output and loss
        # output1, output2 = net(img_var)
        # loss = (output1 - target_var) ** 2 + (output2 - target_var) ** 2
        output = net(img_var)
        # exit()
        # print(type(output))
        # print(len(output))
        loss = 0
        for per_out in output:
            tmp_loss = (per_out - target_var) ** 2
            loss = loss + tmp_loss.sum() / tmp_loss.numel()

        # gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # """measure optimization time"""
        # batch_time.update(time.time() - end)
        # end = time.time()
        # print log
        losses.update(loss.data[0])

        pckh = Evaluation.accuracy(output[-1].data.cpu(), target_var.data.cpu(), idx)
        pckhs.update(pckh[0])
        pckh_origin_res = Evaluation.accuracy_origin_res(output[-1].data.cpu(),
                                                         c, s, [64, 64],
                                                         grnd_pts, normalizer, r)
        pckhs_origin_res.update(pckh_origin_res[0])

        loss_dict = OrderedDict([('loss', losses.avg),
                                 ('pckh', pckhs.avg),
                                 ('pckh_origin_res', pckhs_origin_res.avg)])
        if i % opt.print_freq == 0 or i == len(train_loader) - 1:
            visualizer.print_log(epoch, i, len(train_loader),
                                 value1=loss_dict)
        # if i == 1:
        #     break
    return losses.avg, pckhs_origin_res.avg


def validate(val_loader, net, epoch, visualizer, idx, joint_flip_index, num_classes):
    batch_time = AverageMeter()
    losses = AverageMeter()
    pckhs = AverageMeter()
    pckhs_origin_res = AverageMeter()
    img_batch_list = []
    pts_batch_list = []
    predictions = torch.Tensor(val_loader.dataset.__len__(), num_classes, 2)

    # switch to evaluate mode
    net.eval()

    # end = time.time()
    for i, (img, heatmap, center, scale, rot, grnd_pts,
            normalizer, index) in enumerate(val_loader):
        # input and groundtruth
        input_var = torch.autograd.Variable(img, volatile=True)

        heatmap = heatmap.cuda(async=True)
        target_var = torch.autograd.Variable(heatmap)

        # output and loss
        # output1, output2 = net(input_var)
        # loss = (output1 - target_var) ** 2 + (output2 - target_var) ** 2
        output1 = net(input_var)
        loss = 0
        for per_out in output1:
            tmp_loss = (per_out - target_var) ** 2
            loss = loss + tmp_loss.sum() / tmp_loss.numel()

        # flipping the image
        img_flip = img.numpy()[:, :, :, ::-1].copy()
        img_flip = torch.from_numpy(img_flip)
        input_var = torch.autograd.Variable(img_flip, volatile=True)
        # output11, output22 = net(input_var)
        output2 = net(input_var)
        output2 = HumanAug.flip_channels(output2[-1].data.cpu())
        output2 = HumanAug.shuffle_channels_for_horizontal_flipping(output2, joint_flip_index)
        output = (output1[-1].data.cpu() + output2) / 2

        # calculate measure
        # pred_pts = HumanPts.heatmap2pts(output)  # b x L x 2
        # pts = HumanPts.heatmap2pts(target_var.cpu().data)
        # pckh = HumanAcc.approx_PCKh(pred_pts, pts, idx, heatmap.size(3))  # b -> 1
        pckh = Evaluation.accuracy(output, target_var.data.cpu(), idx)
        pckhs.update(pckh[0])
        pckh_origin_res = Evaluation.accuracy_origin_res(output, center, scale, [64, 64],
                                                         grnd_pts, normalizer, rot)
        pckhs_origin_res.update(pckh_origin_res[0])
        # """measure elapsed time"""
        # batch_time.update(time.time() - end)
        # end = time.time()

        # print log
        losses.update(loss.data[0])
        loss_dict = OrderedDict([('loss', losses.avg),
                                 ('pckh', pckhs.avg),
                                 ('pckh_origin_res', pckhs_origin_res.avg)])
        visualizer.print_log(epoch, i, len(val_loader), value1=loss_dict)
        # img_batch_list.append(img)
        # pts_batch_list.append(pred_pts*4.)
        preds = Evaluation.final_preds(output, center, scale, [64, 64], rot)
        for n in range(output.size(0)):
            predictions[index[n], :, :] = preds[n, :, :]

        # if i == 1:
        #     break
    return losses.avg, pckhs_origin_res.avg, predictions


if __name__ == '__main__':
    main()
