# Zhiqiang Tang, May 2017
# import sys, warnings, traceback, torch
# def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
#     sys.stderr.write(warnings.formatwarning(message, category, filename, lineno, line))
#     traceback.print_stack(sys._getframe(2))
# warnings.showwarning = warn_with_traceback; warnings.simplefilter('always', UserWarning);
# torch.utils.backcompat.broadcast_warning.enabled = True
# torch.utils.backcompat.keepdim_warning.enabled = True

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
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.nn.parameter import Parameter

from options.train_options import TrainOptions
from data.face_bbx import FACE
from models.cu_net_prev_version import create_cu_net 
from utils.util import AverageMeter
from utils.util import TrainHistoryFace, get_n_params, get_n_trainable_params, get_n_conv_params
from utils.visualizer import Visualizer
from utils.checkpoint import Checkpoint
from utils.logger import Logger
from utils.util import AdjustLR
from pylib import FaceAcc, Evaluation, HumanAug
cudnn.benchmark = True
flip_index = np.array([[0, 16], [1, 15], [2, 14], [3, 13], [4, 12], [5, 11], [6, 10], [7, 9], # outline
            [17, 26], [18, 25], [19, 24], [20, 23], [21, 22], # eyebrow
            [36, 45], [37, 44], [38, 43], [39, 42], [40, 47], [41, 46], # eye
            [31, 35], [32, 34], # nose
            [48, 54], [49, 53], [50, 52], [59, 57], [58, 56], # outer mouth
            [60, 64], [61, 63], [67, 65]]) # inner mouth
def main():
    opt = TrainOptions().parse() 
    train_history = TrainHistoryFace()
    checkpoint = Checkpoint()
    visualizer = Visualizer(opt)
    exp_dir = os.path.join(opt.exp_dir, opt.exp_id)
    log_name = opt.vis_env + 'log.txt'
    visualizer.log_name = os.path.join(exp_dir, log_name)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
    num_classes = 68
    layer_num = opt.layer_num
    net = create_cu_net(neck_size=4, growth_rate=32, init_chan_num=128,
                        class_num=num_classes, layer_num=layer_num,
                        order=1, loss_num=layer_num)

    #num1 = get_n_params(net)
    #num2 = get_n_trainable_params(net)
    #num3 = get_n_conv_params(net)
    #print 'number of params: ', num1
    #print 'number of trainalbe params: ', num2
    #print 'number of conv params: ', num3
    #exit()
    net = torch.nn.DataParallel(net).cuda()
    optimizer = torch.optim.RMSprop(net.parameters(), lr=opt.lr, alpha=0.99,
                                    eps=1e-8, momentum=0, weight_decay=0)
    """optionally resume from a checkpoint"""
    if opt.resume_prefix != '':
        # if 'pth' in opt.resume_prefix:
        #     trunc_index = opt.resume_prefix.index('pth')
        #     opt.resume_prefix = opt.resume_prefix[0:trunc_index - 1]
        checkpoint.save_prefix = os.path.join(exp_dir, opt.resume_prefix)
        checkpoint.load_prefix = os.path.join(exp_dir, opt.resume_prefix)[0:-1]
        checkpoint.load_checkpoint(net, optimizer, train_history)
    else:
        checkpoint.save_prefix = exp_dir + '/'
    print 'save prefix: ', checkpoint.save_prefix
    # model = {'state_dict': net.state_dict()}
    # save_path = checkpoint.save_prefix + 'test-model-size.pth.tar'
    # torch.save(model, save_path)
    # exit()
    """load data"""
    train_loader = torch.utils.data.DataLoader(
        FACE('dataset/face.json', '/bigdata1/zt53/data/face', is_train=True),
        batch_size=opt.bs, shuffle=True,
        num_workers=opt.nThreads, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        FACE('dataset/face.json', '/bigdata1/zt53/data/face', is_train=False),
        batch_size=opt.bs, shuffle=False,
        num_workers=opt.nThreads, pin_memory=True)

    """optimizer"""
    #optimizer = torch.optim.SGD( net.parameters(), lr=opt.lr,
    #                             momentum=opt.momentum, 
    #                             weight_decay=opt.weight_decay )
    # optimizer = torch.optim.RMSprop(net.parameters(), lr=opt.lr, alpha=0.99,
    #                                 eps=1e-8, momentum=0, weight_decay=0)
    print type(optimizer)
    # idx = range(0, 16)
    # idx = [e for e in idx if e not in (6, 7, 8, 9, 12, 13)]
    # idx = [0, 1, 2, 3, 4, 5, 10, 11, 14, 15]
    logger = Logger(os.path.join(opt.exp_dir, opt.exp_id, opt.resume_prefix+'face-training-log.txt'),
    title='face-training-summary')
    logger.set_names(['Epoch', 'LR', 'Train Loss', 'Val Loss', 'Train RMSE', 'Val RMSE'])
    if not opt.is_train:
        visualizer.log_path = os.path.join(opt.exp_dir, opt.exp_id, 'val_log.txt')
        val_loss, val_rmse, predictions = validate(val_loader, net,
                train_history.epoch[-1]['epoch'], visualizer, num_classes, flip_index)
        checkpoint.save_preds(predictions)
        return
    """training and validation"""
    start_epoch = 0
    if opt.resume_prefix != '':
        start_epoch = train_history.epoch[-1]['epoch'] + 1
    for epoch in range(start_epoch, opt.nEpochs):
        AdjustLR(opt, optimizer, epoch)
        # # train for one epoch
        train_loss, train_rmse = train(train_loader, net, optimizer, epoch, visualizer, opt)

        # evaluate on validation set
        val_loss, val_rmse, predictions = validate(val_loader, net, epoch,
                                        visualizer, num_classes, flip_index)
        # visualizer.display_imgpts(imgs, pred_pts, 4)
        # exit()
        # update training history
        e = OrderedDict( [('epoch', epoch)] )
        lr = OrderedDict( [('lr', optimizer.param_groups[0]['lr'])] )
        loss = OrderedDict( [('train_loss', train_loss),('val_loss', val_loss)] )
        rmse = OrderedDict( [('val_rmse', val_rmse)] )
        train_history.update(e, lr, loss, rmse)
        checkpoint.save_checkpoint(net, optimizer, train_history, predictions)
        visualizer.plot_train_history_face(train_history)
        logger.append([epoch, optimizer.param_groups[0]['lr'], train_loss, val_loss, train_rmse, val_rmse])
    logger.close()

    # exit()
        # if train_history.is_best:
        #     visualizer.display_imgpts(imgs, pred_pts, 4)
   
def train(train_loader, net, optimizer, epoch, visualizer, opt):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    rmses0 = AverageMeter()
    rmses1 = AverageMeter()
    rmses2 = AverageMeter()

    # switch to train mode
    net.train()

    end = time.time()
    for i, (img, heatmap, pts) in enumerate(train_loader):
        """measure data loading time"""
        data_time.update(time.time() - end)

        # input and groundtruth
        img_var = torch.autograd.Variable(img)

        heatmap = heatmap.cuda(async=True)
        target_var = torch.autograd.Variable(heatmap)

        # output and loss
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

        """measure optimization time"""
        batch_time.update(time.time() - end)
        end = time.time()

        # print log
        losses.update(loss.data[0])
        pred_pts_0, pred_pts_1, pred_pts_2 = FaceAcc.heatmap2pts(output[-1].data.cpu(), flag=0)
        # pred_pts_0 -= 1
        # pred_pts_1 -= 1
        pred_pts_2 -= 1
        # rmse0 = np.sum(FaceAcc.per_image_rmse(pred_pts_0.numpy() * 4., pts.numpy())) / img.size(0)
        # rmse1 = np.sum(FaceAcc.per_image_rmse(pred_pts_1.numpy() * 4., pts.numpy())) / img.size(0)
        rmse2 = np.sum(FaceAcc.per_image_rmse(pred_pts_2.numpy() * 4., pts.numpy())) / img.size(0)
        # rmses0.update(rmse0, img.size(0))
        # rmses1.update(rmse1, img.size(0))
        rmses2.update(rmse2, img.size(0))
        loss_dict = OrderedDict([('loss', losses.avg),
                                 ('rmse', rmses2.avg)])
        if i % opt.print_freq == 0 or i==len(train_loader)-1:
            visualizer.print_log( epoch, i, len(train_loader), batch_time.avg,
                              value1=loss_dict)

    return losses.avg, rmses2.avg

def validate(val_loader, net, epoch, visualizer, num_classes, flip_index):
    batch_time = AverageMeter()
    losses_det = AverageMeter()
    losses = AverageMeter()
    rmses0 = AverageMeter()
    rmses1 = AverageMeter()
    rmses2 = AverageMeter()
    img_batch_list = []
    pts_batch_list = []
    predictions = torch.Tensor(val_loader.dataset.__len__(), num_classes, 2)

    # switch to evaluate mode
    net.eval()

    end = time.time()
    for i, (img, heatmap, pts, index, center, scale) in enumerate(val_loader):
        # input and groundtruth
        input_var = torch.autograd.Variable(img, volatile=True)

        heatmap = heatmap.cuda(async=True)
        target_var = torch.autograd.Variable(heatmap)

        # output and loss
        #output1, output2 = net(input_var)
        #loss = (output1 - target_var) ** 2 + (output2 - target_var) ** 2
        output1 = net(input_var)
        loss = 0
        for per_out in output1:
            tmp_loss = (per_out - target_var) ** 2
            loss = loss + tmp_loss.sum() / tmp_loss.numel()


        # # flipping the image
        # img_flip = img.numpy()[:, :, :, ::-1].copy()
        # img_flip = torch.from_numpy(img_flip)
        # input_var = torch.autograd.Variable(img_flip, volatile=True)
        # #output11, output22 = net(input_var)
        # output2 = net(input_var)
        # output2 = HumanAug.flip_channels(output2[-1].cpu().data)
        # output2 = HumanAug.shuffle_channels_for_horizontal_flipping(output2, flip_index)
        # output = (output1[-1].cpu().data + output2) / 2

        # calculate measure
        output = output1[-1].data.cpu()
        # pred_pts_0, pred_pts_1, pred_pts_2 = FaceAcc.heatmap2pts(output)
        # pred_pts_origin_0 = pred_pts_0.clone()
        # pred_pts_origin_1 = pred_pts_1.clone()
        # pred_pts_origin_2 = pred_pts_2.clone()
        # for j in range(pred_pts_0.size(0)):
        #     # print type(coords[i]), type(center[i]), type(scale[i])
        #     tmp_pts = HumanAug.TransformPts(pred_pts_0[j].numpy()-1,
        #                                     center[j].numpy(), scale[j].numpy(), 0, 64, 200, invert=1)
        #     pred_pts_origin_0[j] = torch.from_numpy(tmp_pts)
        #     pred_pts_origin_1[j] = Evaluation.transform_preds(pred_pts_1[j], center[j], scale[j], [64, 64])
        #     pred_pts_origin_2[j] = Evaluation.transform_preds(pred_pts_2[j], center[j], scale[j], [64, 64])
        # rmse0 = np.sum(FaceAcc.per_image_rmse(pred_pts_origin_0.numpy(), pts.numpy())) / img.size(0)
        # rmse1 = np.sum(FaceAcc.per_image_rmse(pred_pts_origin_1.numpy(), pts.numpy())) / img.size(0)
        # rmse2 = np.sum(FaceAcc.per_image_rmse(pred_pts_origin_2.numpy(), pts.numpy())) / img.size(0)
        # rmses0.update(rmse0, img.size(0))
        # rmses1.update(rmse1, img.size(0))
        # rmses2.update(rmse2, img.size(0))
        preds = Evaluation.final_preds(output, center, scale, [64, 64])
        rmse = np.sum(FaceAcc.per_image_rmse(preds.numpy(), pts.numpy())) / img.size(0)
        rmses2.update(rmse, img.size(0))
        """measure elapsed time"""
        batch_time.update(time.time() - end)
        end = time.time()

        # print log
        losses.update(loss.data[0])
        loss_dict = OrderedDict( [('loss', losses.avg),
                                  ('rmse', rmses2.avg)] )
        visualizer.print_log( epoch, i, len(val_loader), batch_time.avg,
                              value1=loss_dict)
        # preds = Evaluation.final_preds(output, center, scale, [64, 64])
        for n in range(output.size(0)):
            predictions[index[n], :, :] = preds[n, :, :]

    return losses.avg, rmses2.avg, predictions



if __name__ == '__main__':
    main()
