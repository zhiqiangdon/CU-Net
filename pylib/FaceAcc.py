# Xi Peng, Apr 2017
import os, sys
import numpy as np
from PIL import Image
import torch
import math
import Evaluation

########## detection accuracy ##########
def per_class_f1score(output, target):
    # output: b x n x h x w tensor [0, 1] sigmoid output
    # target: b x n x h x w tensor [0, 1] discrete 0, 1
    assert (output.size() == target.size())
    f1score = torch.zeros(output.size(1))
    counter = torch.ones(output.size(1)) * output.size(0)
    for i in range(0, output.size(0)):
        for j in range(0, output.size(1)):
            per_predict_map = output[i, j].gt(0.5)
            per_recall_map = target[i, j][per_predict_map]
            grnd_total = torch.sum(target[i, j])
            pred_total = per_predict_map.sum()
            if grnd_total == 0:
                counter[j] -= 1
                continue
            if pred_total == 0:
                continue
            pred_correct = per_recall_map.sum()
            recall = float(pred_correct) / float(grnd_total)
            precision = float(pred_correct) / float(pred_total)
            if (precision + recall == 0):
                continue
            f1score[j] += 2 * precision * recall / (precision + recall)
    f1score = f1score / counter
    return f1score

def per_class_acc_single(pred, ann, num_class):
    # pred: argmax output h x w numpy [0,C-1)
    # ann: class annotation h x w numpy [0,C-1)
    acc_arr = np.zeros(num_class)
    for c in range(num_class):
        idx = np.where(ann==c)
        idx_match = np.where(pred[idx]==c)
        acc = 1.0 * len(idx_match[0]) / len(idx[0])
        acc_arr[c] = acc
    return acc_arr

def per_class_acc_batch(output, target):
    # output: softmax output b x c x h x w tensor [0,1]
    # target: class annotation b x h x w tensor [0,C-1)
    batch_size = target.size(0)
    num_class = output.size(1)
    output, target = output.numpy(), target.numpy()
    output = np.argmax(output, axis=1)

    acc_arr_sum = np.zeros(num_class)
    for b in range(batch_size):
        pred = np.squeeze(output[b,])
        ann = np.squeeze(target[b,])
        acc_arr = per_class_acc_single(pred, ann, num_class)
        acc_arr_sum += acc_arr
    return acc_arr_sum/batch_size

########## rmse ##########
def per_image_rmse(pred, ann):
    # pred: N x L x 2 numpy
    # ann:  N x L x 2 numpy
    # rmse: N numpy 
    N = pred.shape[0]
    L = pred.shape[1]
    rmse = np.zeros(N)

    for i in range(N):
        pts_pred, pts_gt = pred[i,], ann[i,]
        if L == 7:
            interocular = np.linalg.norm(pts_gt[0,]-pts_gt[3,])
        elif L == 68:
            interocular = np.linalg.norm(pts_gt[36,]-pts_gt[45,])
        rmse[i] = np.sum(np.linalg.norm(pts_pred-pts_gt, axis=1))/(interocular*L)
    return rmse
 
def per_image_rmse_component(pred, ann):
    # pred: N x L x 2 numpy
    # ann:  N x L x 2 numpy
    # rmse: N numpy 
    N = pred.shape[0]
    L = pred.shape[1]
    rmse = np.zeros(N)
    rmse_le = np.zeros(N)
    rmse_re = np.zeros(N)
    rmse_ns = np.zeros(N)
    rmse_mt = np.zeros(N)

    for i in range(N):
        pts_pred, pts_gt = pred[i,], ann[i,]
        if L == 7:
            interocular = np.linalg.norm(pts_gt[0,]-pts_gt[3,])
        elif L == 68:
            interocular = np.linalg.norm(pts_gt[36,]-pts_gt[45,])
        rmse[i] = np.sum(np.linalg.norm(pts_pred-pts_gt, axis=1))/(interocular*L)
        if L == 7:
            rmse_le[i] = np.sum(np.linalg.norm(pts_pred[0:2,]-pts_gt[0:2,], axis=1))/(interocular*2)
            rmse_re[i] = np.sum(np.linalg.norm(pts_pred[2:4,]-pts_gt[2:4,], axis=1))/(interocular*2)
            rmse_ns[i] = np.sum(np.linalg.norm(pts_pred[4,]-pts_gt[4,], axis=1))/(interocular*1)
            rmse_mt[i] = np.sum(np.linalg.norm(pts_pred[5:7,]-pts_gt[5:7,], axis=1))/(interocular*2)
        elif L == 68:
            rmse_le[i] = np.sum(np.linalg.norm(pts_pred[36:42,]-pts_gt[36:42,], axis=1))/(interocular*6)
            rmse_re[i] = np.sum(np.linalg.norm(pts_pred[42:48,]-pts_gt[42:48,], axis=1))/(interocular*6)
            rmse_ns[i] = np.sum(np.linalg.norm(pts_pred[27:36,]-pts_gt[27:36,], axis=1))/(interocular*9)
            rmse_mt[i] = np.sum(np.linalg.norm(pts_pred[48:68,]-pts_gt[48:68,], axis=1))/(interocular*20)
    return rmse,rmse_le,rmse_re,rmse_ns,rmse_mt

def heatmap2pts(output, flag=None):
    # output: b x n x h x w
    res = [None]*2
    res[0] = output.size(3)
    res[1] = output.size(2)
    coords = Evaluation.get_preds(output)  # float type
    # coords -= 1
    coords_0 = coords.clone()
    # post-processing
    for n in range(coords.size(0)):
        for p in range(coords.size(1)):
            hm = output[n][p]
            px = int(math.floor(coords[n][p][0]))
            py = int(math.floor(coords[n][p][1]))
            if px > 1 and px < res[0] and py > 1 and py < res[1]:
                diff = torch.Tensor([hm[py - 1][px] - hm[py - 1][px - 2], hm[py][px - 1] - hm[py - 2][px - 1]])
                coords[n][p] += diff.sign() * .25
    coords_1 = coords.clone()
    coords += 0.5

    return coords_0, coords_1, coords