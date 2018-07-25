# Zhiqiang Tang, May 2017
import torch

def weighted_sigmoid_crossentropy(pred, gt, weight):
    # pred: torch.sigmoid(output)
    # gt: 0/1 label map
    # weight: w=1 for gt==0, w>1 for gt==1
	loss = (gt * torch.log(pred+1e-6) + (1-gt) * torch.log(1-pred+1e-6))
	loss = loss * weight
	return -loss.sum()/loss.numel()

def weighted_L2(pred, gt, weight):
    # pred: net output
    # gt: [0,1] heatmap (gaussian circle)
    # weight: w=1 for gt==0, w>1 for gt>0
    loss = (pred - gt)**2
    loss = loss * weight
    return loss.sum() / loss.numel()
 
