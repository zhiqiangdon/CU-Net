import torch
import numpy as np
import math
import HumanAug

def get_preds(scores):
    ''' get predictions from score maps in torch Tensor
        return type: torch.LongTensor
    '''
    assert scores.dim() == 4, 'Score maps should be 4-dim'
    maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2)

    maxval = maxval.view(scores.size(0), scores.size(1), 1)
    idx = idx.view(scores.size(0), scores.size(1), 1) + 1

    preds = idx.repeat(1, 1, 2).float()

    preds[:,:,0] = (preds[:,:,0] - 1) % scores.size(3) + 1
    preds[:,:,1] = torch.floor((preds[:,:,1] - 1) / scores.size(2)) + 1

    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    preds *= pred_mask
    return preds

def calc_dists(preds, target, normalize, use_zero=False):
    preds = preds.float()
    target = target.float()
    normalize = normalize.float()
    dists = torch.zeros(preds.size(1), preds.size(0))
    if use_zero:
        boundary = 0
    else:
        boundary = 1
    for n in range(preds.size(0)):
        for c in range(preds.size(1)):
            if target[n,c,0] > boundary and target[n, c, 1] > boundary:
                dists[c, n] = torch.dist(preds[n,c,:], target[n,c,:])/normalize[n]
            else:
                dists[c, n] = -1
    return dists

def dist_acc(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    # print 'dists: ', dists
    # print 'dists.ne(-1): ', dists.ne(-1)
    # print 'dists.ne(-1).sum(): ', dists.ne(-1).sum()
    # print 'dists.le(thr): ', dists.le(thr)
    # print 'dists.le(thr).eq(dists.ne(-1)): ', dists.le(thr).eq(dists.ne(-1))
    # print 'dists.le(thr).eq(dists.ne(-1)).sum(): ', dists.le(thr).eq(dists.ne(-1)).sum()
    if dists.ne(-1).sum() > 0:
        return dists.le(thr).eq(dists.ne(-1)).sum().float() / dists.ne(-1).sum().float()
    else:
        return -1

def accuracy(output, target, idxs, thr=0.5):
    ''' Calculate accuracy according to PCK, but uses ground truth heatmap rather than x,y locations
        First value to be returned is average accuracy across 'idxs', followed by individual accuracies
    '''
    preds   = get_preds(output)
    # print preds
    gts     = get_preds(target)
    # print gts
    # exit()
    norm    = torch.ones(preds.size(0))*output.size(3)/10
    # print norm
    dists   = calc_dists(preds, gts, norm)
    # print dists
    # exit()
    acc = torch.zeros(len(idxs)+1)
    avg_acc = 0
    cnt = 0

    for i in range(len(idxs)):
        acc[i+1] = dist_acc(dists[idxs[i]])
        if acc[i+1] >= 0: 
            avg_acc = avg_acc + acc[i+1]
            cnt += 1
            
    if cnt != 0:  
        acc[0] = avg_acc / cnt
    # print acc
    # exit()
    return acc


def accuracy_origin_res(output, center, scale, res, grnd_pts, normalizers, rot):
    ''' Calculate accuracy according to PCK, but uses ground truth heatmap rather than x,y locations
        First value to be returned is average accuracy across 'idxs', followed by individual accuracies
    '''
    idxs = torch.LongTensor([0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15])
    pred_pts = final_preds(output, center, scale, res, rot)
    dists = calc_dists(pred_pts, grnd_pts, normalizers, use_zero=True)

    acc = torch.zeros(len(idxs) + 1)
    avg_acc = 0
    cnt = 0

    for i in range(len(idxs)):
        acc[i + 1] = dist_acc(dists[idxs[i]])
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    if cnt != 0:
        acc[0] = avg_acc / cnt
    return acc

def final_preds(output, center, scale, res, rot):
    coords = get_preds(output) # float type

    # pose-processing
    for n in range(coords.size(0)):
        for p in range(coords.size(1)):
            hm = output[n][p]
            px = int(math.floor(coords[n][p][0]))
            py = int(math.floor(coords[n][p][1]))
            if px > 1 and px < res[0] and py > 1 and py < res[1]:
                diff = torch.Tensor([hm[py - 1][px] - hm[py - 1][px - 2], hm[py][px - 1]-hm[py - 2][px - 1]])
                coords[n][p] += diff.sign() * .25
    coords += 0.5
    preds = coords.clone()

    # Transform back
    # print coords.size(), len(center), len(scale)
    for i in range(coords.size(0)):
        # print type(coords[i]), type(center[i]), type(scale[i])
        preds[i] = transform_preds(coords[i], center[i], scale[i], res, rot[i])

    if preds.dim() < 3:
        preds = preds.view(1, preds.size())

    return preds

def transform_preds(coords, center, scale, res, rot):
    # size = coords.size()
    # coords = coords.view(-1, coords.size(-1))
    # print(coords.size())
    coords = coords.numpy()
    # print type(coords), type(center), type(scale)
    # exit()
    center = center.numpy()
    scale = scale.numpy()
    rot = rot.numpy()
    coords = TransformPts(coords, center, scale, rot, res[0], size=200, invert=1)
    # exit()
    coords = torch.from_numpy(coords)
    # for p in range(coords.size(0)):
    #     # coords[p, 0:2] = torch.from_numpy(transform(coords[p, 0:2], center, scale, res, 1, 0))

    return coords
    
def GetTransform(center, scale, rot, res, size):
    # Generate transformation matrix
    h = size * scale # size_src = size_dst * scale
    t = np.zeros((3, 3))
    # print res, float(res), type(res), float(res) / h
    t[0, 0] = float(res) / h
    t[1, 1] = float(res) / h
    t[0, 2] = res * (-float(center[0]) / h + .5)
    t[1, 2] = res * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot # To match direction of rotation from cropping
        rot_mat = np.zeros((3,3))
        rot_rad = rot * np.pi / 180
        sn,cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0,:2] = [cs, -sn]
        rot_mat[1,:2] = [sn, cs]
        rot_mat[2,2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0,2] = -res/2
        t_mat[1,2] = -res/2
        t_inv = t_mat.copy()
        t_inv[:2,2] *= -1
        t = np.dot(t_inv,np.dot(rot_mat,np.dot(t_mat,t)))
    return t

def TransformPts(pts, center, scale, rot, res, size, invert=0):
    NLMK, DIM = pts.shape
    t = GetTransform(center, scale, rot, res, size)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.concatenate( (pts - 1, np.ones((NLMK,1))), axis=1 ).T
    new_pt = np.dot(t, new_pt)
    new_pt = new_pt[0:2,:].T
    return new_pt.astype(int) + 1
