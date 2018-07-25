# Zhiqiang Tang, May 2017
import numpy as np
import scipy.misc
import torch
from PIL import Image

# =============================================================================
# General image processing functions from Img.py
# =============================================================================
def GetTransform(center, scale, rot, res, size):
    # Generate transformation matrix
    h = size * scale # size_src = size_dst * scale
    t = np.zeros((3, 3))
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

def TransformSinglePts(pts, center, scale, rot, res, size, invert=0):
    t = GetTransform(center, scale, rot, res, size)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pts[0], pts[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int)

def TransformPts(pts, center, scale, rot, res, size, invert=0):
    NLMK, DIM = pts.shape
    t = GetTransform(center, scale, rot, res, size)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.concatenate( (pts, np.ones((NLMK,1))), axis=1 ).T
    new_pt = np.dot(t, new_pt)
    new_pt = new_pt[0:2,:].T
    return new_pt.astype(int)

def TransformImg(img, center, scale, rot, res, size):
    # ndim = img.ndim
    # if ndim == 2:
    #     img = np.expand_dims(img, axis=2)
    scale_factor = float(scale * size) / float(res)
    # height, width = img.shape[0], img.shape[1]
    if scale_factor < 2:
        scale_factor = 1
    else:
        new_img_size = np.floor(max(img.shape[0], img.shape[1]) / scale_factor)
        if new_img_size < 2:
            return img
        else:
            img = scipy.misc.imresize(img, size=1/scale_factor, interp='bilinear')
            # height, width = tmp_img.shape[0], tmp_img.shape[1]
    center = center / scale_factor
    scale = scale / scale_factor

    # Upper left point
    ul = np.array(TransformSinglePts([0,0], center, scale, 0, res, size, invert=1))
    # Bottom right point
    br = np.array(TransformSinglePts([res,res], center, scale, 0, res, size, invert=1))

    # make sure br - ul = (res, res)
    if scale_factor >= 2:
        br = br - (br - ul - res)

    # Padding so that when rotated proper amount of context is included
    pad = np.ceil(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2).astype(int)
    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if img.ndim > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape)

    # Range to fill new array
    ht = img.shape[0]
    wd = img.shape[1]
    new_x = max(0, -ul[0]), min(br[0], wd) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], ht) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(wd, br[0])
    old_y = max(0, ul[1]), min(ht, br[1])

    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]

    if not rot == 0:
        # Remove padding
        new_img = scipy.misc.imrotate(new_img, rot, interp='bilinear')
        new_img = new_img[pad:-pad, pad:-pad]

    # new_img = Image.fromarray(new_img.astype('uint8'), 'RGB')
    new_img = scipy.misc.toimage(new_img.astype('uint8'))
    # new_img.save('tmp.jpg')
    # return scipy.misc.imresize(new_img, (res,res,3))
    return new_img.resize((res,res), Image.BILINEAR)
# === End of Img.py === #

def crop(img, center, scale, rot, res, size):
    # ndim = img.ndim
    # if ndim == 2:
    #     img = np.expand_dims(img, axis=2)
    scale_factor = float(scale * size) / float(res)
    # height, width = img.shape[0], img.shape[1]
    if scale_factor < 2:
        scale_factor = 1
    else:
        new_img_size = np.floor(max(img.shape[0], img.shape[1]) / scale_factor)
        if new_img_size < 2:
            return img
        else:
            img = scipy.misc.imresize(img, size=1/scale_factor, interp='bilinear')
            # height, width = tmp_img.shape[0], tmp_img.shape[1]
    center = center / scale_factor
    scale = scale / scale_factor

    # Upper left point
    ul = np.array(TransformSinglePts([0,0], center, scale, 0, res, size, invert=1))
    # Bottom right point
    br = np.array(TransformSinglePts([res,res], center, scale, 0, res, size, invert=1))

    # make sure br - ul = (res, res)
    if scale_factor >= 2:
        br = br - (br - ul - res)

    # Padding so that when rotated proper amount of context is included
    pad = np.ceil(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2).astype(int)
    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if img.ndim > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape)

    # Range to fill new array
    ht = img.shape[0]
    wd = img.shape[1]
    new_x = max(0, -ul[0]), min(br[0], wd) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], ht) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(wd, br[0])
    old_y = max(0, ul[1]), min(ht, br[1])

    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]

    if not rot == 0:
        # Remove padding
        # print new_img.shape
        new_img = scipy.misc.imrotate(new_img, rot, interp='bilinear')
        new_img = new_img[pad:-pad, pad:-pad]

    # new_img = Image.fromarray(new_img.astype('uint8'), 'RGB')
    # new_img = scipy.misc.toimage(new_img.astype('uint8'))
    # new_img.save('tmp.jpg')
    return scipy.misc.imresize(new_img, (res, res))
    # return new_img.resize((res,res), Image.BILINEAR)
# === End of Img.py === #

def shuffle_channels_for_horizontal_flipping(maps, flip_indxs):
    # when the image is horizontally flipped, its corresponding groundtruth maps should be shuffled.
    # maps is a tensor of dimension n x c x h x w or c x h x w
    # flip_indxs = np.array([[1, 4], [0, 5], [12, 13], [11, 14], [10, 15], [2, 3]])
    if maps.ndimension() == 4:
        dim = 1
    elif maps.ndimension() == 3:
        dim = 0
    else:
        exit('tensor dimension is not right')

    for i in range(0, flip_indxs.shape[0]):
        idx1, idx2 = flip_indxs[i]
        tmp = maps.narrow(dim, idx1, 1).clone()
        maps.narrow(dim, idx1, 1).copy_(maps.narrow(dim, idx2, 1))
        maps.narrow(dim, idx2, 1).copy_(tmp)

    return maps

def flip_channels(maps):
    # horizontally flip the channels
    # maps is a tensor of dimension n x c x h x w or c x h x w
    if maps.ndimension() == 4:
        maps = maps.numpy()
        maps = maps[:, :, :, ::-1].copy()
    elif maps.ndimension() == 3:
        maps = maps.numpy()
        maps = maps[:, :, ::-1].copy()
    else:
        exit('tensor dimension is not right')

    return torch.from_numpy(maps).float()

def flip_back(flip_output, dataset='mpii'):
    """
    flip output map
    """
    if dataset ==  'mpii':
        matchedParts = (
            [0,5],   [1,4],   [2,3],
            [10,15], [11,14], [12,13]
        )
    else:
        print('Not supported dataset: ' + dataset)

    # flip output horizontally
    flip_output = fliplr(flip_output.numpy())

    # Change left-right parts
    for pair in matchedParts:
        tmp = np.copy(flip_output[:, pair[0], :, :])
        flip_output[:, pair[0], :, :] = flip_output[:, pair[1], :, :]
        flip_output[:, pair[1], :, :] = tmp

    return torch.from_numpy(flip_output).float()


def shufflelr(x, width, dataset='mpii'):
    """
    flip coords
    """
    if dataset ==  'mpii':
        matchedParts = (
            [0,5],   [1,4],   [2,3],
            [10,15], [11,14], [12,13]
        )
    elif dataset == 'face':
        matchedParts = (
            [0, 16], [1, 15], [2, 14], [3, 13], [4, 12], [5, 11], [6, 10], [7, 9], # outline
            [17, 26], [18, 25], [19, 24], [20, 23], [21, 22], # eyebrow
            [36, 45], [37, 44], [38, 43], [39, 42], [40, 47], [41, 46], # eye
            [31, 35], [32, 34], # nose
            [48, 54], [49, 53], [50, 52], [59, 57], [58, 56], # outer mouth
            [60, 64], [61, 63], [67, 65] # inner mouth
        )
    else:
        print('Not supported dataset: ' + dataset)

    # Flip horizontal
    x[:, 0] = width - x[:, 0]

    # Change left-right parts
    for pair in matchedParts:
        tmp = x[pair[0], :].clone()
        x[pair[0], :] = x[pair[1], :]
        x[pair[1], :] = tmp

    return x


def fliplr(x):
    if x.ndim == 3:
        x = np.transpose(np.fliplr(np.transpose(x, (0, 2, 1))), (0, 2, 1))
    elif x.ndim == 4:
        for i in range(x.shape[0]):
            x[i] = np.transpose(np.fliplr(np.transpose(x[i], (0, 2, 1))), (0, 2, 1))
    return x.astype(float)
    

if __name__=='__main__':
    print 'Face Augmentation by Xi Peng'

