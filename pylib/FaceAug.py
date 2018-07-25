# Xi Peng, Jan 2017
import numpy as np
import scipy.misc
import math
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
    # Upper left point
    ul = np.array(TransformSinglePts([0,0], center, scale, 0, res, size, invert=1))
    # Bottom right point
    br = np.array(TransformSinglePts([res,res], center, scale, 0, res, size, invert=1))

    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape, dtype=np.uint8)

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
        new_img = scipy.misc.imrotate(new_img, rot)
        new_img = new_img[pad:-pad, pad:-pad]

    new_img = Image.fromarray(new_img.astype('uint8'), 'RGB')
    return new_img.resize((res,res), Image.ANTIALIAS)
# === End of Img.py === #

def Gray2RGB(img):
    # img: PIL Image
    img_np = np.asarray(img)
    img_rgb = np.dstack([img_np] * 3)
    return Image.fromarray(img_rgb, 'RGB')

def GetFaceBbox(pts):
    NLMK,dim = pts.shape
    if dim != 2:
        pts = pts.T
    ptx = np.squeeze( np.asarray(pts[:,0]) )
    pty = np.squeeze( np.asarray(pts[:,1]) )
    ptx = ptx[np.where(ptx>0)]
    pty = pty[np.where(pty>0)]

    centerx = (min(ptx)+max(ptx)) / 2
    centery = (min(pty)+max(pty)) / 2
    sl = max(max(ptx)-min(ptx), max(pty)-min(pty))
    bbox = [centerx-sl/2, centerx+sl/2, centery-sl/2, centery+sl/2] # l,r,b,t
    bbox = np.round(bbox).astype(int)
    return bbox

def AugImgPts(img, pts, res_dst, size_dst, scale, rot):
    # scale: [0.7, 1.3]
    # rot: [-30, 30]
    bbox_src = GetFaceBbox(pts)
    center_src = ( 0.5*(bbox_src[1]+bbox_src[0]), 0.5*(bbox_src[3]+bbox_src[2]) )
    size_src = bbox_src[1] - bbox_src[0]

    NLMK,_ = pts.shape
    if NLMK==68:
        base_scale = (1.5 * size_src) / size_dst
    elif NLMK==7:
        base_scale = (2.5 * size_src) / size_dst
    scale_aug = base_scale * scale
    rot_aug = rot

    # scale: input face size / output face size (augment)
    img_aug = TransformImg(img, center_src, scale_aug, rot_aug, res_dst, size_dst)
    pts_aug = TransformPts(pts, center_src, scale_aug, rot_aug, res_dst, size_dst)

    return img_aug, pts_aug
    

if __name__=='__main__':
    print 'Face Augmentation by Xi Peng'

