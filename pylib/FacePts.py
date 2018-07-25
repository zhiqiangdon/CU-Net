# Xi Peng, Jan 2017
import os
import numpy as np
from PIL import Image, ImageDraw
import torch

############ read and write ##########
def ReadLmkFromTxt(path,format):
    ct = 0
    list = []
    for root, dirs, files in os.walk(path):
        for file in sorted(files):
            if file.endswith(format):
                lmk = np.loadtxt(root+file) # 68x2
                n,d = lmk.shape
                lmk = lmk.reshape((n*d))
                list.append(lmk)
                ct = ct + 1
                if ct == 10000:
                    return list
    return list

def ReadLmkFromTxtRecursive(path,format):
    ct = 0
    list = []
    for root, dirs, files in os.walk(path):
        for fold in dirs:
            files = os.listdir(root+fold)
            for file in sorted(files):
                if file.endswith(format):
                    lmk = np.loadtxt(root+fold+'/'+file) # 68x2
                    n,d = lmk.shape
                    lmk = lmk.reshape((n*d))
                    list.append(lmk)
                    ct = ct + 1
                    if ct == 10000:
                        return list
    return list

def DrawImgPts(img,pts):
	# img: Image
	# pts: N x 2 numpy
    NLMK = pts.shape[0]
    img_draw = img.copy()
    draw = ImageDraw.Draw(img_draw)
    for l in range(NLMK):
        draw.ellipse((pts[l,0]-3,pts[l,1]-3,pts[l,0]+3,pts[l,1]+3), fill='white')
    del draw
    return img_draw


############ pts vs. lmk ##########
def Pts2Lmk(fname):
    n_lmk = 68
    lmk = np.genfromtxt(fname, delimiter=' ', skip_header=3, skip_footer=1)
    return lmk

def Lmk68to7(lmk):
    lmk2 = np.zeros( (7,2) )
    lmk2[0] = lmk[37-1]
    lmk2[1] = lmk[40-1]
    lmk2[2] = lmk[43-1]
    lmk2[3] = lmk[46-1]
    lmk2[4] = lmk[31-1]
    lmk2[5] = lmk[49-1]
    lmk2[6] = lmk[55-1]
    return lmk2

def Lmk68to7_batch(lmk):
    bs = lmk.shape[0]
    lmk2 = np.zeros( (bs,7,2) )
    for b in range(bs):
        lmk2[b,:] = Lmk68to7(lmk[b,:])
    return lmk2

def GetCenterDist_68lmk(lmk):
    eyec = np.mean(lmk[36:48,:], axis=0)
    mouc = np.mean(lmk[48:60,:], axis=0)
    eyec_mouc_dist = np.sqrt(np.sum((eyec-mouc)**2))
    cx = int((eyec[0]+mouc[0]) / 2)
    cy = int((eyec[1]+mouc[1]) / 2)
    return (cx, cy, eyec_mouc_dist)

def GetCenterDist_7lmk(lmk):
    eyec = np.mean(lmk[0:4,:], axis=0)
    mouc = np.mean(lmk[5:7,:], axis=0)
    eyec_mouc_dist = np.sqrt(np.sum((eyec-mouc)**2))
    cx = int((eyec[0]+mouc[0]) / 2)
    cy = int((eyec[1]+mouc[1]) / 2)
    return (cx, cy, eyec_mouc_dist)

def Lmk2Bbox_7lmk(lmk, DISTRATIO):
    cx,cy,dist = GetCenterDist_7lmk(lmk)
    sl = int(dist * DISTRATIO)
    bbox = (cx-sl/2, cy-sl/2, cx+sl/2, cy+sl/2) # left, top, right, bottom 
    return bbox



########## resmap ##########
def Lmk2Resmap_mc(pts, resmap_shape, radius):
    # generate multi-channel resmap, one map for each point
    pts_num = pts.shape[0]
    resmap = np.zeros((pts_num, resmap_shape[1], resmap_shape[0]))
    for i in range(0, pts_num):
        y, x = np.ogrid[-pts[i][1]:resmap_shape[1] - 
						pts[i][1], -pts[i][0]:resmap_shape[0] - pts[i][0]]
        mask = x * x + y * y <= radius * radius
        resmap[i][mask] = 1
        # print('channel %d sum is %.f' % (i, np.sum(resmap[i])))
    return resmap

def Lmk2Resmap(lmk, shape, circle_size):
    #RADIUS = GetCircleSize_L128_R4(scale)
    RADIUS = circle_size
    resmap = Image.new('L', shape)
    draw = ImageDraw.Draw(resmap)
    for l in range(lmk.shape[0]):
        draw.ellipse((lmk[l,0]-RADIUS,lmk[l,1]-RADIUS,
					  lmk[l,0]+RADIUS,lmk[l,1]+RADIUS), fill=l+1)
    del draw
    return resmap

def Resmap2Lmk(resmap, NLMK):
    # resmap: h x w numpy [0,NLMK)
    lmk = np.zeros((NLMK, 2))
    for l in range(NLMK):
        try:
            y,x = np.where(resmap == l+1)
            yc,xc = np.mean(y), np.mean(x)
            lmk[l,:] = [xc+1, yc+1]
        except:
            print('Not found %d-th landmark' % l)
    return lmk

def Resmap2Lmk_batch(resmap):
    # resmap: softmax output b x c x h x w pytorch tensor [0,1]
    # pts: b x num_pts x 2 numpy
    batch_size, num_class = resmap.size(0), resmap.size(1)
    num_lmk = num_class - 1

    resmap = np.argmax(resmap.numpy(), axis=1) # b x h x w
    lmk = np.zeros((batch_size, num_lmk, 2))
    for b in range(batch_size):
        resmap = np.squeeze(resmap[b,]) # h x w
        lmk[b,] = Resmap2Lmk(resmap, num_lmk)
    return lmk   


def GetCircleSize_L128_R4(scale):
    size = np.round( 4 / scale )
    if size<2:
        size = 2
    elif size>5:
        size = 5
    return size

def CircleSize(base_size=4, scale=1):
    size = np.round( base_size / scale) # L128-R4
    size = size-2 if size<base_size-2 else size
    size = size+2 if size>base_size+2 else size
    return size

def GtMap2WeightMap(gt_map, reduce_factor=0.5):
    # GtMap: c x h x w tensor, could be resmap or heatmap
    # WeightMap: c x h x w tensor
    weight_map = np.ones(gt_map.shape)
    per_map_sum = gt_map.shape[1] * gt_map.shape[2]
    for i in range(0, gt_map.shape[0]):
        mask_foregrnd = gt_map[i] > 0
        foregrnd_pixel_num = mask_foregrnd.sum()
        if foregrnd_pixel_num == 0:
            continue
        per_weight = float(per_map_sum - foregrnd_pixel_num) / float(foregrnd_pixel_num) * reduce_factor
        weight_map[i][mask_foregrnd] = int(per_weight)
    return weight_map



########## heatmap ##########
def draw_gaussian(img, pt, sigma):
    # Draw a 2D gaussian
    tmp_size = np.round(3 * sigma)
    ul = [int(pt[0] - tmp_size), int(pt[1] - tmp_size)]
    br = [int(pt[0] + tmp_size), int(pt[1] + tmp_size)]
    # Check that any part of the gaussian is in-bounds
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return img

    # Generate gaussian
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (tmp_size ** 2))

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0]+1, img.shape[1]) - max(0, ul[0]) + max(0, -ul[0])
    g_y = max(0, -ul[1]), min(br[1]+1, img.shape[0]) - max(0, ul[1]) + max(0, -ul[1])
    # Image range
    img_x = max(0, ul[0]), min(br[0]+1, img.shape[1])
    img_y = max(0, ul[1]), min(br[1]+1, img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img

def Lmk2Heatmap(pts, res, sigma=1):
    # generate heatmap n x res[0] x res[1], each row is one pt (x, y)
    heatmap = np.zeros((pts.shape[0], res[0], res[1]))
    for i in range(0, pts.shape[0]):
        if pts[i,0]>0 and pts[i,0]<=res[1] and pts[i,1]>0 and pts[i,1]<=res[0]:
            heatmap[i] = draw_gaussian(heatmap[i], pts[i,], sigma)
    return heatmap

def Heatmap2Lmk_batch(heatmap):
    # heatmap: b x c x h x w torch tensor [0,1]
    # lmk: b x num_pts x 2 numpy
    b,c,h,w = heatmap.size()
    max_score,idx = torch.max(heatmap.view(b,c,h*w), 2)
    lmk = idx.repeat(1,1,2).float()
    lmk[:,:,0] = lmk[:,:,0] % w
    lmk[:,:,1] = (lmk[:,:,1] / w).floor()
    mask = max_score.gt(0).repeat(1,1,2).float()
    #lmk = lmk.mul(mask).add(1)
    lmk = lmk.add(1)
    return lmk.numpy()

def Heatmap2Lmk(heatmap):
    # heatmap: NLMK x h x w numpy [0,1]
    # lmk: NLMK x 2 numpy
    NLMK = heatmap.shape[0]
    lmk = np.zeros((NLMK, 2))
    for l in range(NLMK):
        y,x = np.unravel_index(heatmap[l,].argmax(), heatmap[l,].shape)
        lmk[l,:] = [x+1, y+1]
    return lmk

def Heatmap2Lmk_batch_for(heatmap):
    # heatmap: b x num_pts x h x w torch tensor [0,1]
    # lmk: b x num_pts x 2 numpy
    batch_size, num_lmk = heatmap.size(0), heatmap.size(1)
    
    lmk = np.zeros((batch_size, num_lmk, 2))
    for b in range(batch_size):
        heatmap1 = heatmap[b,].numpy() # c x h x w
        lmk[b,] = FacePts.Heatmap2Lmk(heatmap1)
    return lmk


if __name__=='__main__':
    print 'Python pts to landmark by Xi Peng'

