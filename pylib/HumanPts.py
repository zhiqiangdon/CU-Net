# Zhiqiang Tang, May 2017
import os
import numpy as np
from PIL import Image, ImageDraw
import torch
from matplotlib.path import Path

############ read and write ##########
def ReadAnnotMPII(path):
    annot = {}
    with open(path, 'r') as fd:
        annot['imgName'] = next(fd).rstrip('\n')
        annot['headSize'] = float(next(fd).rstrip('\n'))
        annot['center'] = [int(x) for x in next(fd).split()]
        annot['scale'] = float(next(fd).rstrip('\n'))
        annot['pts'] = []
        annot['vis'] = []
        for line in fd:
            x, y, isVis = [int(float(x)) for x in line.split()]
            annot['pts'].append((x,y))
            annot['vis'].append(isVis)
    return annot

def DrawImgPts(img,pts):
    NLMK = pts.shape[0]
    img_draw = img.copy()
    draw = ImageDraw.Draw(img_draw)
    for l in range(NLMK):
        draw.ellipse((pts[l,0]-3,pts[l,1]-3,pts[l,0]+3,pts[l,1]+3), fill='white')
    del draw
    return img_draw


############ pts and heatmap ##########
def pts2heatmap(pts, heatmap_shape, sigma=1):
    # generate heatmap n x res[0] x res[1], each row is one pt (x, y)
    heatmap = np.zeros((pts.shape[0], heatmap_shape[0], heatmap_shape[1]))
    valid_pts = np.zeros((pts.shape))
    for i in range(0, pts.shape[0]):
        # if vis_arr[i] == -1 or pts[i][0] <= 0 or pts[i][1] <= 0 or \
        #                 pts[i][0] > heatmap_shape[1] or pts[i][1] > heatmap_shape[0]:
        #     continue
	if pts[i][0] <= 0 or pts[i][1] <= 0:
		continue
        heatmap[i] = draw_gaussian(heatmap[i], pts[i], sigma)
        valid_pts[i] = pts[i]
    return heatmap, valid_pts

def draw_gaussian(img, pt, sigma):
    # Draw a 2D gaussian
    tmp_size = np.ceil(3 * sigma)
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

def pts2heatmap_part(pts, heatmap_shape, sigma=1):
    part_index = np.array([[0, 1], [1, 2], [3, 4], [4, 5],
                           [10, 11], [11, 12], [13, 14], [14, 15], [6, 7], [8, 9]])
    heatmap = np.zeros((part_index.shape[0], heatmap_shape[0], heatmap_shape[1]))
    for i in range(0, part_index.shape[0]):
        pt1 = pts[part_index[i, 0]]
        pt2 = pts[part_index[i, 1]]
        if (pt1[0] <= 0 and pt1[1] <= 0) or (pt2[0] <= 0 and pt2[1] <= 0):
            continue
        part_pt = (pt1 + pt2) / 2.
        heatmap[i] = draw_gaussian(heatmap[i], part_pt, sigma)
    return heatmap

def heatmap2pts(heatmap):
    # heatmap: b x n x h x w tensor
    # preds: b x n x 2 tensor
    max, idx = torch.max(heatmap.view(heatmap.size(0), heatmap.size(1), heatmap.size(2) * heatmap.size(3)), 2)
    # print('hahahah')
    # print(max)
    # print(idx)
    pts = torch.zeros(idx.size(0), idx.size(1), 2)
    pts[:, :, 0] = idx % heatmap.size(3)
    # preds[:, :, 0].add_(-1).fmod_(heatmap.size(3)).add_(1)
    # idx is longTensor type, so no floor function is needed
    pts[:, :, 1] = idx / heatmap.size(3)
    pts[:, :, 1].floor_().add_(0.5)
    # preds[:, :, 1].div_(heatmap.size(3)).floor_()
    predMask = max.gt(0).repeat(1, 1, 2).float()
    # print(preds.size())
    # print(predMask.size())
    pts = pts * predMask
    # print(preds[:, :, 0])
    return pts

def pts2resmap(pts, resmap_shape, radius):
    # generate multi-channel resmap, one map for each point
    pts_num = pts.shape[0]
    resmap = np.zeros((pts_num, resmap_shape[0], resmap_shape[1]))
    valid_pts = np.zeros((pts.shape))
    for i in range(0, pts_num):
        # if vis_arr[i] == -1:
        #     continue
        # note that here we can't use vis_arr to indicate whether to draw the annotation
        # because some pts are labeled visible but not within the effective crop area due to the
        # inaccurate person scale in the original annotation
        if pts[i][0] <= 0 or pts[i][1] <= 0 or \
                        pts[i][0] > resmap_shape[1] or pts[i][1] > resmap_shape[0]:
            continue
        y, x = np.ogrid[-pts[i][1]:resmap_shape[0] - pts[i][1], -pts[i][0]:resmap_shape[1] - pts[i][0]]
        mask = x * x + y * y <= radius * radius
        resmap[i][mask] = 1
        valid_pts[i] = pts[i]
        # print('channel %d sum is %.f' % (i, np.sum(resmap[i])))
    return resmap, valid_pts

def weights_from_grnd_maps(maps, fgrnd_weight, bgrnd_weight):
    # maps: c x h x w tensor, zero is background, maps could be resmap or heatmap
    # weights: c x h x w tensor
    weights = torch.ones(maps.size())
    per_map_sum = maps.size(1) * maps.size(2)
    factor = float(fgrnd_weight) / float(bgrnd_weight)
    for i in range(0, maps.size(0)):
        mask_foregrnd = maps[i] > 0
        foregrnd_pixel_num = mask_foregrnd.sum()
        if foregrnd_pixel_num == 0:
            continue
        per_weight = float(per_map_sum - foregrnd_pixel_num) / float(foregrnd_pixel_num) * factor
        weights[i][mask_foregrnd] = int(per_weight)

    return weights

def pts2resmap_body_part(pts, resmap_shape, ann_size, vis_arr=None):
    part_index = np.array([[0, 1], [1, 2], [3, 4], [4, 5],
                           [10, 11], [11, 12], [13, 14], [14, 15], [8, 9]]) # , [6, 7]
    part_num = part_index.shape[0]
    resmap = np.zeros((part_num+1, resmap_shape[0], resmap_shape[1]))
    # pts_no_zero = pts[pts[:, 1] > 0]
    # body_height = max(pts_no_zero[:, 1]) - min(pts_no_zero[:, 1])
    # body_width = max(pts_no_zero[:, 0]) - min(pts_no_zero[:, 0])
    # body_len = max(body_height, body_width)
    for i in range(0, part_num):
        pt1 = pts[part_index[i, 0]].astype('float')
        pt2 = pts[part_index[i, 1]].astype('float')
        if vis_arr != None and (vis_arr[part_index[i, 0]] == 0 or vis_arr[part_index[i, 1]] == 0):
            continue
        if pt1[0] <= 0 or pt1[1] <= 0 or pt2[0] <= 0 or pt2[1] <= 0 or \
                        pt1[0] > resmap_shape[0] or pt1[1] > resmap_shape[0] or \
                        pt2[0] > resmap_shape[1] or pt2[1] > resmap_shape[1]:
            continue
        center = (pt1 + pt2) / 2.
        semi_major = np.linalg.norm(pt1 - pt2) / 2
        # semi_minor = body_len / 30
        if i in (0, 3):
            semi_minor = ann_size * 2
        elif i in (1, 2):
            semi_minor = ann_size * 2
        elif i in (4, 7):
            semi_minor = ann_size
        elif i in (5, 6):
            semi_minor = ann_size * 1.5
        elif i == 8:
            semi_minor = semi_major
        if semi_minor > semi_major * 2. / 3:
            semi_minor = semi_major * 2. / 3
        if semi_minor < semi_major * 1. / 3:
            semi_minor = semi_major * 1. / 3
        if i == 8:
            semi_minor = semi_major
        if semi_major < ann_size:
            semi_major = ann_size
        if semi_minor < ann_size:
            semi_minor = ann_size
        vector = pt1 - pt2
        angle = np.pi - np.arctan2(vector[1], vector[0])
        y, x = ellipse(center[1], center[0], semi_minor, semi_major, rotation=angle)
        y_in_bound_mask = (y < resmap_shape[0]) & (y >= 0)
        y = y[y_in_bound_mask]
        x = x[y_in_bound_mask]
        x_in_bound_mask = (x < resmap_shape[1]) & (x >= 0)
        x = x[x_in_bound_mask]
        y = y[x_in_bound_mask]
        resmap[i][y, x] = 1
    # draw the torso polygon
    if vis_arr ==None or (vis_arr != None and np.sum(vis_arr[np.r_[12, 13, 3, 2]]) == 0):
        vertices = pts[np.r_[12, 13, 3, 2]]
        less_zero_mask = vertices <= 0
        out_bound_mask = vertices >= resmap_shape[0]
        if np.sum(less_zero_mask) == 0 and np.sum(out_bound_mask) == 0:
            polygon_mask = polygon(vertices, resmap_shape)
            resmap[part_num][polygon_mask] = 1
    return resmap

def polygon(pts, img_shape):
    # codes = [Path.MOVETO,
    #          Path.LINETO,
    #          Path.LINETO,
    #          Path.LINETO,
    #          Path.CLOSEPOLY,
    #          ]
    x, y = np.meshgrid(np.arange(img_shape[0]), np.arange(img_shape[1]))
    x, y = x.flatten(), y.flatten()

    points = np.vstack((x, y)).T

    path = Path(pts)
    grid = path.contains_points(points)
    mask = grid.reshape((img_shape[0], img_shape[1]))
    # y, x = np.where(grid)
    return mask

def ellipse(r, c, r_radius, c_radius, shape=None, rotation=0.):
    """Generate coordinates of pixels within ellipse.
    Parameters
    ----------
    r, c : double
        Centre coordinate of ellipse.
    r_radius, c_radius : double
        Minor and major semi-axes. ``(r/r_radius)**2 + (c/c_radius)**2 = 1``.
    shape : tuple, optional
        Image shape which is used to determine the maximum extent of output pixel
        coordinates. This is useful for ellipses which exceed the image size.
        By default the full extent of the ellipse are used.
    rotation : float, optional (default 0.)
        Set the ellipse rotation (rotation) in range (-PI, PI)
        in contra clock wise direction, so PI/2 degree means swap ellipse axis
    Returns
    -------
    rr, cc : ndarray of int
        Pixel coordinates of ellipse.
        May be used to directly index into an array, e.g.
        ``img[rr, cc] = 1``.
    Examples
    --------
    # >>> from skimage.draw import ellipse
    # >>> img = np.zeros((10, 12), dtype=np.uint8)
    # >>> rr, cc = ellipse(5, 6, 3, 5, rotation=np.deg2rad(30))
    # >>> img[rr, cc] = 1
    # >>> img
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
           [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)
    Notes
    -----
    The ellipse equation::
        ((x * cos(alpha) + y * sin(alpha)) / x_radius) ** 2 +
        ((x * sin(alpha) - y * cos(alpha)) / y_radius) ** 2 = 1
    """

    center = np.array([r, c])
    radii = np.array([r_radius, c_radius])
    # allow just rotation with in range +/- 180 degree
    rotation %= np.pi

    # compute rotated radii by given rotation
    r_radius_rot = abs(r_radius * np.cos(rotation)) \
                   + c_radius * np.sin(rotation)
    c_radius_rot = r_radius * np.sin(rotation) \
                   + abs(c_radius * np.cos(rotation))
    # The upper_left and lower_right corners of the smallest rectangle
    # containing the ellipse.
    radii_rot = np.array([r_radius_rot, c_radius_rot])
    upper_left = np.ceil(center - radii_rot).astype(int)
    lower_right = np.floor(center + radii_rot).astype(int)

    if shape is not None:
        # Constrain upper_left and lower_right by shape boundary.
        upper_left = np.maximum(upper_left, np.array([0, 0]))
        lower_right = np.minimum(lower_right, np.array(shape[:2]) - 1)

    shifted_center = center - upper_left
    bounding_shape = lower_right - upper_left + 1

    rr, cc = _ellipse_in_shape(bounding_shape, shifted_center, radii, rotation)
    rr.flags.writeable = True
    cc.flags.writeable = True
    rr += upper_left[0]
    cc += upper_left[1]
    return rr, cc

def _ellipse_in_shape(shape, center, radii, rotation=0.):
    """ Generate coordinates of points within ellipse bounded by shape.
    Parameters
    ----------
    shape :  iterable of ints
        Shape of the input image.  Must be length 2.
    center : iterable of floats
        (row, column) position of center inside the given shape.
    radii : iterable of floats
        Size of two half axes (for row and column)
    rotation : float, optional
        Rotation of the ellipse defined by the above, in radians
        in range (-PI, PI), in contra clockwise direction,
        with respect to the column-axis.
    Returns
    -------
    rows : iterable of ints
        Row coordinates representing values within the ellipse.
    cols : iterable of ints
        Corresponding column coordinates representing values within the ellipse.
    """
    r_lim, c_lim = np.ogrid[0:float(shape[0]), 0:float(shape[1])]
    r_org, c_org = center
    r_rad, c_rad = radii
    rotation %= np.pi
    sin_alpha, cos_alpha = np.sin(rotation), np.cos(rotation)
    r, c = (r_lim - r_org), (c_lim - c_org)
    distances = ((r * cos_alpha + c * sin_alpha) / r_rad) ** 2 \
                + ((r * sin_alpha - c * cos_alpha) / c_rad) ** 2
    return np.nonzero(distances <= 1)

if __name__=='__main__':
    print 'Python pts to landmark by Xi Peng'

