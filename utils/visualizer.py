# Xi Peng modify from CycleGAN, May 2017
import numpy as np
import os
import ntpath
import time
from . import util
from . import html
from pylib import HumanPts
from PIL import Image
import itertools

class Visualizer():
    def __init__(self, opt):
        self.opt = opt
        self.name = opt.exp_id
        self.use_visdom = opt.use_visdom
        self.use_html = opt.use_html
        self.win_size = opt.display_winsize
        self.log_name = ''
        if self.use_visdom:
            import visdom
            self.vis = visdom.Visdom(env=opt.vis_env)

        if self.use_html:
            self.web_dir = os.path.join(opt.exp_dir, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('=> create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
    
    def plot_train_history(self, history):
        """print options"""
        msg = ''
        for k,v in sorted( vars(self.opt).items() ):
            msg = msg + '%s: %s<br />' % (str(k), str(v))
        self.vis.text(text=msg, opts={'title':'options'}, win=0)

        """plot loss, acc, lr"""
        self.plot_value(history.epoch, history.loss, 'loss', 1)
        self.plot_value(history.epoch, history.pckh, 'pckh', 2)
        self.plot_value(history.epoch, history.lr, 'lr', 3)

    def plot_train_history_face(self, history):
        """print options"""
        msg = ''
        for k,v in sorted( vars(self.opt).items() ):
            msg = msg + '%s: %s<br />' % (str(k), str(v))
        self.vis.text(text=msg, opts={'title':'options'}, win=0)

        """plot loss, acc, lr"""
        self.plot_value(history.epoch, history.loss, 'loss', 1)
        self.plot_value(history.epoch, history.rmse, 'rmse', 2)
        self.plot_value(history.epoch, history.lr, 'lr', 3)

    def plot_value(self, epoch, value, title, win_id):
        # lr, epoch, loss, rmse (OrderedDict)
        # epoch = OrderedDict([('epoch',1)] )
        # loss = OrderedDict( [('train_loss',0.1),('val_loss',0.2)] )
       # if epoch==1:
       #     return
        e = np.array( [epoch[i].values() for i in range(len(epoch))] ).squeeze(1)
        v = np.array( [value[i].values() for i in range(len(value))] )
        l = list(value[0].keys())

        X,Y = np.stack([e]*len(l),1), v
        if Y.shape[1]==1:
            X,Y = X.squeeze(1), Y.squeeze(1)
        self.vis.line( X=X, Y=Y, opts={'title':title, 'legend':l}, win=win_id )

    def print_log(self, epoch, iter, total_iter, value1, value2=None):
        # value (OrderedDict)
       # if iter % self.opt.print_freq != 0:
       #     return
        msg = 'epoch:%d, iters:%d/%d ' % (epoch, iter, total_iter)
        for k, v in value1.items():
            msg += '%s: %.4f ' % (k, v)
        if value2:
            msg += '\n'
            for k, v in value2.items():
                msg += '%s:%.3f ' % (k, v)
        if iter == total_iter - 1:
            msg += '\n##########################################'
        print(msg)
        self.write_log(msg)

    def write_log(self, msg):
        # expr_dir = os.path.join(self.opt.exp_dir, self.opt.exp_id)
        # self.log_name = os.path.join(expr_dir, 'log.txt')
        with open(self.log_name, 'a+') as log_file:
            log_file.write(msg + '\n')
    
    def display_imgpts(self, img_batch_list, pts_batch_list, win_id):
	for img_batch, pts_batch in itertools.izip(img_batch_list, pts_batch_list):
	    img_plot_batch = np.zeros(img_batch.size())
            for b in range(0, img_batch.size(0)):
	        img_np = img_batch[b,:].mul(255.).numpy().astype('uint8')
                pts_np = pts_batch[b,:].numpy()
                img_pil = Image.fromarray(img_np.transpose([1,2,0]), 'RGB')
	        img_plot = HumanPts.DrawImgPts(img_pil, pts_np)
                img_plot = np.asarray(img_plot, dtype='uint8')
                img_plot_batch[b,] = img_plot.transpose([2,0,1])
            self.vis.images( img_plot_batch,
                        opts={'title':str(win_id)},
                        win=win_id )
            win_id += 1

	    	
    """TODO. visuals: dictionary of images to display or save"""
    def display_current_results(self, visuals, epoch):
        if self.display_id > 0: # show images in the browser
            idx = 1
            for label, image_numpy in visuals.items():
                #image_numpy = np.flipud(image_numpy)
                self.vis.image(image_numpy.transpose([2,0,1]), opts=dict(title=label),
                                   win=self.display_id + idx)
                idx += 1

        if self.use_html: # save images to a html file
            for label, image_numpy in visuals.items():
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                util.save_image(image_numpy, img_path)
            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, reflesh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims = []
                txts = []
                links = []

                for label, image_numpy in visuals.items():
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    """TODO. save image to the disk"""
    def save_images(self, webpage, visuals, image_path):
        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)
