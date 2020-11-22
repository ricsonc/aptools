import numpy as np
import scipy as sp
from ipdb import set_trace as st
import skimage as ski
import utils
from matplotlib import pyplot as plt
import matplotlib as mpl
from common import *
from munch import Munch as M
from scipy import sparse
from scipy.interpolate import Rbf, griddata
from scipy.ndimage import median_filter, minimum_filter, generic_filter
import os
from scipy.fftpack import fftn, ifftn
from tqdm.contrib.itertools import product
from tqdm import tqdm
import tifffile
import torch
from skimage import color
from torch.nn import functional as F
import imageio as ii

class Postprocessor:
    def __init__(
        self,
        work_dir,
        params = M(
            gradient_window = 32+1, #should be odd
            dilation = 16, #computational savings...
            gradient_max = 95, #max percentile of light pollution
            # excl_box = [600, 2800, 2000, 3600], #miny, maxy, minx, maxx

            # excl_box = [500, 2400, 1200, 2300], #<- andro2
            # excl_box = [1950, 2600, 3000, 3750], <- tri
            # excl_box = [ 750, 1600, 2150, 3100], #hatri
            # excl_box = [1500, 3000, 0, 1000], #haorion
            excl_box = [700, 1900, 2000, 3800],
            # excl_box = None,
            
            #mask_file = 'mask.png', #or None
            mask_file = None,
            
            gradient_iters = 5,
            outlier_factor = 3.0,

            border_crop = 400,

            tone_curve = [
                (0.05, -0.02),
                (0.3, 0.0),
            ],
            curve_type = 'thin-plate' #cubic less prone to overshoot
            
        )):

        self.work_dir = work_dir
        for k in params:
            setattr(self, k, params[k])

    def subtract_gradient(self, x_): #what if we just subtract the linterpolation?

        x = x_.copy()

        lum = utils.luminance(x)[...,0]
        lum_thresh = np.percentile(lum, self.gradient_max)
        x[lum > lum_thresh,:] = np.nan #everything over this is likely not bkg

        if self.excl_box is not None:
            miny, maxy, minx, maxx = self.excl_box
        else:
            miny = minx = 1000000
            maxx = maxy = 0
        x[miny:maxy, minx:maxx] = np.nan

        if self.mask_file is not None:
            mask_img = ii.imread(f'{self.work_dir}/{self.mask_file}')
            mask_img = mask_img[...,0] > 128
            mask_img = mask_img.astype(np.float64)
            mask_img[mask_img < 0.5] = np.nan
            x *= mask_img[...,None]
            
        # due to computational constraints, compute an "atrous median"
        x_dila = x[::self.dilation,::self.dilation]
        medians = np.stack( #nasty source of bug!!! median_filter does not deal with nans at all...
            [generic_filter(x_dila[...,c], np.nanmedian, size=self.gradient_window, mode='mirror') for c in range(3)],
            axis=-1
        )
        
        mxsd, mysd = np.meshgrid(np.arange(x_dila.shape[1]), np.arange(x_dila.shape[0]))
        mxs, mys = mxsd * self.dilation, mysd * self.dilation

        validity = ~np.isnan(medians.max(axis=-1))
        validity_flat = validity.reshape(-1)

        yx = np.stack((mys, mxs), axis = -1)
        yx_flat = yx.reshape(-1,2)[validity_flat]
        median_flat = medians.reshape(-1,3)[validity_flat]

        #med_max = medians.copy()
        #med_max[np.isnan(medians)] = 1000.0
        #imshow((medians-med_max.min(axis=(0,1)))*400)
        
        xs, ys = np.meshgrid(np.arange(x.shape[1]), np.arange(x.shape[0]))
        yxs = np.stack((ys, xs), axis = -1).reshape(-1,2)

        ### begin interp

        if False:
            #what to do about nans...
            fallback = griddata(
                yx_flat,
                median_flat,
                yxs, 
                method='nearest'
            )

            avg_iters = 100
            dropout_prob = 0.01

            print('computing average gradient')
            out = 0
            for i in tqdm(range(avg_iters)):
                mask = np.random.random(yx_flat.shape[0]) < dropout_prob
                out_ = griddata(
                    yx_flat[mask],
                    median_flat[mask],
                    yxs,
                    method='linear'
                )
                out += np.where(
                    np.isnan(out_),
                    fallback,
                    out_
                )
            out /= avg_iters
            
        elif False:
            med_num = np.zeros_like(median_flat)
            med_denom = np.zeros_like(median_flat)
            avg_iters = 1000
            dropout_prob = 0.05

            print('computing average gradient')
            for i in tqdm(range(avg_iters)):
                mask = np.random.random(yx_flat.shape[0]) < dropout_prob
                out_ = griddata(
                    yx_flat[mask],
                    median_flat[mask],
                    yx_flat[~mask],
                    method='linear'
                )
                med_num[~mask] += out_
                med_denom[~mask] += ~np.isnan(out_)

            med_refit = med_num/med_denom

            fallback = griddata(
                yx_flat,
                median_flat,
                yxs, 
                method='nearest'
            )

            out = griddata(
                yx_flat,
                med_refit,
                yxs, 
                method='linear'
            )
            out = np.where(np.isnan(out), fallback, out)
        else:
            #fit quadratic...

            def basis_fn(yx):
                y = yx[...,0]
                x = yx[...,1]
                ones = np.ones(y.shape)
                return np.stack((x, y, x**2, y**2, x*y, ones), axis=-1)

            T, pred = utils.fit_transform_linear(basis_fn(yx_flat), median_flat)
            res = median_flat-pred

            resstd = res.std(axis=0)
            print('first pass, resstd is', resstd)
            res_clip = res.copy()
            for i in range(3):
                res_clip[...,i] = np.clip(res_clip[...,i], -resstd[i], resstd[i])
            median_flat_clip = pred + res_clip

            T, res = utils.fit_linear(basis_fn(yx_flat), median_flat_clip)
            print('second pass, resstd is', res.std(axis=0))

            #gen out
            out = basis_fn(yxs) @ T

        ### end interp
        out[np.isnan(out)] = out[~np.isnan(out)].mean()

        out = out.reshape(x.shape)

        #i should be dividing actually...
        #this might be helpful for debugging
        # for c in range(3): plt.axes().contour(out[...,c], levels=20, colors=['red', 'green', 'blue'][c])
        # imshow_norm(out)
        # st()

        # x_ = utils.blur(x_, 10.0)
        # SUBTRACTION MODEL
        #0.02 bkg
        #don't clip too much
        rval = np.clip(x_ - out, 0.0, 1.0) 
        return rval / rval.max(axis=(0,1)) # renormalize per channel

    def global_tonemap(self, x):

        #5 and 99.5
        lb = 10
        ub = 99.5
        # gamma adjustment
        x = utils.renormalizep(x, lb, ub) #<- very interesting...5 works, 10??
        # xg = x**(1/self.gamma)
        xg = utils.transfer_function(x)
        
        xg = utils.renormalizep(xg, lb, ub)

        xt = xg
        # oops this was a bad idea
        # # reinhard but with luma
        # lum = utils.luminance(x) #luma, not luminance!
        # lum_tone = 2*lum/(lum+1)
        # scale_factor = lum_tone/(lum+1E-8)
        # #since we're operating in luma space, we need to clip a bit
        # xt = np.clip(xg * scale_factor, 0, 1)
        
        # spline curve to stretch out things a bit
        curve_x, curve_y = zip(*self.tone_curve)
        curve_x = np.array((0,)+curve_x+(1,))
        curve_y = np.array((0,)+curve_y+(0,))
        curve_ = Rbf(curve_x, curve_y, function=self.curve_type)
        curve = lambda x: np.clip(curve_(x)+x, 0.0,1.0)
        # plt.plot(curve(np.linspace(0,1,100))); plt.show()
        xc = curve(xt)

        st()

        border = 400
        miny, maxy, minx, maxx = self.excl_box
        xc_crop = xc[miny-border:maxy+border, minx-border:maxx+border]
        # imshow(xc)
        st() #<- to tiff?

        return xc
        
    def star_shrink(self, x):
        '''
        uhhh probably want to denoise here as well
        '''
        
        return x
    
    def __call__(self, path):
        x = tifffile.imread(path)
        c = self.border_crop
        x = x[c:-c,c:-c] #crop out some of the unstacked area
        
        x = self.subtract_gradient(x) #uhh
        x = self.global_tonemap(x)
        
        x = self.star_shrink(x)
