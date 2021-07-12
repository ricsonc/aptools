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
from skimage.color.colorconv import convert_colorspace
from skimage import color

from torch.nn import functional as F
import imageio as ii

class Postprocessor:
    def __init__(
        self,
        work_dir,
        params = M(
            gradient_window = 32+1, 
            dilation = 16, 
            gradient_max = 90, 
            excl_box = None,
            mask_file = None,
            border_crop = 400,
            tone_curve = [
                (0.05, -0.02),
                (0.3, 0.0),
            ],
            curve_type = 'thin-plate',
            output_box = None,
            output_border = 400,
        )):

        self.work_dir = work_dir
        for k in params:
            setattr(self, k, params[k])

    def subtract_gradient(self, x_): 

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
        #median_filter does not deal with nans at all, so we need to do it w/ generic_filter
        medians = np.stack([
            generic_filter(x_dila[...,c], np.nanmedian, size=self.gradient_window, mode='mirror')
            for c in range(3)
        ],axis=-1)
        
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
        # imshow( np.isnan(x).astype(np.float32) )
        # st()
        
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
            
        elif True:
            med_num = np.zeros_like(median_flat)
            med_denom = np.zeros_like(median_flat)
            avg_iters = 200
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

            H,W,_ = x_.shape
            def basis_fn(yx):
                y = yx[...,0] - H/2.0
                x = yx[...,1] - W/2.0
                ones = np.ones(y.shape)
                #return np.stack((x, y, x**2, y**2, x*y, ones), axis=-1)
                return np.stack((x**2, y**2, x*y, ones), axis=-1)

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


        if input('[y] to visualize gradient ') == 'y':
            ax = plt.axes()
            for c in range(3):
                ax.contour(out[...,c], levels=20, colors=['red', 'green', 'blue'][c])
            imshow_norm(out, ax=ax)
        #st()

        # x_ = utils.blur(x_, 10.0)
        # SUBTRACTION MODEL
        #0.02 bkg
        #don't clip too much
        rval = np.clip(x_ - out, 0.0, 1.0)
        #rval = np.clip(x_, 0,1) #do nothing

        if np.isnan(rval).max() > 0:
            print('detected nans, replacing with 0')
            rval = np.nan_to_num(rval, 0.0) #<- only necessary in rare cases...
        
        return rval / rval.max(axis=(0,1)) # renormalize per channel

    def global_tonemap(self, x, ha = None, ha_frac=0.4):
        
        #5 and 99.5
        lb = 1 #5 for brighter skies, 1 for darker
        ub = 99.9 #adjust 99.5 also works
        # gamma adjustment
        x = utils.renormalizep(x, lb, ub) #<- very interesting...5 works, 10??
        xg = utils.transfer_function(x)

        if ha is not None:
            halb = 5
            haub = 99.5
            
            ha = utils.renormalizep(ha, halb, haub)
            ha *= 1.0 #tune at will, 0.25 works
            hag = utils.transfer_function(ha)
            
            x_xyz = color.rgb2lab(xg)
            
            #convert to monochrome with this weird weighting first..
            hag_mono = (hag * np.array([0.5, 0.25, 0.25])).sum(axis=-1)
            hag_mono = np.stack([hag_mono]*3, axis=-1)
            halum = color.rgb2lab(hag_mono)
            x_xyz[...,0] = np.maximum(0.5 * x_xyz[...,0], halum[...,0]) #tune mixture at will
            xg = color.lab2rgb(x_xyz)
            xg = xg*(1-ha_frac) + hag*ha_frac
        
        xg = utils.renormalizep(xg, lb, ub)

        xt = xg
        # this was a bad idea
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

        if input('[y] to show entire output: ') == 'y':
            imshow(xc)

        #uhhh need better output spec
        if self.output_box is not None:
            miny, maxy, minx, maxx = self.output_box
            xc_crop = xc[miny:maxy,minx:maxx]
        elif self.output_border is not None and self.excl_box is not None:
            miny, maxy, minx, maxx = self.excl_box
            xc_crop = xc[
                miny-self.output_border:maxy+self.output_border,
                minx-self.output_border:maxx+self.output_border
            ]
        else:
            xc_crop = xc
        
        print('showing cropped output')
        imshow(xc_crop)
        if input('[y] to save: ') == 'y':
            tifffile.imwrite(f'{self.work_dir}/post.tiff', xc_crop)
            
        return xc
        
    def star_shrink(self, x):
        return x
    
    def __call__(self, path):
        dst_path = path.replace('out.tiff', 'nograd.tiff')
        
        if os.path.exists(dst_path) and input('[y] to use cached bkg: ') == 'y':
            x = tifffile.imread(dst_path)
        else:
            x = tifffile.imread(path)
            c = self.border_crop
            x = x[c:-c,c:-c] #crop out some of the unstacked area
            x = self.subtract_gradient(x) #uhh
            tifffile.imwrite(dst_path, x)

        x = self.global_tonemap(x)
        # if False:
        #     assert 'visrosette' in path
        #     y = tifffile.imread(dst_path.replace('visrosette','rosette'))
        #     x = self.global_tonemap(x, ha=y)
        # elif False:
        #     assert 'visorion' in path
        #     y = tifffile.imread(dst_path.replace('visorion','oneb'))
        #     x = self.global_tonemap(x, ha=y)
            
        x = self.star_shrink(x)
