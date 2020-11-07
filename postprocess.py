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

class Postprocessor:
    def __init__(
        self,
        work_dir,
        params = M(
            gradient_window = 32+1, #should be odd
            dilation = 8, #computational savings...
            gradient_max = 95, #max percentile of light pollution
            # excl_box = [600, 2800, 2000, 3600], #miny, maxy, minx, maxx

            # excl_box = [500, 2400, 1200, 2300], #<- andro2
            excl_box = [1950, 2600, 3000, 3750],
            
            # excl_box = None,
            gradient_iters = 5,
            outlier_factor = 3.0,
            gamma = 2.0,

            border_crop = 200,

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
        
        if False:

            #a quadratic seems to be a decent way to estimate gradients..
            # basis_fn = lambda z: cat((np.ones((z.shape[0],1)), z, z**2), axis=-1) #need to fix...
            basis_fn = lambda z: cat((np.ones((z.shape[0],1)), z), axis=-1) #need to fix...
            basis = basis_fn(yx_flat)
            T, residuals = utils.fit_linear(basis, median_flat)

            valid = np.ones((yx_flat.shape[0]), dtype = np.bool)

            for i in range(self.gradient_iters):
                T, residuals = utils.fit_linear(basis[valid], median_flat[valid])
                residuals = np.linalg.norm(residuals, axis = -1)
                valid[valid] &= residuals < np.median(residuals)*self.outlier_factor

            #transform the entire image
            xs, ys = np.meshgrid(np.arange(x.shape[1]), np.arange(x.shape[0]))
            yxs = np.stack((ys, xs), axis = -1).reshape(-1,2)
            out = basis_fn(yxs) @ T
            out = out.reshape(x.shape)

        elif True:

            xs, ys = np.meshgrid(np.arange(x.shape[1]), np.arange(x.shape[0]))
            yxs = np.stack((ys, xs), axis = -1).reshape(-1,2)

            def mean_input(arr):
                arr[np.isnan(arr)] = arr[~np.isnan(arr)].mean() #<- fill in annoying nans...
            
            out = griddata(
                yx_flat,
                median_flat,
                yxs, 
                method='linear'
            )
            mean_input(out)
            
            out = out.reshape(x.shape)

        else:

            M = yx_flat.shape[0]
            random_mask = np.random.random(M) < (5000 / M)
            out = Rbf(
                yx_flat[random_mask][...,0], yx_flat[random_mask][...,1], median_flat[random_mask],
                function='thin-plate',
                # function='gaussian',
                smooth = 0.0, #<- adjust carefully ok this does bad things..
                mode = 'N-D',
            )

            self.coarseness = 10
            H,W,_ = x.shape
            Hc,Wc = H//self.coarseness, W//self.coarseness
            
            mxs, mys = np.meshgrid(np.linspace(0,W-1,Wc), np.linspace(0,H-1,Hc))
            gradient = out(mys, mxs)
            
            out = F.upsample(
                torch.from_numpy(gradient).reshape(1,Hc,Wc,3).permute(0,3,1,2),
                size = (H,W),
                mode='bilinear',
                align_corners = True,
            )[0].permute(1,2,0).numpy()
            #let's try Rbf thin plate method again..
            

        #i should be dividing actually...
        #this might be helpful for debugging
        # for c in range(3): plt.axes().contour(out[...,c], levels=10, colors=['red', 'green', 'blue'][c])
        # imshow_norm(out)
        # st()

        # x_ = utils.blur(x_, 10.0)
        # SUBTRACTION MODEL
        #0.02 bkg
        #don't clip too much
        rval = np.clip(x_ - out, 0.0, 1.0) 
        return rval / rval.max(axis=(0,1)) # renormalize per channel

    def global_tonemap(self, x):

        # x_ = x.copy()

        #5 and 99.5
        lb = 10
        ub = 99.5
        # gamma adjustment
        x = utils.renormalizep(x, lb, ub) #<- very interesting...5 works, 10??
        xg = x**(1/self.gamma)
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

        border = 400
        miny, maxy, minx, maxx = self.excl_box
        xc_crop = xc[miny-border:maxy+border, minx-border:maxx+border]
        # imshow(xc)
        tifffile.imsave('triangulum/triangulum.tiff', xc_crop)
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
        
        x = self.subtract_gradient(x)
        x = self.global_tonemap(x)
        
        x = self.star_shrink(x)
