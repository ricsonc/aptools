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
from scipy.interpolate import Rbf
import os
from scipy.fft import fftn, ifftn, set_workers
from tqdm.contrib.itertools import product
from tqdm import tqdm
from tqdm.contrib import tenumerate
import tifffile
import imageio as ii
from scipy import misc
import random

class Stacker:
    def __init__(
        self,
        work_dir,
        params = M(
            # higher noise mul = more denoising, less robust
            # lower noise mul = more robustness, less denoising
            noise_mul = 32.0, # fudge factor (also try 4, 16, 64)
            #patch_size = 32, # <- according to hasinoff...
            patch_size = 32,

        ),
        custom_name = 'out',
        method = 'fuse',
    ):

        self.custom_name = custom_name
        self.method = method

        self.work_dir = work_dir
        for k in params:
            setattr(self, k, params[k])

    def fuse_clip(self, patches, iters=2, k=2):
        valid = np.ones_like(patches)

        for _ in range(iters):
            #compute the masked mean
            num = (valid*patches).sum(axis=0)
            denom = valid.sum(axis=0)+1E-9
            mu = num/denom

            diff = patches-mu[None]
            stds = (diff**2).sum(axis=0)**0.5/np.sqrt(patches.shape[0])
            valid = np.abs(diff) < stds[None]*k

            #print('valid mean is', valid.mean())

        return mu

    def fuse(self, patches):
        avg = patches.mean(axis = (0))
        dist = np.abs(patches-avg[None]).max(axis = (1,2))

        ffts = fftn(patches, axes = (1,2), norm='ortho')
        reffft = ffts[dist.argmin()] #reference patch

        num_frames = patches.shape[0]
        noise = (patches.std(axis=0)**2).mean()
        # fudge factors from hasinoff paper
        noise *= num_frames**2 / 8 * self.noise_mul
        
        fft_err = np.abs(reffft[None] - ffts)**2
        err_scores = fft_err / (fft_err + noise)
        # print('score', err_scores.min(), err_scores.mean(), err_scores.max())
        # if err_scores.max() > 0.5 and dist.max() < 2E-3:
        #     st()
        
        correction = err_scores * (reffft[None] - ffts)
        fused_fft = (ffts + correction).mean(axis = 0)

        fused_patch = ifftn(fused_fft, axes = (0,1), norm='ortho').real #imag components 0
        # if dist.max() < 2E-3:
        #     st()
        
        return fused_patch
            
    def __call__(self, paths):
        N = len(paths)
        H,W,C = np.load(paths[0]).shape

        PS = self.patch_size
        HPS = PS//2 #img size must be a multiple of this

        padH = np.ceil(H/HPS).astype(np.int)*HPS-H
        padW = np.ceil(W/HPS).astype(np.int)*HPS-W
        paddedH, paddedW = H+padH+PS, W+padW+PS
        
        imgs = np.memmap('.cache', dtype=np.float32, mode='w+', shape=(N,C,paddedH,paddedW))
        for i, path in tenumerate(paths):
            imgs[i] = np.pad(
                np.load(path, mmap_mode='r').transpose((2,0,1)),
                ((0,0), (HPS, padH+HPS), (HPS, padW+HPS)),
                mode='edge'
            )
        
        out = np.zeros((C, paddedH, paddedW))

        xs, ys = np.meshgrid(np.arange(PS), np.arange(PS))
        foo = lambda z: 0.5 - 0.5*np.cos( 2 * np.pi * (z.astype(np.float32)+0.5)/PS)
        window_fn = foo(xs) * foo(ys)


        for (i, j) in product(range(paddedH//HPS-1), range(paddedW//HPS-1)):
            patch_slice = np.index_exp[
                i*HPS:(i+2)*HPS,
                j*HPS:(j+2)*HPS,
            ]

            patchws = [imgs[np.index_exp[:,c]+patch_slice] * window_fn[None] for c in range(C)]

            if True:
                from pathos.pools import ProcessPool 
                pool = ProcessPool(nodes=3)
                results = pool.map(self.fuse, patchws)
            else:
                results = [getattr(self, self.method)(patchw) for patchw in patchws]

            for c in range(C):
                out[c][patch_slice] += results[c]
            
        out = out[:,HPS:H+HPS,HPS:W+HPS]
        out = np.clip(out, 0.0, 1.0).transpose(1,2,0)
        np.save(f'{self.work_dir}/stacked/out', out)
        tifffile.imwrite(f'{self.work_dir}/stacked/out.tiff', out)
        ii.imsave(f'{self.work_dir}/stacked/{self.custom_name}.jpg', np.round(out*255).astype(np.uint8), quality=100)

if __name__ == '__main__':
    #let's test it out...
    
    face_ = misc.face() #an RGB image
    N = 20
    sigma = 100
    sigma2 = 500
    
    Nblur = 4
    Nalign = 4

    #1. add misalignments
    #2. add blurring

    np.random.seed(0)
    
    paths = []
    print('generating fake data')
    for i in tqdm(range(N)):
        face = (
            face_.astype(np.float32) +
            np.random.randn(*face_.shape) * sigma +
            utils.blur(np.random.randn(*face_.shape), 7) * sigma2
        )

        if i < Nblur:
            face = utils.blur(face, [1,2,3,4][i])
        elif Nblur <= i < Nalign+Nblur:
            face = np.roll(face, [5,7,9,3][i-Nblur], axis = 0)
            face = np.roll(face, [2,-4,5,-6][i-Nblur], axis = 1)
        else:
            face = utils.blur(face, 0.5)
            L = 2
            face = np.roll(face, random.randint(-L,L), axis=0)
            face = np.roll(face, random.randint(-L,L), axis=1)

        #let's not clip actually...
        #face = np.clip(face, 0.0, 255.0)/255.0
        face = face/255.0
        
        path = f'debug/testimgs/{i:02d}.npy'
        np.save(path, face)
        ii.imsave(path.replace('npy', 'jpg'), np.round(np.clip(face,0,1)*255.0).astype(np.uint8), quality=98)
        paths.append(path)
    
    bar = Stacker('debug', method='fuse', custom_name='hasinoff32')
    bar.patch_size=64
    bar(paths)

    bar.custom_name='hasinoff8'
    bar.noise_mul=1.0
    bar(paths)
    
    bar.custom_name='hasinoff128'
    bar.noise_mul=10000.0
    bar(paths)
    
    
    foo = Stacker('debug', method='fuse_clip', custom_name='clipavg')
    foo.patch_size=256
    foo(paths)
    
