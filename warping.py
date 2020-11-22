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
import imageio as ii
from glob import glob
from torch.nn import functional as F
import torch

class Warp:
    def __init__(self,
        work_dir,
        params = M(
            coarseness = 10
        )):

        self.work_dir = work_dir
        for k in params:
            setattr(self, k, params[k])

    def warp(self, reg, raw):
        T, residuals = utils.fit_affine(reg[...,:2], reg[...,2:])

        USE_TP = False

        if USE_TP:
            out = Rbf( #<- thin plates are crazy... be very careful
                reg[...,0], reg[...,1], residuals,
                function='thin-plate',
                smooth = 2000000000.0, # <- yes this looks large... but it's correct.
                mode = 'N-D',
            )

            resres = np.linalg.norm(out(reg[...,0], reg[...,1]) - residuals, axis=-1).mean()
        else:
            resres = np.linalg.norm(residuals, axis=-1).mean()
            
        print('average residual is', resres)

        H,W,_ = raw.shape
        Hc,Wc = H//self.coarseness, W//self.coarseness
        mxs, mys = np.meshgrid(np.linspace(0,W-1,Wc), np.linspace(0,H-1,Hc))


        if USE_TP:
            flow = out(mys, mxs)
            flow_mag = np.linalg.norm(flow**2, axis=-1)
            if flow_mag.mean() > 0.5 and resres > np.linalg.norm(residuals,axis=-1).mean():
                print('your smoothing parameter is set too low and is causing overshoot issues!!')
                imshow(np.clip(flow_mag, 0, 2)/2)
                st()
        else:
            flow = 0
    
        flow += affine(T, np.stack((mys, mxs), axis = -1).reshape(-1,2)).reshape(Hc,Wc,-1)
        flow_up = F.upsample(
            torch.from_numpy(flow).reshape(1,Hc,Wc,2).permute(0,3,1,2),
            size = (H,W),
            mode='bilinear',
            align_corners = True,
        )
        #flow is in dy,dx mode

        flow_up[:,0] *= 2/(H-1) #01 now...
        flow_up[:,1] *= 2/(W-1)
        flow_up -= 1
        #need to flip for some ridiculous convention
        flow_up = torch.flip(flow_up, dims=(1,)).type(torch.float32)

        warped = F.grid_sample(
            torch.from_numpy(raw).reshape(1,H,W,3).permute(0,3,1,2),
            flow_up.permute(0,2,3,1),
            mode='bilinear',
            padding_mode='zeros',
            align_corners = True
        )
        warped = warped[0].permute(1,2,0).numpy().astype(np.float32)
        return warped

    def save(self, out, path):
        print(f'warped {path}')
        path = path.replace('raw', 'warped').split('.')[0]
        np.save(path, out)
        ii.imsave(path+'.jpg', np.sqrt(out))

    def __call__(self, raw_path):
        
        raw = utils.load_raw(raw_path)
        name = os.path.basename(raw_path).split('.')[0]
        reg_path = glob(f'{self.work_dir}/registration/*{name}*')
        
        if len(reg_path) == 1:
            reg = np.load(reg_path[0])
            if reg.shape[0] < 100: #min needed for "good" registration:
                print(f'giving up for {reg_path}')
                return 
            out = self.warp(reg, raw)
            self.save(out, raw_path)
        else:
            #no warping needed for ref frame
            self.save(raw, raw_path)

