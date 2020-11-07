import numpy as np
import scipy as sp
from ipdb import set_trace as st
import skimage as ski
import utils
from matplotlib import pyplot as plt
from common import *
from munch import Munch as M
from tqdm import tqdm
from tqdm.contrib import tzip, tenumerate
import pathos
import dill

class Detector:
    def __init__(
            self,
            work_dir,
            params = M(
                min_sig = 1.0, 
                max_sig = 6.0,
                Nsig = 3,
                #lum_pthresh = 98, #bright thresh for star
                #unround_threshold = 1.5, #discard less round than this..

                #less conservative
                lum_pthresh = 96,
                unround_threshold = 2.0,
                #95, 1.5 also works
            )):

        self.work_dir = work_dir
        for k in params:
            setattr(self, k, params[k])

    def get_all_candidates(self, lum):
        ys, xs = [], []
        for sigma in np.geomspace(self.min_sig, self.max_sig, self.Nsig):
            blurred_lum = utils.blur(lum, sigma)
            local_max = utils.local_maximum(
                blurred_lum[...,0],
                radius = int(2*sigma),
                factor = 1.1
            ) #yx coords
            
            ys.append(local_max[0])
            xs.append(local_max[1])

        ys = cat(ys, axis = 0)
        xs = cat(xs, axis = 0)
        return np.stack((ys, xs), axis = 1)#Nx2

    def fit_gaussians(self, lum, candidates):
        #first, extract a ~4sig x 4sig patch around each candidate
        radius = int(self.max_sig*1.5)
        patches = utils.grab_patches(lum, candidates, radius) #NPP1

        fitted = []

        from pathos.pools import ProcessPool 
        pool = ProcessPool(nodes=8)
        fitted = pool.map( utils.fit_gauss_elliptical, patches[...,0] )
        
        # for i, (candidate, patch) in tenumerate(tzip(candidates, patches)):
        #     fitted.append( utils.fit_gauss_elliptical(patch[...,0]) )

        fitted = np.stack(fitted, axis = 0) #Nx7 now

        centers = candidates + fitted[:,2:4] - radius

        sx, sy = np.abs(fitted[:,-2]), np.abs(fitted[:,-3])
        std = ((sx+sy)/2.0)

        # luma = std * scale
        brightness = std * fitted[:,1]
        
        balls = cat((centers, brightness[:,None], std[:,None]), axis = 1)
        
        unroundness = np.maximum(sx/sy, sy/sx)
        
        valid = (
            (unroundness < self.unround_threshold) &
            (std < self.max_sig*1.0) #we shouldn't trust anything too big
        )
        
        return balls[valid]
    
    def __call__(self, path):
        '''
        img is a 1 channel numpy raster (luminance)
        '''

        out_path = path.split('/')[-1].split('.')[0]
        out_path = f'{self.work_dir}/detections/{out_path}'
        if os.path.exists(out_path+'.npy'):
            print(f'skipping {out_path} cause already done')
            return
        
        print(f'loading and preprocessing {path}')
        img = utils.load_raw(path)
        lum = utils.luminance(img)

        print(f'generating preliminary list of candidates')
        candidates = self.get_all_candidates(lum)
        cand_lum = lum[candidates[...,0], candidates[...,1]][...,0]

        #filter out all candidates too dark to be a star
        thresh = np.percentile(lum, self.lum_pthresh)
        candidates = candidates[cand_lum > thresh]

        #preliminary fuse candidates by 2x minimum sigma
        #if merge, we take the latter point (i.e larger sigma)
        candidates = utils.fuse_nearby_points(candidates, 2*self.min_sig)
        # scatter(candidates); imshow(lum) # <- a nice debug plot
        print(f'{candidates.shape[0]} candidates found, fitting gaussians')
        
        #fit a 2D gaussian to each candidate
        balls = self.fit_gaussians(lum, candidates)
        print(f'{balls.shape[0]} gaussians fitted')
        balls = utils.fuse_balls(balls)
        print(f'{balls.shape[0]} left after merge')

        #sort by brightness
        balls = balls[balls[:,2].argsort()]

        print(f'saving to {out_path}')
        np.save(out_path, balls)

        # scatterr(balls); imshow(np.clip(lum*10, 0,1)) #<- plotting
        return balls

    def test(self):
        balls = self('test_data/m42r/IMG_0750.CR2')

if __name__ == '__main__':
    Detector().test()
