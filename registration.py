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

class Register:
    def __init__(
            self,
            work_dir,
            params = M(

                max_stars = 1000, #take this many stars at most
                
                nneighbors = 500, #must be even (1k before)
                ba_max_ratio = 0.99, 
                cb_max_ratio = 0.99,
                epsilon = 1E-3, #match tol
                
                min_abs_diff = 1, #abs and rel diff for match success
                min_rel_diff = 1.4,

                ransac_iters = 10,
                ransac_keep_percentile = 99,
                linear_fit_tol = 3.0, #pixels tol on linear fit
            )):

        self.work_dir = work_dir
        for k in params:
            setattr(self, k, params[k])

    def gen_triangles(self, pts):
        NN = min(self.nneighbors, 2*((pts.shape[0]-1)//2))
        #1st nn is the point itself, so we need to trim it out...        
        indices = utils.get_nearest_neighbors(pts, NN+1)[1][:,1:] #N x nbs
        
        indices = indices.reshape(pts.shape[0], NN//2, 2)
        indices = cat(
            (indices, 
             np.tile(np.arange(pts.shape[0])[:,None,None], (1,NN//2,1))),
            axis = 2
        )
        indices = indices.reshape(-1,3) #triangle indices..

        distances = np.stack((
            np.linalg.norm(pts[indices[:,0]] - pts[indices[:,1]], axis = 1),
            np.linalg.norm(pts[indices[:,0]] - pts[indices[:,2]], axis = 1),
            np.linalg.norm(pts[indices[:,1]] - pts[indices[:,2]], axis = 1),
        ), axis = 1) #Tx3 distancse

        #a triangle has 5 components...
        #1. ratio of sides b/a, c/b and index of the elements i,j,k
        # long and medium being i, long and short being j, medium and short being k

        dorder = distances.argsort(axis=1)
        dsorted = np.sort(distances, axis = -1)

        # there's a kind of clever trick here...
        # if dorder[:,0] == 0, then shortest edge is 01, which means lm must be 2
        # if dorder[:,2] == 2, then longest edge is 12, which means lm must be 0
        lm_i = 2*(dorder[:,0] == 0) + 1*(dorder[:,0] == 1) + 0*(dorder[:,0] == 2)
        ms_i = 2*(dorder[:,2] == 0) + 1*(dorder[:,2] == 1) + 0*(dorder[:,2] == 2)
        ls_i = 3 - lm_i - ms_i

        ba_r = dsorted[:,1] / dsorted[:,2]
        cb_r = dsorted[:,0] / dsorted[:,1]

        tri_ratio = np.stack((ba_r, cb_r), axis = 1)
        tri_index_local = np.stack((lm_i, ms_i, ls_i), axis = 1)
        tri_index = indices[
            np.arange(indices.shape[0]).repeat(3),
            tri_index_local.reshape(-1)
        ].reshape(-1,3)
        
        #filter ba as described in paper to reduce false matches
        valid = (ba_r < self.ba_max_ratio) & (cb_r < self.cb_max_ratio)
        tri_ratio = tri_ratio[valid]
        tri_index = tri_index[valid]
        
        #let's plot them to make sure we're doing sane things...
        # scatter(pts)
        # tripaths = mmap(lambda tri: mpl.path.Path(pts[tri][:,::-1], closed=False), tri_index)
        # patches = mpl.collections.PathCollection(tripaths, linewidths=0.1, facecolors='none')
        # plt.axes().add_artist(patches); plt.show()

        print(f'{tri_ratio.shape[0]} triangles generated')
        return M(ratio = tri_ratio, index = tri_index)

    def match_triangles(self, source, target, source_tris, target_tris):
        ''' using the voting algorithm '''
        N, M = source.shape[0], target.shape[0]

        # scatterk(source_tris.ratio, c='red')
        # scatterk(target_tris.ratio, c='blue')
        # st()

        matches = utils.nearby_pairs(source_tris.ratio, target_tris.ratio, self.epsilon)
        print(f'{matches.shape[0]} triangle correspondences found')

        source_pts = source_tris.index[matches[:,0]].reshape(-1)
        target_pts = target_tris.index[matches[:,1]].reshape(-1)
        coords = np.stack([source_pts, target_pts], axis = 1)
        ucoords, counts = np.unique(coords, axis = 0, return_counts = True)

        votes = np.zeros((N,M), dtype = int)
        votes[ucoords[...,0], ucoords[...,1]] = counts
        
        #"best" matches
        cy, cx = ((votes >= votes.max(axis=0)[None]) & (votes >= votes.max(axis=1)[...,None])).nonzero()
        cvotes = votes[cy,cx]

        #now we need the runner up votes
        runner_up = np.maximum(
            np.sort(votes, axis = 1)[:,2][:,None],
            np.sort(votes, axis = 0)[-2][None],
        )[cy,cx]

        good_match = (cvotes-runner_up >= self.min_abs_diff) & (cvotes/(runner_up+1E-3) > self.min_rel_diff)

        matches = np.stack((cy,cx),axis=1)[good_match]
        print(f'matched {len(matches)} stars')
        return matches

    def ransac_linear(self, source, target):
        '''
        fit a affine transform from source to target
        discard outlier points
        '''

        valid = np.ones(source.shape[0], dtype = np.bool)

        for i in range(self.ransac_iters):
            T, residuals = utils.fit_affine(source[valid], target[valid])
            residuals = np.linalg.norm(residuals, axis = -1)

            if i == self.ransac_iters -1:
                valid_criteria = residuals < self.linear_fit_tol
            else:
                valid_criteria = residuals < np.percentile(residuals, self.ransac_keep_percentile)
            valid[valid] &= valid_criteria

        source = source[valid]
        target = target[valid]
        print(f'{source.shape[0]} inlier stars, mean error {residuals.mean():.3f} px')
        print(T)
        return source, target
    
    def register(self, source, target):
        source, target = source[...,:2], target[...,:2] #don't make use of std
        
        source_tris = self.gen_triangles(source)
        target_tris = self.gen_triangles(target)
        matches = self.match_triangles(source, target, source_tris, target_tris)

        source_matched = source[matches[...,0]]
        target_matched = target[matches[...,1]]

        scatterk(source_matched, c='red')
        scatterk(target_matched, c='blue')
        corresp = mmap(lambda st: mpl.path.Path(np.stack(st,axis=0)[:,::-1]), zip(source_matched, target_matched))
        patches = mpl.collections.PathCollection(corresp, linewidths=1, facecolors='none', alpha=0.5)
        # plt.axes().add_artist(patches); plt.show()
        # st()

        #this step fits a linear transform and discards outliers
        source_lin, target_lin = self.ransac_linear(source_matched, target_matched)

        return cat((source_lin, target_lin), axis=1) #Nx4

    def __call__(self, paths):
        stars = mmap(np.load, paths)

        #sort by #stars, descending
        paths, stars = zip(*sorted(
            zip(paths, stars),
            key = lambda x: -x[1].shape[0]
        ))

        stars = [star[-self.max_stars:] for star in stars]

        for i, star in enumerate(stars[1:]):
            matches = self.register(stars[0], star)
            name0 = os.path.basename(paths[0]).replace('.npy', '')
            namei = os.path.basename(paths[i+1]).replace('.npy', '')
            out_path = f'{self.work_dir}/registration/{name0}-{namei}'
            np.save(out_path, matches)
            

if __name__ == '__main__':
    pass
