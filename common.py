import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import os

def scatter(yx, ax=None):
    p = ax if ax is not None else plt
    p.scatter(yx[...,1], yx[...,0], s=1, c='red')

def scatterk(yx, ax=None, **kwargs):
    p = ax if ax is not None else plt    
    p.scatter(yx[...,1], yx[...,0], s=4, **kwargs)

def scatterr(yxr, ax=None):
    p = ax if ax is not None else plt    
    p.scatter(yxr[...,1], yxr[...,0], s=20*yxr[...,2], linewidth=1, facecolors='none', edgecolors='r')

def imshow(x, ax = None, **kwargs):
    if ax:
        p = ax
    else:
        p = plt
    p.imshow(x, **kwargs)
    plt.show()

def imshow_norm(x, **kwargs):
    x = x-x.min()
    x = x / x.max()
    imshow(x, **kwargs)
    
def imshow_tm(x):
    #tonemap
    x = x-x.min()
    x = x/x.max()
    x = np.sqrt(x)
    imshow(x)
    
def hist(x):
    plt.hist(x.reshape(-1,), bins = 200)
    plt.show()

cat = np.concatenate

def files_in(work_dir, sub_dir, ext = None):
    rval = glob(f'{work_dir}/{sub_dir}/*')
    if ext is None:
        return rval
    return [x for x in rval if x.endswith('.'+ext)]

def smap(f, xs): #strict side effects map
    for x in xs:
        f(x)

def mmap(f, xs): #strict side effects map
    return list(map(f, xs))
        
def hom(xs):
    N,K = xs.shape
    ones = np.ones((N,1),dtype = xs.dtype)
    return cat((xs, ones), axis = 1)

def dehom(xs):
    last_col = xs[:,-1][:,None]
    return xs[:,:-1]/last_col

def affine(T, xs):
    return dehom(hom(xs) @ T)

def ensure(path):
    if os.path.exists(path):
        return
    os.mkdir(path)

# def npload(x):
#     #maybe this'll help with memory?
#     return np.load(x, mmap_mode='r').astype(np.float32) 

def discard(f):
    def g(*args, **kwargs):
        f(*args, **kwargs)
    return g
