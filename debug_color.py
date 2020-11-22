from common import *
import utils
from ipdb import set_trace as st
from glob import glob
from tqdm import tqdm

#work_dir='haorion'
work_dir='oneb'

flats = glob(f'{work_dir}/flats/*')

def standardize(flat):
    flat += flat[:,::-1] #flip vert and hor
    flat += flat[::-1,] 
    flat /= flat.mean(axis=(0,1)) #mean brightness = 1
    return flat

total = 0
for flat in tqdm(flats):
    flat = utils.load_raw(flat, profile=False, fmul=0.1) #expect around 1/40 the dark current due to exposure time
    sflat = standardize(flat)
    total += sflat

total /= len(flats)

#can get a decent flat here by blurring with a good size
master_flat = utils.blur(total, 20.0)
master_flat /= master_flat.max(axis=(0,1))
np.save(f'{work_dir}/flat.npy', master_flat)

#what happens if we teim the edgese...
total = utils.blur(total, 10.0)
total = total[20:-20,20:-20]

#HW3 now..

H,W,_ = total.shape
mxs, mys = np.meshgrid(np.arange(W), np.arange(H))

#the center of the image is technically (H-1)/2
dx = mxs-(W-1)/2
dy = mys-(H-1)/2
# let's use 1000 as the fudge factor on pixels
r2 = (dx**2 + dy**2).reshape(-1,1) / 1000000.0

basis_fn = lambda z: cat((np.ones((z.shape[0],1)), z, z**2, z**3), axis=-1) #need to fix...

T, pred = utils.fit_transform_linear(basis_fn(r2), total.reshape(-1,3))

# st()

gt = total.reshape(-1,3)
mask = np.random.random(r2.shape[0]) < 0.001 #keep 1%
r = r2**0.5
r = r[mask]
gt = gt[mask]
plt.scatter(r, gt[:,0], c='red', s=1, alpha=0.1)
plt.scatter(r, gt[:,1], c='green', s=1, alpha=0.1)
plt.scatter(r, gt[:,2], c='blue', s=1, alpha=0.1)
plt.show()
# plt.scatter(tot

st()

#yeah it's weird

