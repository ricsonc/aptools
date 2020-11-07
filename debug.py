from common import *
import utils
from ipdb import set_trace as st
from glob import glob

flats = glob('flat/*')

def standardize(lum):
    lum += lum[:,::-1] #flip vert and hor
    lum += lum[::-1,] 
    lum /= lum.mean() #mean brightness = 1
    return lum

total = 0
for flat in flats:
    flat = utils.load_raw(flat, profile=False)
    lum = utils.luminance(flat)
    slum = standardize(lum)
    total += slum

total /= len(flats)

#can get a decent flat here by blurring with a good size
master_flat = utils.blur(total, 20.0)
master_flat /= master_flat.max()
np.save('flat.npy', master_flat)

exit()

H,W,_ = total.shape
mxs, mys = np.meshgrid(np.arange(W), np.arange(H))

#the center of the image is technically (H-1)/2
dx = mxs-(W-1)/2
dy = mys-(H-1)/2
# let's use 1000 as the fudge factor on pixels
r2 = (dx**2 + dy**2).reshape(-1,1) / 1000000.0

basis_fn = lambda z: cat((np.ones((z.shape[0],1)), z, z**2, z**3), axis=-1) #need to fix...

lum = total.reshape(-1,1)

T, res = utils.fit_linear(basis_fn(r2), lum)
st()

#yeah it's weird

#need three channels
