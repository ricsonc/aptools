import numpy as np
import scipy as sp
from munch import Munch as M
from ipdb import set_trace as st
from skimage import color
from scipy import ndimage
from scipy import spatial
from common import *
import exifread
import rawpy
# import lensfunpy
# from lensfunpy.util import remapScipy
import os
from scipy.optimize import leastsq, least_squares
from tqdm import tqdm

def prepare_dark(paths):
    darks = np.stack([x.raw_image for x in mmap(rawpy.imread, paths)], axis=0).mean(axis=0)

    mu = np.median(darks)
    std = darks.std()
    #center -1 to +1 std between 0 and 1
    deviation = 0.5 + 0.5*(darks-mu)/std

    imshow(np.clip(deviation,0,1)) #visually inspect for patterns to determine if darks helpful

    workdir = paths[0].split('/')[0]
    np.save(f'{workdir}/dark.npy', darks)

def load_raw(path, profile=True, fmul = 4.0):
    path_ = path.replace('raw', 'demosaic')+'.npy'
    if os.path.exists(path_):
        return np.load(path_)
    
    #output: 16 bit HWC
    with open(path, 'rb') as f:
        tags = exifread.process_file(f)
        
    raw = rawpy.imread(path)

    #try to load dark frame
    if load_raw.cache is None:
        workdir = path.split('/')[0]
        dark_path = f'{workdir}/dark.npy'
        if os.path.exists(dark_path):
            print('LOADING DARK')
            #being very careful here -- need to multiply by 4 because we're going from 14 to 16 bits
            load_raw.cache = (np.load(dark_path)*fmul).astype(np.uint16)


    dark = load_raw.cache
    if dark is not None:
        raw_data = raw.raw_image
        #avoid underflow with this trick
        raw_data = np.minimum(raw_data, raw_data-dark)

    demosaiced = raw.postprocess(rawpy.Params(
        # demosaic_algorithm = rawpy.DemosaicAlgorithm.DCB,
        demosaic_algorithm = rawpy.DemosaicAlgorithm.LMMSE,
        use_camera_wb = False,
        use_auto_wb = False,
        output_bps = 16,
        # output_color = rawpy.ColorSpace.raw, #keep linear
        gamma = (1,1),
        no_auto_bright = True,
        user_flip = 0,
    )) / 65535.0

    if not profile: #debug mostly
        return demosaiced

    if load_raw.cache2 is None:
        workdir = path.split('/')[0]
        flat_path = f'{workdir}/flat.npy'
        if os.path.exists(flat_path):
            print('LOADING FLAT')
            load_raw.cache2 = np.load(flat_path)
        else:
            assert False, 'probably need a flat file?'

    flat = load_raw.cache2
    if flat is not None:
        demosaiced /= flat
        demosaiced = np.clip(demosaiced, 0.0, 1.0)
        
    np.save(path_, demosaiced.astype(np.float32))
    return demosaiced
    
    #ignore below... i didn't have much success using lensfunpy
    
    # # correct for lens distortion .. NOT IMPLEMENTED    
    # H,W,_ = demosaiced.shape
    # db = lensfunpy.Database()
    # cam = db.find_cameras(tags['Image Make'].values, tags['Image Model'].values)[0]
    # lens = db.find_lenses(cam, 'Nikon', 'Nikkor 80-200mm f/2.8 ED')[0]

    # mod = lensfunpy.Modifier(lens, cam.crop_factor, W, H)
    # mod.initialize(
    #     float(tags['EXIF FocalLength'].values[0]),
    #     float(tags['EXIF FNumber'].values[0]),
    #     1000.0,
    #     pixel_format = np.float64
    # )

    # # # remapScipy(??, mod.apply_geometry_distortion()) #<- geom correction -- let's not bother
    # # st()
    # status = mod.apply_color_modification(demosaiced)
    # assert status
    # # st()

    # #i think clipping here is "acceptable"
    # demosaiced = np.clip(demosaiced, 0.0, 1.0)
    
    # np.save(path_, demosaiced)
    # return demosaiced

load_raw.cache = None
load_raw.cache2 = None

def luminance(img): #from linear components
    return color.rgb2gray(img)[...,None]
    # return 0.2126 * img[...,0] + 0.7152 * img[...,1] + 0.0722 * img[...,2]

def blur(img, radius):
    #convolve input with gaussian
    #input: HWC np raster
    _,_,C = img.shape
    return np.stack([
        ndimage.gaussian_filter(
            img[...,c],
            radius,
            order = 0,
            mode = 'nearest',
            truncate = 4.0,
        )
        for c in range(C)
    ], axis = 2)
    
def local_maximum(HW_img, radius, factor=1.0):
    #checks for strict maximum
    max_filter = ndimage.filters.maximum_filter(HW_img, size=radius)
    min_filter = ndimage.filters.minimum_filter(HW_img, size=radius)
    return (
        (HW_img == max_filter) &
        (max_filter > min_filter*factor)
    ).nonzero() #3x3 window

def grab_patches(image, centers, radius):
    ''' i am lazy, and surely this approach is fast enough '''
    C = image.shape[-1]
    
    N = centers.shape[0]
    P = radius*2+1

    #first pad image with radius wide border to avoid any issues
    padded = np.pad(image, ((radius, radius), (radius, radius), (0,0)), mode='edge')

    output = np.zeros((N,P,P,C))
    for i, (y, x) in enumerate(centers):
        output[i] = padded[y:y+P, x:x+P]

    return output

def fit_gauss_elliptical(patch):
    P = patch.shape[0]
    bkg = np.median(patch)

    initialization = np.array([
        bkg, 
        patch.max()-bkg, #height
        (P-1)/2, #center
        (P-1)/2,
        P/8, #std 
        P/8,
        0, #angle
    ])
    
    def gauss(floor, scale, mux, muy, sx, sy, theta):
        inv_sxx = 1.0/sx**2
        inv_syy = 1.0/sy**2
        ct = np.cos(theta)
        st = np.sin(theta)
        A = ct**2 * inv_sxx + st**2 * inv_syy
        B = st**2 * inv_sxx + ct**2 * inv_syy
        C = 2 * st * ct * (inv_sxx - inv_syy)
        # return lambda x,y: floor + scale * np.exp(-0.5*(A*((x-mux)**2)+B*((y-muy)**2)+C*(x-mux)*(y-muy)))
        #clips at 1.0
        return lambda x,y: np.minimum(floor + scale * np.exp(-0.5*(A*((x-mux)**2)+B*((y-muy)**2)+C*(x-mux)*(y-muy))), 1.0)

    def err(p):
        return np.ravel(gauss(*p)(*np.indices(patch.shape))-patch)

    #sx and sy is kind of misleading, not necessarily axis aligned...
    # return leastsq(err, initialization, maxfev=200)[0]
    return least_squares(err, initialization, method='lm', max_nfev=100).x
    

def nearby_pairs(points1, points2, radius):
    tree1 = spatial.ckdtree.cKDTree(points1)
    tree2 = spatial.ckdtree.cKDTree(points2)
    nbhs = tree1.query_ball_tree(tree2, r = radius)
    self_index = [[i]*len(nbh) for i, nbh in enumerate(nbhs)]
    nbhs = cat(nbhs).astype(np.int)
    self_index = cat(self_index).astype(np.int)
    pairs = np.stack((self_index, nbhs), axis = 1)
    return pairs

def fuse_nearby_points(points, radius):
    tree = spatial.ckdtree.cKDTree(points)
    pairs = tree.query_pairs(radius)
    uppers = np.array([x[0] for x in pairs])
    return np.delete(points, uppers, axis = 0)

def fuse_balls(balls):
    balls = balls[balls[:,-1].argsort()] #sort by radius, ascending
    
    #given a bunch of centers and radius, fuse any touching balls
    tree = spatial.ckdtree.cKDTree(balls[:,:2])

    #should be "good enough"
    max_radius = np.percentile(balls[:,-1], 99)*2 
    pairs = tree.query_pairs(max_radius)

    fst = np.array([x[0] for x in pairs])
    snd = np.array([x[1] for x in pairs])
    distance = np.linalg.norm(balls[fst,:2] - balls[snd,:2], axis = 1)
    clearance = distance - balls[fst,-1] - balls[snd,-1]
    do_merge = clearance <= 0.0

    #fst has smaller radius
    return np.delete(balls, fst[do_merge], axis = 0)

def get_nearest_neighbors(points, n):
    tree = spatial.ckdtree.cKDTree(points)
    return tree.query(points, n)

def fit_affine(X_, Y_):
    X, Y = hom(X_), hom(Y_)
    T, _ = fit_linear(X, Y)
    residuals = Y_ - dehom(X @ T)
    return T, residuals
    
def fit_linear(X, Y):
    T = np.linalg.inv(X.T @ X) @ X.T @ Y
    residuals = Y - X @ T
    return T, residuals

def fit_transform_linear(X, Y):
    T, residuals = fit_linear(X, Y)
    return T, X @ T

def renormalize(x):
    x = x-x.min()
    x /= x.max()
    return x

def renormalizep(x, p_low, p_hi, axis=(0,1)):
    x = x-np.percentile(x, p_low, axis = axis)
    x /= np.percentile(x, p_hi, axis=axis)
    return np.clip(x,0,1)

def gamma_curve(gamma = 2.4, k = 12.92):
    ''' default parameters define the sRGB gamma'''
    
    '''
    the two parts of the curve are
    f(x) = kx for x <= z
    and
    h(x) = (1+w)x^(1/g)-w

    where w, z are constants such that...
    f(z) = h(z)
    f'(z) = h'(z)

    so we have a system of two qeustions...
    kz = (1+w)z^(1/g)-w
    k = (1+w)/g z^(1/g-1)

    let's fix w first...
    kz = z^(1/g) + w(z^1/g-1)
    w = [ kz - z^(1/g) ] / (z^1/g-1)

    k = { 1 + [ kz - z^(1/g) ] / (z^1/g-1) } /g * z^(1/g-1)
    k = kz/g
    z = g
    ^ this can' tbe right...
    
    '''

    #nonlinearity ends when slope = d/dx x^(1/gamma) = (1/gamma) x^(1/gamma-1)
    # k gamma = x^(1/gamma-1)
    pass

def transfer_function(x):
    #the standard srgb transfer... i don't really get this
    return np.where(
        x <= 0.0031308,
        12.92 * x,
        1.055*x**(1/2.4) - 0.055
    )
        
def generate_flat(files):
    
    def standardize(flat):
        flat += flat[:,::-1] #flip vert and hor
        flat += flat[::-1,] 
        flat /= flat.mean(axis=(0,1)) #mean brightness = 1
        return flat

    total = 0
    for flat in tqdm(files):
        flat = load_raw(flat, profile=False, fmul=0.1)
        #expect around 1/40 the dark current due to exposure time        
        sflat = standardize(flat)
        total += sflat

    total /= len(files)

    #can get a decent flat here by blurring with a good size
    master_flat = blur(total, 20.0)
    master_flat /= master_flat.max(axis=(0,1))

    work_dir = files[0].split('/')[0]
    np.save(f'{work_dir}/flat.npy', master_flat)

    # below is only for visualization purposes...
    print('vignetting looks like this: ')
    
    #trim the edges
    total = blur(total, 10.0)
    total = total[20:-20,20:-20] # HW3

    H,W,_ = total.shape
    mxs, mys = np.meshgrid(np.arange(W), np.arange(H))

    #the center of the image is technically (H-1)/2
    dx = mxs-(W-1)/2
    dy = mys-(H-1)/2
    # let's use 1000 as the fudge factor on pixels
    r2 = (dx**2 + dy**2).reshape(-1,1) / 1000000.0

    #a polynomial basis is popular...
    # basis_fn = lambda z: cat((np.ones((z.shape[0],1)), z, z**2, z**3), axis=-1) 
    # T, pred = fit_transform_linear(basis_fn(r2), total.reshape(-1,3))

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
