from munch import Munch as M

cores = 20

demosaic_params = M(
    # at most one of use_flat or use_lens_profile should be True
    # strongly recommended to have at least 1 be True
    use_flat = False,
    use_lens_profile = True,
    alg = 'DCB', #alternatively, use LMMSE

    camera = 'auto', # alternatively, specify something like "Canon EOS 6D Mark II"
    lens_make = 'auto', # alternatively, specify somnething like 'Nikon'
    lens = 'Canon EF 70-200mm f/2.8L IS II USM', #'Nikkor 80-200mm f/2.8 ED',
)

detection_params = M(
    Nsig = 3, # number of kernel sizes to try 
    min_sig = 1.0, # smallest kernel in px/std
    max_sig = 6.0, # largest kernel in px/std

    # only consider star candidates above this percentile of pixel luminosity
    # 99.5 good for HA images, 99 for dark skies, 90 for typical use
    lum_pthresh = 90,

    # only consider candidates with an aspect ratio of no more than
    unround_threshold = 1.5,
)

registration_params = M(
    max_stars = 500, # use this many stars to register at most. 
    nneighbors = 500, 

    ba_max_ratio = 0.99,
    cb_max_ratio = 0.99,
    epsilon = 1E-3, # match tolerance. 

    min_abs_diff = 1, #abs and rel diff for match success
    min_rel_diff = 1.4,

    # we discard outliers from the registration via a ransac process
    ransac_iters = 50,
    ransac_keep_percentile = 99,
    # a point is an outlier if it's more than this many pixels from the linear fit
    linear_fit_tol = 2.0, 
)

warping_params = M(
    coarseness = 10, 
    use_thinplate = False, # recommend only for multi-spectra
    thinplate_smoothing=0,
    min_stars = 20, # don't attempt warp with fewer stars
)

stacking_params = M(
    # higher noise mul = more denoising, less robust to registration errors
    # lower noise mul = more robust, less denoising
    noise_mul = 32.0, # could also try 4, 16, 64, usually looks the same
    patch_size = 32,
    cache_path = '.cache', # a large np array is temporarily stored here
)

postprocess_params = M(
    # crop this many pixels from the edge of the image before any processing
    border_crop = 400,
    
    # parameters for removing background "gradient".
    gradient_window = 32+1, # size of the median kernel (odd)
    dilation = 16, # dilation factor for median kernel
    gradient_max = 90, # all pixels more luminous than this threshold are not counted as bkg

    # excl_box is either None, or a list of 4 integers [miny, maxy, minx, maxx]
    # this region will be ignored for the purposes of estimating background
    excl_box = None, 
    # alternatively, you can pass in a path to a mask file, to ignore non-box regions
    mask_file = None, #

    # a pair of (input, output) pairs for the tone curve.
    tone_curve = [
        (0.05, -0.02),
        (0.3, 0.0),
    ],
    curve_type = 'thin-plate', # can also choose "cubic" for less overshoot

    # if output border is given, the saved output will be the excl box, plus output border
    # otherwise, you can manually specify the [miny, maxy, minx, maxx] for the output
    output_border = 400,
    output_box = None,
)
