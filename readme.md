There's already a lot of software out there for assisting with astrophotography. But I don't really trust any code I didn't write myself, and since astrophotography is heavily reliant on data processing (compared to "conventional" photography), I thought I should write my own set of tools. The goal in mind is to have simple and clean implementations of all the necessary algorithms to go from the raw data to a nearly finished image.

Of course, even though I tried out this code on a few seperate data sources, it's probably not nearly as robust as other available codes, and doesn't run as fast either. Some of the steps require a bit of parameter tuning. I do think i've improved on the "conventional implementations" of some processing steps though.

---

Step 0: raw to linear

In step 0, dark current is estimated via "dark frames", and vignetting is estimated via "flat frames". 

<img src="https://github.com/ricsonc/aptools/blob/master/readme_imgs/dark_current.png" width="500">

With modern sensors, dark current is actually very small, and I found that on my camera sensor (shown above), with 15 second exposures, non-uniform patterns in the dark current were limited to roughly 7 units in a 14-bit range -- with nonuniformity was only present in the bottom 10% of the sensor. So it can probably be omitted without harm.

For vignetting, I tried both a parametric vignetting model with data coming from the [lensfun project](https://lensfun.github.io/), and by directly estimating via "flat frames". The parametric lensfun model was nowhere near accurate enough. The flat frame approach was much closer, but still had brightness deviations on the order of 1-2% (much better than ~50% uncorrected).

<img src="https://github.com/ricsonc/aptools/blob/master/readme_imgs/vignette_model.png" width="500">
This picture shows that vignetting is actually wavelength dependent, and it was necessary to take this into account, lest I end up with discolored "corrected" images.

For demosaicing, I used the [rawpy](https://github.com/letmaik/rawpy) library, which interfaces with libraw. I used the [LMMSE demosaicing method](https://www4.comp.polyu.edu.hk/~cslzhang/paper/LMMSEdemosaicing.pdf), as I've read it's the best in high noise situations.

Step 1: star detection

I actually found very few references for exactly how star detection is done, but I've put together a pretty "common sense" approach. In the preliminary pass, I convolve a bunch of gaussians of different scales with the input, and take all the local maximums (if they're bright enough) as candidate stars. Then I extract an image patch around each candidate, and fit a specific gaussian (mean, cov, height) to each patch with levenberg marquardt. Only round and bright candidates are kept. A final pass filters out any duplicates.

<img src="https://github.com/ricsonc/aptools/blob/master/readme_imgs/star_detection.png" width="500">

A "novelty" I added here is in the gaussing fitting step. From my reading, other software assumes non-saturated pixels when fitting gaussians, which is a pretty bad assumption, as stars are often saturated. In my gaussian fitting code, I clip the gaussian at 1, which should let me obtain pretty good estimates even in saturated case. Of course this is such a simple tweak it's hard to imagine no one else has done it.

Step 2: star correspondence

The next step is to match stars across images. I implemented the [classic triangle similarity algorithm](https://hal.inria.fr/inria-00548539/document), which worked pretty well. As opposed to the original prescription, as well as other variations of the algorithm, my triangle construction implementation uses only smaller "local" triangles. 

<img src="https://github.com/ricsonc/aptools/blob/master/readme_imgs/registration.png" width="500">

The algorithm works surprisingly well, but often needs to have some parameters tweaked depending on the input. I made up some thresholds on what amount of triangle votes would qualify as a match. After the initial matching stage, I included a RANSAC linear fitting of the positions of the matched stars, with all outliers being discarded. This very effectively discards all the mismatches. 

As an aside, I found this correspondence problem to be really interesting, since, coming from the computer vision community, most of the literature I am familiar with often relies on dense images with varying textures and gradients, and this algorithm is different from anything I've seen before.

Step 3: image warping

In this step, I align all the images so the pixels are hopefully in perfect correspondence. I leverage the star correspondences from the last step, but instead of relying on a linear fit, I use a [thin-plate spline model](https://en.wikipedia.org/wiki/Thin_plate_spline) for the mapping. The reason for this is to account for possible lens distortions or other deformations. Practically speaking, I found that my average residual with a linear fit is typically 0.2 pixels, so it likely doesn't matter either way.

<img src="https://github.com/ricsonc/aptools/blob/master/readme_imgs/deform.png" width="500">

Plot of the deformation magnitude of one frame -- bright yellow = 1 pixel, so it's very little.

Step 4: image stacking

The most common methods for combining many exposures into a single image to reduce noise include average, median, and robust average (aka throw away outliers too far from the current average, and repeat). While this is extremely simple and works fairly well, I've been following some interesting computational photography literature, and methods for camera burst photography, where the same problem (of image stacking) exists. In particular, [Hasinoff et al.](https://people.csail.mit.edu/hasinoff/pubs/HasinoffEtAl16-hdrplus.pdf) suggests computing an FFT of the image (patches) and doing a robust average in that space. This is apparently called Wiener filtering. I implemented this approach, which was a bit challenging since I know absolutely nothing about signals processing. A very tricky "gotcha" concerns the noise term, which must be carefully scaled according to several details in the implementation, including the normalization convention of the FFT.

I suspect that since estimating flow on natural photographs with handshake and fast motion is significantly more challenging than aligning astrophotography images, these burst photography algorithms are probably designed more for coping with alignment failure rather than pure denoising. So i'm not entirely sure how much better (or if any better) this fancy FFT Wiener filter is compared to a simple robust mean, and I haven't done any comparisons yet. 

Step 5: postprocessing

The first step in postprocessing is to remove any remastering nusiance patterns in the data, masterly light pollution. In order to do this, I mark all "bright" regions of the image as invalid, and run a large stride+dilation median filter over the entire image. There is also a user specifiable bbox of pixels to ignore. I interpolate the result over the entire image, and the result is a pretty good "background" field, with no stars or objects of interest. 

<img src="https://github.com/ricsonc/aptools/blob/master/readme_imgs/gradient.png" width="500">
Contours of the estimated nusiance RGB gradient. (Note -- this image is out of date, new gradient method is much better).

There are 3 methods implemented for "interpolation". The first is a simple linear interpolation, which works the best for small, compact objects and when the flat frames aren't great. It has trouble dealing with outliers. The second method is an improvement over linear interp, where I subsample a small fraction of background points, interpolate, use some RANSAC-like strategy to decide which of the background are outliers. The third method fits a quadratic while clipping residuals, and is the most robust to outliers, but if your flat frames are not good enough, there will be problems.

The final preprocessing step clips the extremes of the images, applies gamma correction, and then applies a further user-specifiable spline tone curve. Then a minimal amount of manual editing or corrections are done in GIMP.

---

A single exposure computed to final output.

<img src="https://github.com/ricsonc/aptools/blob/master/readme_imgs/single_exposure.jpg" width="500">
<img src="https://github.com/ricsonc/aptools/blob/master/readme_imgs/triangulum.jpg" width="500">

(Triangulum galaxy, 150 exposures of 30 seconds @ 200mm f/2.8)

<img src="https://github.com/ricsonc/aptools/blob/master/readme_imgs/single_exposure_bl.jpg" width="500">
<img src="https://github.com/ricsonc/aptools/blob/master/readme_imgs/barnards_loop.jpg" width="500">

(Orion Complex, 150 exposures of 30 seconds @ 50mm f/1.8)

---

Usage tips:

1. Flat frames are pretty critical. Dark frames are not really necessary.
2. It's tricky to install the LMMSE demosaicking package. DCB is an ok substitute if you can't get it working.
3. In star_detect.py, tune lum_pthresh so that you no more than 25k stars -- otherwise it's slow.
4. In registration.py, start with max_stars=2000, reduce to 1000 or 500 if you get huge residuals (>1px).
5. In warping.py, set USE_TP to true if you want a nonlinear warp. This sometimes causes overshoot isues.
6. In post.py, you can specify an excl_box OR, even better, an image containing a mask of the object.
6. In post.py, adjust border_crop until no image edges are visible.
