---

There's already a lot of software out there for assisting with astrophotography. But I don't really trust any code I didn't write myself, and since astrophotography is heavily reliant on data processing (compared to "conventional" photography), I thought I should write my own set of tools. The goal in mind is to have simple and clean implementations of all the necessary algorithms to go from the raw data to a nearly finished image.

Of course, even though I tried out this code on a few seperate data sources, it's probably not nearly as robust as other available codes, and doesn't run as fast either. Some of the steps require a bit of parameter tuning. I do think i've improved on the "conventional implementations" of some processing steps though.

---

# Usage:

Create a `$work_dir` and a `raw` subdir inside that. Put all the raw files there. Optionally, put darks in the subdir `dark` and flats in `flats`. 

`python go.py command $work_dir`

Where command is one of `setup`, `dark`, `flat`, `demosaic`, `detect`, `register`, `relregister`, `warp`, `stack`, `post`. Commands should be run in this order, with `dark` and `relregister` usually being unnecessary.

Edit values in `config.py` as needed.

---

## Calibration

First we need to correct for several types of bias. Dark current, which is when which the camera sensor reads non-zero values even when no light is present, and vignetting, which is when the camera sensor reads different values even under uniform light. So called "dark" and "flat" frames are used for this purpose. The `dark` and `flats` command generates a master dark and flat frame which is used to calibrate all other images respectively.

Measured dark current example:
<img src="https://github.com/ricsonc/aptools/blob/master/readme_imgs/dark_current.png" width="500">

On my sensor, with 15 second exposures, non-uniform patterns in the dark current were limited to roughly 7 units in a 14-bit range -- with nonuniformity was only present in the bottom 10% of the sensor. So it can probably be omitted without harm.

Wavelength dependent vignetting:
<img src="https://github.com/ricsonc/aptools/blob/master/readme_imgs/vignette_model.png" width="500">
For vignetting, I tried both a parametric model with data coming from the [lensfun project](https://lensfun.github.io/), and by directly estimating via "flat frames". The parametric lensfun model was nowhere near accurate enough. The flat frame approach was much closer, but still had brightness deviations on the order of 1-2% (much better than ~50% uncorrected).

### Usage tips:
- Dark calibration is probably unnecessary as modern sensors have pretty low dark current.

## Demosaicing

For demosaicing (`demosaic`), I used the [rawpy](https://github.com/letmaik/rawpy) library, which interfaces with libraw. I used the [LMMSE demosaicing method](https://www4.comp.polyu.edu.hk/~cslzhang/paper/LMMSEdemosaicing.pdf), as I've read it's the best in high noise situations.

## Star Detection

I actually found very few references for exactly how star detection (`detect`) is done, but I went with a pretty "common sense" approach. In the preliminary pass, I convolve a bunch of gaussians of different scales with the input, and take all the local maximums (if they're bright enough) as candidate stars. Then I extract an image patch around each candidate, and fit a specific gaussian (mean, cov, height) to each patch with levenberg marquardt. Only round and bright candidates are kept. A final pass filters out any duplicates.

<img src="https://github.com/ricsonc/aptools/blob/master/readme_imgs/star_detection.png" width="500">

A "novelty" I added here is in the gaussing fitting step. From my reading, other software assumes non-saturated pixels when fitting gaussians, which is a pretty bad assumption, as stars are often saturated. In my gaussian fitting code, I clip the gaussian at 1, which should let me obtain pretty good estimates even in saturated case. Of course this is such a simple tweak it's hard to imagine no one else has done it.

### Usage tips:
- Set the value of `detection_params.star_detect` such that at most 25k candidates are found, 5k is usually a good amount.

## Star Correspondence

The next step is to match stars across images (`register`). I implemented the [classic triangle similarity algorithm](https://hal.inria.fr/inria-00548539/document), which worked pretty well. As opposed to the original prescription, as well as other variations of the algorithm, my triangle construction implementation uses only smaller "local" triangles. 

<img src="https://github.com/ricsonc/aptools/blob/master/readme_imgs/registration.png" width="500">

The algorithm works surprisingly well, but often needs to have some parameters tweaked depending on the input. I made up some thresholds on what amount of triangle votes would qualify as a match. After the initial matching stage, I included a RANSAC linear fitting of the positions of the matched stars, with all outliers being discarded. This very effectively discards all the mismatches. 

### Usage tips:
- A `registration_params.max_stars` value of between 500 and 2000 usually works best, more is bad.
- If large residuals (>1 px) are reported, reduce `max_stars`

## Warping

In this step (`warp`), I align all the images so the pixels are hopefully in perfect correspondence using the star correspondences from the last step.

This can be done either with a linear fit, or also with a [thin-plate spline model](https://en.wikipedia.org/wiki/Thin_plate_spline) for the mapping. The reason for this is to account for possible lens distortions or other deformations. Practically speaking, I found that my average residual with a linear fit is typically 0.2 pixels, so it likely doesn't matter either way. However, this turns out to be essential for combining HA with visual images, due to deformations introduced by the filter. 

Plot of the deformation magnitude estimated by the thin-plate spline. Bright yellow ~= 1 pixel.
<img src="https://github.com/ricsonc/aptools/blob/master/readme_imgs/deform.png" width="500">

### Usage tips:
- When using a low focal length, or you have some reason to believe you have nonlinear deformations, turn `warping_params.use_thinplate` on.
- By default, the thin-plate spline perfectly interpolates all points. You can add regularization by adjusting the `thinplate_smoothing` parameter, but this is very tricky to tune, and large values are necessary (roughly 10^9 seemed to work) to avoid bad overshoot behavior. 

## Stacking

The most common methods for combining many exposures into a single image to reduce noise include average, median, and robust average (aka throw away outliers too far from the current average, and repeat). While this is extremely simple and works fairly well, I've been following some interesting computational photography literature, and methods for camera burst photography, where the same problem (of image stacking) exists. In particular, [Hasinoff et al.](https://people.csail.mit.edu/hasinoff/pubs/HasinoffEtAl16-hdrplus.pdf) suggests computing an FFT of the image (patches) and doing a robust average in that space. This is apparently called Wiener filtering. I implemented this approach, which was a bit challenging since I know absolutely nothing about signals processing. A very tricky "gotcha" concerns the noise term, which must be carefully scaled according to several details in the implementation, including the normalization convention of the FFT.

I suspect that since estimating flow on natural photographs with handshake and fast motion is significantly more challenging than aligning astrophotography images, these burst photography algorithms are probably designed more for coping with alignment failure rather than pure denoising. So I'm not entirely sure how much better (or if any better) this fancy FFT Wiener filter is compared to a simple robust mean, and I haven't done any comparisons yet. 

## Postprocessing

The first step in postprocessing (`post`) is to remove any remastering nusiance patterns in the data, mainly light pollution, also known as "background gradient". In order to do this, I mark all "bright" regions of the image as invalid, and run a large stride+dilation median filter over the entire image. There is also a user specifiable bbox of pixels to ignore. I interpolate the result over the entire image, and the result is a pretty good "background" field, with no stars or objects of interest. 

<img src="https://github.com/ricsonc/aptools/blob/master/readme_imgs/gradient.png" width="500">
Contours of the estimated nusiance RGB gradient. (Note -- this image is out of date, new gradient method is much better).

There are 3 methods implemented for "interpolation". The first is a simple linear interpolation, which works the best for small, compact objects and when the flat frames aren't great. It has trouble dealing with outliers. The second method is an improvement over linear interp, where I subsample a small fraction of background points, interpolate, use some RANSAC-like strategy to decide which of the background are outliers. The third method fits a quadratic while clipping residuals, and is the most robust to outliers, but if your flat frames are not good enough, there will be problems.

The final preprocessing step clips the extremes of the images, applies gamma correction, and then applies a further user-specifiable spline tone curve. Then a minimal amount of manual editing or corrections are done in GIMP.

### Usage tips:
- In the first pass, visualize the gradient and write down values for `postprocessing_params.excl_box`.
- In the second pass, don't use the cached background (recompute instead), and you should see that the visualized gradient is much smaller.

---

A single exposure computed to final output.

<img src="https://github.com/ricsonc/aptools/blob/master/readme_imgs/single_exposure.jpg" width="500">
<img src="https://github.com/ricsonc/aptools/blob/master/readme_imgs/triangulum.jpg" width="500">

(Triangulum galaxy, 150 exposures of 30 seconds @ 200mm f/2.8)

<img src="https://github.com/ricsonc/aptools/blob/master/readme_imgs/single_exposure_bl.jpg" width="500">
<img src="https://github.com/ricsonc/aptools/blob/master/readme_imgs/barnards_loop.jpg" width="500">

(Orion Complex, 150 exposures of 30 seconds @ 50mm f/1.8)

