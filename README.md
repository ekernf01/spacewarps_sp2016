##README

This project will try a couple of machine learning techniques to separate images with genuine gravitational lensing from the rest. Contact Eric Kernfeld, ekernf01@uw.edu, for details.

####Code documentation

So far, the code is only documented using comments and docstrings. Consult the included Python class `AstroDataMunger` for details. Eventually, I want to turn `caller_script.py` into a documented example of usage.

####SExtractor files

This project uses SExtractor to obtain features on which to learn image classification. After installation of SExtractor, which I did via HomeBrew, I needed to mess with several default files: `default.conv`, `default.param`, `default.psf`, `default.sex`, and `default.som`. Here's what I did to get it working.

- Put them all in the directory with the `fits` files. For more information about the directory structure of this project, consult the docstring of the included Python class `AstroDataMunger`.
- `default.sex`: change this line:

 `CATALOG_NAME     test.cat      #name of the output catalog`

 to this:
 
 `CATALOG_NAME     STDOUT        #name of the output catalog`
 
- `default.param`: comment out every line. Consult your local astrophysicist about which features you actually need and comment them in sparingly. My output contains this:

_

    1 NUMBER                 Running object number                                     
    2 EXT_NUMBER             FITS extension number                                     
    3 FLUX_ISO               Isophotal flux                                             [count]
    4 FLUXERR_ISO            RMS error for isophotal flux                               [count]
    5 FLUX_AUTO              Flux within a Kron-like elliptical aperture                [count]
    6 FLUXERR_AUTO           RMS error for AUTO flux                                    [count]
    7 MAG_AUTO               Kron-like elliptical aperture magnitude                    [mag]
    8 MAGERR_AUTO            RMS error for AUTO magnitude                               [mag]
    9 FLUX_BEST              Best of FLUX_AUTO and FLUX_ISOCOR                          [count]
    10 FLUXERR_BEST           RMS error for BEST flux                                    [count]
    11 MAG_BEST               Best of MAG_AUTO and MAG_ISOCOR                            [mag]
    12 MAGERR_BEST            RMS error for MAG_BEST                                     [mag]
    13 XMIN_IMAGE             Minimum x-coordinate among detected pixels                 [pixel]
    14 YMIN_IMAGE             Minimum y-coordinate among detected pixels                 [pixel]
    15 XMAX_IMAGE             Maximum x-coordinate among detected pixels                 [pixel]
    16 YMAX_IMAGE             Maximum y-coordinate among detected pixels                 [pixel]
    17 XPEAK_IMAGE            x-coordinate of the brightest pixel                        [pixel]
    18 YPEAK_IMAGE            y-coordinate of the brightest pixel                        [pixel]
    19 XPEAK_FOCAL            Focal-plane x coordinate of the brightest pixel           
    20 YPEAK_FOCAL            Focal-plane y coordinate of the brightest pixel           
    21 X2_IMAGE               Variance along x                                           [pixel**2]
    22 Y2_IMAGE               Variance along y                                           [pixel**2]
    23 XY_IMAGE               Covariance between x and y                                 [pixel**2]
    24 CXX_IMAGE              Cxx object ellipse parameter                               [pixel**(-2)]
    25 CYY_IMAGE              Cyy object ellipse parameter                               [pixel**(-2)]
    26 CXY_IMAGE              Cxy object ellipse parameter                               [pixel**(-2)]
    27 A_IMAGE                Profile RMS along major axis                               [pixel]
    28 B_IMAGE                Profile RMS along minor axis                               [pixel]
    29 THETA_IMAGE            Position angle (CCW/x)                                     [deg]
    30 ERRX2_IMAGE            Variance of position along x                               [pixel**2]
    31 ERRY2_IMAGE            Variance of position along y                               [pixel**2]
    32 ERRXY_IMAGE            Covariance of position between x and y                     [pixel**2]
    33 ERRCXX_IMAGE           Cxx error ellipse parameter                                [pixel**(-2)]
    34 ERRCYY_IMAGE           Cyy error ellipse parameter                                [pixel**(-2)]
    35 ERRCXY_IMAGE           Cxy error ellipse parameter                                [pixel**(-2)]
    36 ERRA_IMAGE             RMS position error along major axis                        [pixel]
    37 ERRB_IMAGE             RMS position error along minor axis                        [pixel]
    38 ERRTHETA_IMAGE         Error ellipse position angle (CCW/x)                       [deg]
    39 FLAGS                  Extraction flags                                          
    40 FWHM_IMAGE             FWHM assuming a gaussian core                              [pixel]
    41 ELONGATION             A_IMAGE/B_IMAGE                                           
    42 ELLIPTICITY            1 - B_IMAGE/A_IMAGE                   