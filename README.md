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
 
- `default.param`: comment out every line. Consult your local astrophysicist about which features you actually need and comment them back in sparingly. My output contains the items below for either the center-most object or the brightest 3 objects. For the second and third object, I add in the distance and delta x and delta y from the brightest object, but that happens later in Python.

_
	
	FLUX_ISO               Isophotal flux                                             [count]
	FLUX_AUTO              Flux within a Kron-like elliptical aperture                [count]
	MAG_AUTO               Kron-like elliptical aperture magnitude                    [mag]
	FLUX_BEST              Best of FLUX_AUTO and FLUX_ISOCOR                          [count]
	MAG_BEST               Best of MAG_AUTO and MAG_ISOCOR                            [mag]
	CXX_IMAGE              Cxx object ellipse parameter                               [pixel**(-2)]
	CYY_IMAGE              Cyy object ellipse parameter                               [pixel**(-2)]
	CXY_IMAGE              Cxy object ellipse parameter                               [pixel**(-2)]
	A_IMAGE                Profile RMS along major axis                               [pixel]
	B_IMAGE                Profile RMS along minor axis                               [pixel]
	THETA_IMAGE            Position angle (CCW/x)                                     [deg]
	FWHM_IMAGE             FWHM assuming a gaussian core                              [pixel]
	ELONGATION             A_IMAGE/B_IMAGE                                          
	ELLIPTICITY            1 - B_IMAGE/A_IMAGE 
	X_IMAGE                Object position along x                                   [pixel]
	Y_IMAGE                Object position along y                                   [pixel]

 
