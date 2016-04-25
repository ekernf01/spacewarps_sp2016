import sewpy
from astropy.io import fits
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import os
import pandas as pd
import numpy as np
import warnings

# This class has various utilities to help manipulate telescope image data.
# In addition to the correct PATH_TO_SEXTRACTOR,
# it expects the following directory structure:
#
# code
#   AstroImageMunger.py      <-- run it from here
# data
#   catalog
#     catalog.csv            <-- contains image metadata
#   cutouts
#     png
#       0_ASW0001f17.png
#       ...                  <-- your images
#       11511_ASW0000o2y.png
#     fits
#       default.param        <-- it requires several sextractor default files to be here.
#       0_ASW0001f17.fits
#       ...                  <-- it converts your PNGs to FITS.
#       11511_ASW0000o2y.fits
#
#

class AstroImageMunger:

    PATH_TO_DATA = "../data"
    PATH_TO_SEXTRACTOR = "/usr/local/Cellar/sextractor/2.19.5/bin/sex"
    IMAGE_SHAPE = (96, 96, 3)
    def __init__(self, path_to_data = PATH_TO_DATA, path_to_sextractor = PATH_TO_SEXTRACTOR):
        self.counter = 0
        self.path_to_data = path_to_data
        self.path_to_sextractor = path_to_sextractor
        self.catalog = pd.read_csv(os.path.join(path_to_data, 'catalog/catalog.csv'), index_col=0)
        self.num_images = self.catalog.shape[0]
        return
    
    def PIL_img_to_np_array(self, img):
        pixvals_np = np.array(img.getdata())
        pixvals_np = pixvals_np.reshape((img.size[0], img.size[1], 3))
        return pixvals_np

    
    def display_fits(self, image_path):
        image_data = fits.open(image_path)[1].data
        plt.imshow(image_data, cmap='gray')
        return
    
    
    def convert_pngs_to_fits(self, test_mode = False):
        for i, ZooID in enumerate(self.catalog['ZooID']):
            some_exist_already = False
            if test_mode and i > 5:
                break

            #find images
            basic_name =  str(i) + '_' + ZooID
            png_path = os.path.join(self.path_to_data, "cutouts/png", basic_name + ".png")
            fits_path = os.path.join(self.path_to_data, "cutouts/fits", basic_name + ".fits")

            #open, convert, save
            if os.path.exists(fits_path):
                some_exist_already = True
                continue
            img = Image.open(png_path)
            hdul = fits.HDUList()
            hdul.append(fits.PrimaryHDU())
            img_as_np = PIL_img_to_np_array(img)
            for i in range(3):
                hdul.append(fits.ImageHDU(data = img_as_np[:,:,i]))
            hdul.writeto(fits_path)
            if some_exist_already:
                warnings.warn("Not converting >=1 file because the FITS version existed already.")
        return

    
    def get_features(self, test_mode = True):
        _ = sewpy.SEW(params=[], config={})
        sew = sewpy.SEW(sexpath=self.path_to_sextractor,
                        params=_.fullparamlist,
                        config={"DETECT_MINAREA":10, "PHOT_FLUXFRAC":"0.3, 0.5, 0.8"},
                        workdir=os.path.join(self.path_to_data, "cutouts/fits"))

        features = [None for ZooID in self.catalog['ZooID']]
        for i, ZooID in enumerate(self.catalog['ZooID']):
            image_path = os.path.join(self.path_to_data, "cutouts/fits", str(i) + '_' + ZooID + ".fits")
            if test_mode:
                self.display_fits(image_path)
            if test_mode and i >= 5:
                break
            features[i] = sew(image_path)

        return features
