import subprocess
from astropy.io import fits
import matplotlib.pyplot as plt
from PIL import Image
import pickle as pkl
import os
import pandas as pd
import numpy as np
import warnings


class AstroImageMunger:
    """
    This class has various utilities to help manipulate telescope image data.
    In addition to the correct PATH_TO_SEXTRACTOR,
    it expects the following directory structure:

    code
      AstroImageMunger.py      <-- run it from here
    data
      catalog
        catalog.csv            <-- contains image metadata.
                                   This should have a column "ZooID" such that the image names are
                                   "0_" + <zeroth image ZooID> + ".png", "1_" + <first image ZooID> + ".png", etc.
      cutouts
        png
          0_ASW0001f17.png
          ...                  <-- your images
          11511_ASW0000o2y.png
        fits
          default.param        <-- it requires several sextractor default files to be here. See README.

          0_ASW0001f17_color0.fits
          0_ASW0001f17_color1.fits
          0_ASW0001f17_color2.fits
          ...                  <-- it converts each PNG to three FITS images, one for each color.
          11511_ASW0000o2y.fits
        features
          0_ASW0001f17.txt
          ...                  <-- it extracts features from each FITS and saves them here.
          11511_ASW0000o2y.txt
    """

    PATH_TO_DATA = "../data"
    PATH_TO_SEXTRACTOR = "/usr/local/Cellar/sextractor/2.19.5/bin/sex"
    IMAGE_SHAPE = (96, 96, 3)
    def __init__(self, path_to_data = PATH_TO_DATA, path_to_sextractor = PATH_TO_SEXTRACTOR, test_mode = False):
        self.counter = 0
        self.path_to_data = path_to_data
        self.path_to_sextractor = path_to_sextractor
        self.catalog = pd.read_csv(os.path.join(path_to_data, 'catalog/catalog.csv'), index_col=0)
        self.num_images = self.catalog.shape[0]
        self.test_mode = test_mode
        return

    
    def displayFits(self, image_path, title):
        image_data = fits.open(image_path)[0].data
        plt.imshow(image_data)
        plt.title(title)
        plt.show()
        return
    

    def imgToRgbArrays(self, img):
        pixvals_np = np.asarray(img).astype(np.int32)
        return [pixvals_np[:,:, i] for i in range(3)]


    def makename(self, i, ZooID, is_fits, c = None):
        if is_fits:
            extension = ".fits"
        else:
            extension = ".png"
        return str(i) + "_" + ZooID + "_color" + str(c) + extension

    def makepath(self, i, ZooID, is_fits, c = None):
        core = str(i) + "_" + ZooID
        if is_fits:
            if c is None:
                raise Exception("In makepath, c should be 0, 1, or 2, not None.")
            path = os.path.join(self.path_to_data, "cutouts/fits", core + "_color" + str(c) + ".fits")
        else:
            path = os.path.join(self.path_to_data, "cutouts/png", core + ".png")
        return path


    def pngsToFits(self, overwrite = False):
        for i, ZooID in enumerate(self.catalog['ZooID']):
            if self.test_mode and i > 5:
                break

            #find images
            basic_name =  str(i) + '_' + ZooID
            png_path = os.path.join(self.path_to_data, "cutouts/png", basic_name + ".png")
            img = Image.open(png_path)
            img_by_color = self.imgToRgbArrays(img)
            for c in range(3):
                fits_path = self.makepath(i, ZooID, True, c)
                #open, convert, save
                if os.path.exists(fits_path):
                    if overwrite:
                        warnings.warn("Overwriting FITS files.")
                        os.remove(fits_path)
                    else:
                        warnings.warn("Leaving existing FITS files untouched.")
                        continue
                fits.writeto(fits_path, data = img_by_color[c])
        return


    def extract_header(self, raw_features):
        lines = raw_features.splitlines()
        feature_names = []
        # Lines look like this:
        #    #  42 ELLIPTICITY            1 - B_IMAGE/A_IMAGE
        # We want "ELLIPTICITY".
        for line in lines:
            if line[0] == '#':
                feature_names.append(line.split()[2])
        return feature_names


    def parse_raw_features(self, raw_features):
        """
        Takes whatever SExtractor spit into STDOUT and converts it into a list of lists of numbers ("rows").
        Each row of the output contains features from a certain object.
        :param raw_features:
        :return:
        """
        lines = raw_features.splitlines()
        relevant_lines = []
        for line in lines:
            if line[0] == '#':
                continue
            numbers = [float(x) for x in line.split()]
            relevant_lines.append(numbers)
        return relevant_lines

    def saveFeatures(self):
        """
        Calls S Extractor and saves the features to files.
        :return:
        """
        for i, ZooID in enumerate(self.catalog['ZooID']):
            if i % 500 == 1:
                print "Progress in feature extraction:"
                print features.tail()
            for c in range(3):
                sex_image_name = self.makename(i, ZooID, True, c)
                py_image_path  = self.makepath(i, ZooID, True, c)
                is_dud = "dud" != self.catalog['object_flavor'][i]
                metadata = [is_dud, i, ZooID, c]
                try:
                    raw_features = subprocess.check_output([self.path_to_sextractor, sex_image_name],
                                                           cwd=os.path.join(self.path_to_data, "cutouts/fits"))
                except subprocess.CalledProcessError, e:
                    print "SExtractor stdout output:\n", e.output

                if (i==0 and c==0):
                    sex_cols = self.extract_header(raw_features)
                    features = pd.DataFrame(columns = ["is_dud", "image_number", "ZooID", "color"] + sex_cols)

                for vector in self.parse_raw_features(raw_features):
                    features.loc[len(features), :] = metadata + vector

                #if self.test_mode:
                #    self.displayFits(py_image_path, title=sex_image_name)
            if self.test_mode and i >= 5:
                break
        with open(os.path.join(self.path_to_data + "/catalog/features.pkl"), "wb") as f:
            pkl.dump(features, f)
        return


    def loadFeatures(self):
        with open(os.path.join(self.path_to_data + "/catalog/features.pkl")) as f:
            features = pkl.load(f, "rb")
        return features


    def nextImage(self, binary_label = True, cycle = True):
        """
        # This is the intended interface to TensorFlow.
        # Returns a tuple containing a 3d array of pixel values (first element)
        # and a label (second element). Label is binary (zero for "dud", 1 otherwise)
        # unless you set binary_label to False, in which case it gives a string, e.g.
        # "simulated lensing cluster" or "dud".
        #
        # When cycle == True, which is the default, it will return
        # to the beginning instead of running out of images. This is so
        # an SGD routine can make another pass.
        """
        if cycle       and self.counter >= self.num_images:
            self.counter = 0
        if (not cycle) and self.counter >= self.num_images:
            raise Exception("Ran out of images in nextImage.")
        if binary_label:
            label = "dud" != self.catalog['object_flavor'][self.counter]
        else:
            label = self.catalog['object_flavor'][self.counter]
        ZooID = self.catalog['ZooID'][self.counter]
        png_path = os.path.join(self.path_to_data, "cutouts/png", str(self.counter) + '_' + ZooID + ".png")
        pixels = self.imgToArray(Image.open(png_path))
        self.counter += 1
        return pixels, label