import subprocess
from astropy.io import fits
import matplotlib.pyplot as plt
from PIL import Image
import pickle as pkl
import os
import pandas as pd
import numpy as np
import warnings

PATH_TO_DATA = "../data"
PATH_TO_SEXTRACTOR = "/usr/local/Cellar/sextractor/2.19.5/bin/sex"
IMAGE_SHAPE = (96, 96, 3)
    
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
        features|all_obj
          0_ASW0001f17.pkl
          ...                  <-- it extracts features from each FITS and saves them here as pickled pandas dataframes.
          11511_ASW0000o2y.pkl
        features|1obj
          0_ASW0001f17.pkl
          ...                  <-- features for the center-most object
          11511_ASW0000o2y.pkl
        features|1obj
          0_ASW0001f17.pkl
          ...                  <-- features for the three brightest objects
          11511_ASW0000o2y.pkl
    """


    def __init__(self, path_to_data = PATH_TO_DATA, path_to_sextractor = PATH_TO_SEXTRACTOR,
                 image_shape = IMAGE_SHAPE, test_mode = False):
        self.counter = 0
        self.SEXTRACTOR_FT_OUT = 16
        self.path_to_data = path_to_data
        self.path_to_sextractor = path_to_sextractor
        self.catalog = pd.read_csv(os.path.join(path_to_data, 'catalog/catalog.csv'), index_col=0)
        self.num_images = self.catalog.shape[0]
        self.test_mode = test_mode
        self.num_examples_available = self.num_images
        self.image_shape = image_shape
        self.center_pixel = [(self.image_shape[i] + 1.0) / 2 for i in (0, 1)]
        self.test_set_idcs = np.random.choice(range(self.num_images), size = round(self.num_images / 10), replace = False)
        return


    def displayFits(self, image_path, title):
        """
        Whenever I needed a sanity check.
        """
        image_data = fits.open(image_path)[0].data
        plt.imshow(image_data)
        plt.title(title)
        plt.show()
        return


    def imgToRgbArray(self, img):
        """
        :param img: PILLOW image object
        :return: 3d np array, 96x96x3
        """
        return np.asarray(img).astype(np.int32)
    
    
    def make_feature_path(self, i, ZooID, num_obj):
        """
        The features from SExtractor get saved here.
        :param i: image index in catalog
        :param ZooID: ZooID in catalog
        :return:
        """
        assert num_obj in ("all_obj", "1obj", "3obj")
        return os.path.join(self.path_to_data, "cutouts", "features|" + num_obj, str(i) + "_" + ZooID + ".pkl")


    def make_img_name(self, i, ZooID, is_fits, c = None):
        """
        This puts together the name of an image from this dataset. The FITS files have and extra suffix
        "color_<c>.pkl" for <c> in {0, 1, 2}.
        :param i:
        :param ZooID:
        :param is_fits:
        :param c:
        :return:
        """
        if is_fits:
            extension = ".fits"
        else:
            extension = ".png"
        return str(i) + "_" + ZooID + "_color" + str(c) + extension

    def make_img_path(self, i, ZooID, is_fits, c = None):
        """
        Similar to make_img_name, but this returns a full path.
        :param i:
        :param ZooID:
        :param is_fits:
        :param c:
        :return:
        """
        core = str(i) + "_" + ZooID
        if is_fits:
            if c is None:
                raise Exception("In makepath, c should be 0, 1, or 2, not None.")
            path = os.path.join(self.path_to_data, "cutouts/fits", core + "_color" + str(c) + ".fits")
        else:
            path = os.path.join(self.path_to_data, "cutouts/png", core + ".png")
        return path


    def pngsToFits(self, overwrite = False):
        """
        Converts each PNG file to three FITS files, once per color.
        :param overwrite:
        :return:
        """
        for i, ZooID in enumerate(self.catalog['ZooID']):
            if self.test_mode and i > 5:
                break

            #find images
            basic_name =  str(i) + '_' + ZooID
            png_path = os.path.join(self.path_to_data, "cutouts/png", basic_name + ".png")
            img = Image.open(png_path)
            img_by_color = [self.imgToRgbArray(img)[:,:,c] for c in range(3)]
            for c in range(3):
                fits_path = self.make_img_path(i, ZooID, True, c)
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

    def saveFeatures(self, pick_up_where_left_off = False):
        """
        Calls S Extractor and saves the features to files.
        If pick_up_where_left_off, it will look for the first image that
         doesn't have extracted features and start with that.
        :return:
        """
        for num_obj in ("1obj", "3obj"):
            dir = os.path.join(self.path_to_data, "cutouts", "features|" + num_obj)
            if not os.path.exists(dir):
                os.mkdir(dir)
        for i, ZooID in enumerate(self.catalog['ZooID']):
            files_there = [os.path.exists(self.make_feature_path(i, ZooID, num_obj)) for num_obj in ("1obj", "3obj")]
            if pick_up_where_left_off and all(files_there):
                continue
            for c in [1]:
                warnings.warn("Only using channel 1 (second channel) in SExtractor features")
                sex_image_name = self.make_img_name(i, ZooID, True, c)
                # call sextractor
                try:
                    raw_features = subprocess.check_output([self.path_to_sextractor, sex_image_name],
                                                           cwd=os.path.join(self.path_to_data, "cutouts/fits"))
                except subprocess.CalledProcessError, e:
                    print "SExtractor stdout output:\n", e.output
                #make a dataframe for features for image i and put features in it
                if c==1:
                    sex_cols = self.extract_header(raw_features)
                    assert self.SEXTRACTOR_FT_OUT == len(sex_cols)
                    feature = pd.DataFrame(columns = sex_cols)
                for vector in self.parse_raw_features(raw_features):
                    feature.loc[len(feature), :] = vector
            if self.test_mode and i >= 5:
                break
            #save one image's worth
            with open(self.make_feature_path(i, ZooID, "1obj"), "wb") as f:
                pkl.dump(self.get_brightest(feature, "1obj"), f)
            with open(self.make_feature_path(i, ZooID, "3obj"), "wb") as f2:
                pkl.dump(self.get_brightest(feature, "3obj"), f2)

        return


    def increment_counter(self, cycle):
        self.counter += np.random.choice(range(50))
        #This block handles running out and/or cycling back
        outta_stuff = self.counter >= self.num_examples_available
        if cycle       and outta_stuff:
            warnings.warn("Returning to beginning of dataset.")
            self.counter = self.counter % self.num_examples_available
        if (not cycle) and outta_stuff:
            raise Exception("Ran out of images in nextImage.")
        return

    def nextExample(self, datum_type, binary_label = True, cycle = True, CV_type = "any"):
        """
         This is the intended interface from this class to machine learning algorithms.
         It alters the state of self.counter, so that it works like a generator,
         returning a new image every time. When cycle == True, which is the default, it
         will return to the beginning instead of running out of images. This is so that
         an SGD routine can make another pass.

         Returns a tuple containing a datum (first element)
         and a label (second element). Label is binary (zero for "dud", 1 otherwise)
         unless you set binary_label to False, in which case it gives a string, e.g.
         "simulated lensing cluster" or "dud".

         If datum_type == "image", then the datum is
         a 3d array of pixel values (built to be a tensorflow interface).
         If datum_type == "sextractor", then it gives a pd Series of features from
         S Extractor.

         The constructor builds in a random split of the data to 1/10 test, 9/10 training.
         If CV_type == "any" (default), then this function does what's outlined above.
         If CV_type == "test" (default), then it skips to the next test example.
         If CV_type == "train" (default), then it skips to the next train example.
        """

        #This block moves to the next image, skipping to test/trainset if necessary
        self.increment_counter(cycle)
        if CV_type == "any":
            pass
        elif CV_type == "test":
            while not self.counter in self.test_set_idcs:
                self.increment_counter(cycle)
        elif CV_type == "train":
            while self.counter in self.test_set_idcs:
                self.increment_counter(cycle)
        else:
            raise Exception("CV_type must be one of 'any', 'train', or 'test'.")

        #This block fills in the right type of label
        if binary_label:
            label = ("dud" != self.catalog['object_flavor'][self.counter])
        else:
            label = self.catalog['object_flavor'][self.counter]
        ZooID = self.catalog['ZooID'][self.counter]

        #This block fills in the right type of feature
        valid_types = ("image", "sextractor|1obj", "sextractor|3obj")
        assert datum_type in valid_types
        if datum_type == "image":
            png_path = os.path.join(self.path_to_data, "cutouts/png", str(self.counter) + '_' + ZooID + ".png")
            datum = self.imgToRgbArray(Image.open(png_path))
        else:
            assert datum_type[0:11] == "sextractor|"
            num_obj = datum_type[11:]
            with open(self.make_feature_path(self.counter, ZooID, num_obj), "rb") as f:
                datum = pkl.load(f)
        return datum, label



    def get_batch(self, batch_size, CV_type):
        """
        Return a tuple of numpy arrays images, labels.
        images is the images, RBG. labels is the labels, boolean. Nevertheless, Theano may read them as floats.
        images.shape is (batch_size, 3, 96, 96) while labels.shape is batch_size.

        :param batch_size:
        :param CV_type: see nextExample documentation.
        :return:
        """

        images = np.zeros((batch_size, self.image_shape[2], self.image_shape[0], self.image_shape[1]))
        labels = np.zeros(batch_size, dtype = "int32")
        for im_idx in range(batch_size):
            (img, lbl) = self.nextExample(datum_type="image", CV_type=CV_type)
            images[im_idx,...] = img.transpose((2, 0, 1))
            labels[im_idx] = lbl
        return images, labels

    def get_brightest(self, objects, num_obj):
        """
        Takes a PD dataframe of SExtractor extracted features and returns
        the features from the three brightest objects.
        Performs some additional computation: instead of just returning the features listed in
        the default.param file discussed in the readme about SExtractor files, I add in the
        distance and delta x and delta y from the brightest object for the second and third objects.
        :return:
        """

        #sort
        objects_by_flux = objects.sort_values("FLUX_ISO", ascending=False)

        #pad with zeros
        num_needed = int(num_obj[0]) - objects_by_flux.shape[0]
        if num_needed > 0:
            for n in range(num_needed):
                objects_by_flux.loc[objects_by_flux.shape[0], :] = np.zeros(self.SEXTRACTOR_FT_OUT)

        if num_obj == "1obj":
            return(np.array(objects_by_flux.iloc[0,:]))

        assert num_obj == "3obj"
        brightest_three = objects_by_flux.iloc[0:3,:]
        brightest_three.index = (0,1,2)
        if self.test_mode:
            assert all([brightest_three.loc[0, "FLUX_ISO"] >= flux for flux in objects_by_flux.loc[:, "FLUX_ISO"]])
        features = list(brightest_three.loc[0, :])
        #Add in xdist, y dist, and total dist
        for rank in (1, 2):
            delta_x = brightest_three.loc[rank, "X_IMAGE"] - brightest_three.loc[0, "X_IMAGE"]
            delta_y = brightest_three.loc[rank, "Y_IMAGE"] - brightest_three.loc[0, "Y_IMAGE"]
            distance = np.sqrt(delta_x ** 2 + delta_y ** 2)
            features.extend(brightest_three.iloc[rank,])
            features.extend((delta_x, delta_y, distance))
        return np.array(features)