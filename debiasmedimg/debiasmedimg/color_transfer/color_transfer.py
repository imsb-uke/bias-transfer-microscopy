import os
import random
import numpy as np
import pandas as pd
from matplotlib import image
import skimage.color
from skimage.exposure import cumulative_distribution
from copy import deepcopy
from debiasmedimg.cyclegan.util import get_sample_from_path, get_filenames, normalize_for_evaluation, \
            ssim_score, get_fid, get_all_samples, save_to_csv, get_filtered_filenames
import debiasmedimg.settings as settings


class ColorTransfer:
    """
    Encapsulates the color transfer approach, which is based on histogram equalization in the LAB color space
    """
    def __init__(self, csv_file, seed):
        """
        Initalize random seed and folder to load from and save to
        :param csv_file: File to read image paths from
        :param seed: Random seed
        """
        self.csv_file = csv_file
        # Set random seed
        self.seed = seed
        random.seed(seed)

    def apply_transfer(self, domain_to_transfer, target_domain):
        """
        Load in images, color transfer them and export them
        :param domain_to_transfer: Which domain to transform
        :param target_domain: Domain to transfer to
        """
        files_to_transfer = get_filtered_filenames(self.csv_file, domain_to_transfer)
        target_files = get_filtered_filenames(self.csv_file, target_domain)
        for path in files_to_transfer:
            img = image.imread(path)
            if img.shape[2] == 4:
                # Cut off alpha channel
                img = img[:, :, :-1]
                print("Cutting off alpha channel")            

            # Read in random target image
            target_img_file = random.choice([x for x in target_files])
            target_img = image.imread(target_img_file)
            if target_img.shape[2] == 4:
                # Cut off alpha channel
                target_img = target_img[:, :, :-1]

            # Color transfer images
            color_transferred_img = self.lab_color_transfer(img, target_img)
            filename = path.split(settings.DB_DIR)[1]
            # Cut off filename
            path_to_file, filename = filename.rsplit('/', 1)
            path_sample_out = settings.OUTPUT_DIR + "/generated_images/color_transfer/" + "to_" + \
                              target_domain + "/" + str(self.seed) + "/" + path_to_file + "/"
            if not os.path.exists(path_sample_out):
                os.makedirs(path_sample_out)
            image.imsave(path_sample_out + filename, color_transferred_img)
            print("Exported:", path_sample_out + filename)

    def lab_color_transfer(self, source, target):
        """
        Transfer color to a source image given a target
        :param source: Image to change
        :param target: Image to use for target colours
        :return: Color transferred image
        """
        # Convert the RGB images to the LAB color space
        lab_source = skimage.color.rgb2lab(source)
        lab_target = skimage.color.rgb2lab(target)
        # CDFs require image values as ints
        lab_source_int = self.lab_to_lab_int(lab_source)
        lab_target_int = self.lab_to_lab_int(lab_target)
        # Calculate the CDFs of the source and target imgs
        cdf_lab_source = self.cdf(lab_source_int)
        cdf_lab_target = self.cdf(lab_target_int)
        # Perform histogram matching
        lab_result_int = self.hist_matching(cdf_lab_source, cdf_lab_target, deepcopy(lab_source_int))
        lab_result_int = np.clip(lab_result_int, 0, 255)
        # Convert LAB to RGB
        lab_result = self.lab_int_to_lab(lab_result_int)
        result = skimage.color.lab2rgb(lab_result)
        return result

    @staticmethod
    def lab_to_lab_int(img):
        """
        Convert an image from regular lab to integer lab representation for histogram matching
        :param img: Image to transform
        """
        img[:, :, 0] = img[:, :, 0] * 255 / 100
        img[:, :, 1] = img[:, :, 1] + 127
        img[:, :, 2] = img[:, :, 2] + 127
        img = img.astype(np.uint8)
        return img

    @staticmethod
    def lab_int_to_lab(img):
        """
        Convert an image from integer lab representation to regular lab representation
        :param img: Image to transform
        """
        img = img.astype(np.float)
        img[:, :, 0] = img[:, :, 0] * 100 / 255
        img[:, :, 1] = img[:, :, 1] - 127
        img[:, :, 2] = img[:, :, 2] - 127
        return img

    @staticmethod
    def cdf(im):
        """
        Computes the CDF of an image im as 2D numpy ndarray
        :param im: Image to calculate the CDF of
        """
        cdf_rgb = []
        for i in range(3):
            c, b = cumulative_distribution(im[:, :, i])
            # pad the beginning and ending pixels and their CDF values
            c = np.insert(c, 0, [0] * b[0])
            c = np.append(c, [1] * (255 - b[-1]))
            cdf_rgb.append(c)
        cdf_rgb = np.array(cdf_rgb)
        return cdf_rgb

    @staticmethod
    def hist_matching(c, c_t, im):
        """
        Match the histograms via closest pixel-matches of the given CDFs
        :param c: CDF of input image computed with the function cdf()
        :param c_t: CDF of target image computed with the function cdf()
        :param im: input image as 2D numpy ndarray
        :return: modified pixel values
        """
        for ix, layer in enumerate(c):
            pixels = np.arange(256)
            # find closest pixel-matches corresponding to the CDF of the input image, given the value of the CDF H of
            # the template image at the corresponding pixels, s.t. c_t = H(pixels) <=> pixels = H-1(c_t)
            new_pixels = np.interp(c[ix], c_t[ix], pixels)
            im[:, :, ix] = (np.reshape(new_pixels[im[:, :, ix].ravel()], (im.shape[0], im.shape[1])).astype(np.uint8))
        return im

    def evaluate(self, validate, csv_file, domain_a, domain_b, dataset):
        """
        Evaluate the color transferred images regarding SSIM and FID
        :param validate: Whether we are validating or testing
        :param csv_file: CSV files containing info on all images used during the evaluation
        :param domain_a: Name of domain A
        :param domain_b: Name of domain b
        :param dataset: Name of the dataset
        """
        run_id = "Baseline"
        files_a = get_filtered_filenames(csv_file, domain_a)
        files_b = get_filtered_filenames(csv_file, domain_b)

        print("Evaluating set a")
        ssims_a = []
        for ix, path in enumerate(files_a):
            # Read in images in full size, remove batch dimension
            original_fullsize = np.squeeze(get_sample_from_path(path)[0])
            filename = path.split(settings.DB_DIR)[1]
            # Cut off filename
            path_to_file, filename = filename.rsplit('/', 1)
            path_sample_out = settings.OUTPUT_DIR + "/generated_images/color_transfer/" + "to_" + \
                              domain_b + "/" + str(self.seed) + "/" + path_to_file + "/"
            transformed_upsampled = np.squeeze(get_sample_from_path(path_sample_out + filename)[0])
            # Evaluate ssim
            original_fullsize = normalize_for_evaluation(original_fullsize)
            transformed_upsampled = normalize_for_evaluation(transformed_upsampled)
            # Get the SSIM scores between input and output of the generator
            ssim_inout = ssim_score(original_fullsize, transformed_upsampled)
            ssims_a.append(ssim_inout)
            print("Completed {}/{}".format(ix + 1, len(files_a)))
        ssim_a = sum(ssims_a) / len(files_a)

        # Read in all images again for FID and wasserstein distance on histograms
        a_fullsize = get_all_samples(files_a)
        transformed_files = [path_sample_out + f for f in os.listdir(path_sample_out)]
        to_b_upsampled = get_all_samples(transformed_files)
        b_fullsize = get_all_samples(files_b)

        # FID score:
        print("Calculating FID score between real domains and generated domains")
        fid_b = get_fid(b_fullsize, to_b_upsampled)
        fid_original = get_fid(a_fullsize, b_fullsize)

        # Add summary to csv file
        values = [ssim_a, fid_original, fid_b]
        save_to_csv(run_id, self.seed, domain_a, domain_b, values, "color transfer", dataset,
                    validate=validate, only_ab=True)
