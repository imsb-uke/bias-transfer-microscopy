from copy import deepcopy
import cv2
import numpy as np
from matplotlib import image
import matplotlib.pyplot as plt
import tensorflow as tf


def downsampling(originals):
    """
    Function for downsampling multiple images to 256x256 pixels.
    If an image is smaller than 256x256 the original is returned
    :param originals: Images to downscale using a Gaussian pyramid
    :return: Downsampled images
    """
    downsampled_originals = []
    for img in originals:
        assert img.shape[0] == img.shape[1], "Images have to be squares"
        # Find largest power of two that is less than the image size
        expo = img.shape[0].bit_length() - 1
        # Make sure image isn't smaller than 256x256 pixels
        if expo < 8:
            return img
        img = cv2.resize(img, dsize=(2 ** expo, 2 ** expo), interpolation=cv2.INTER_CUBIC)
        g = img.copy()
        # Resize image to 256x256 (=2**8)
        for i in range(expo - 8):
            g = cv2.pyrDown(g)
        downsampled_originals.append(g)
    downsampled_originals = np.array(downsampled_originals)
    return downsampled_originals


def remove_minimal_pad(img, original_shape):
    """
    Removes the padding that was added before downsampling an image
    :param img: Image to remove the padding from
    :param original_shape: Shape the image should have after removing the padding
    """
    if original_shape[0] < original_shape[1]:
        # Pad height
        cut_height = original_shape[1] - original_shape[0]
        cut_height_1 = int(cut_height / 2)
        cut_height_2 = int(cut_height / 2)
        if not cut_height % 2 == 0:
            cut_height_1 = int(cut_height / 2)
            cut_height_2 = int(cut_height / 2) + 1
        img_cut = img[cut_height_1:-cut_height_2, :, :]
    elif original_shape[1] < original_shape[0]:
        # Pad width
        cut_width = original_shape[0] - original_shape[1]
        cut_width_1 = int(cut_width / 2)
        cut_width_2 = int(cut_width / 2)
        if not cut_width % 2 == 0:
            cut_width_1 = int(cut_width / 2)
            cut_width_2 = int(cut_width / 2 + 1)
        img_cut = img[:, cut_width_1:-cut_width_2, :]
    else:
        img_cut = img
    return img_cut


def laplacian_upsampling(originals, inputs, original_shape):
    """
    Perform upsampling of generated images as explained by Engin in the CycleDehaze paper (2018)
    :param originals: Input the images were generated from (original size)
    :param inputs: Generated images (small size)
    :param original_shape: Shape of the original image
    :return: Generated images (original size
    """
    upsampled_inputs = []
    for idx, img in enumerate(originals):
        assert img.shape[0] == img.shape[1], "Images have to be squares"
        # Find largest power of two that is less than the image size
        expo = img.shape[0].bit_length() - 1
        img = cv2.resize(img, dsize=(2 ** expo, 2 ** expo), interpolation=cv2.INTER_CUBIC)

        # Calculate laplacian pyramid
        # Downsample original
        g_pyramid = []
        ga = img.copy()
        g_pyramid.append(ga.copy())
        # Downsample image to 256x256 (=2**8)
        for i in range(expo - 8):
            ga = cv2.pyrDown(ga)
            g_pyramid.append(ga.copy())
        l_pyramid = []
        for i in range(expo - 8):
            lap = cv2.subtract(g_pyramid[i], cv2.pyrUp(g_pyramid[i + 1]))
            l_pyramid.append(lap.copy())
        # Last element of g pyramid is last element of l pyramid
        l_pyramid.append(g_pyramid[-1].copy())
        # Laplacian upsampling based on laplacian pyramid of the original
        up_pyramid = []
        up = inputs[idx]
        up_pyramid.append(up.copy())
        for i in range(expo - 8):
            up = cv2.pyrUp(up) + l_pyramid[expo - (9 + i)]
            up_pyramid.append(up.copy())
        upsampled = up_pyramid[-1].copy()
        upsampled = np.clip(upsampled, -1, 1)
        # Re-size image to have original size
        original_size = max(original_shape[0], original_shape[1])
        upsampled = cv2.resize(upsampled, dsize=(original_size, original_size), interpolation=cv2.INTER_CUBIC)
        # Have to cut off padding:
        upsampled_cut = remove_minimal_pad(upsampled, original_shape)
        upsampled_inputs.append(upsampled_cut)
    upsampled_inputs = np.array(upsampled_inputs)
    return upsampled_inputs
