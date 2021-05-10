import os
import numpy as np
import tensorflow as tf
from matplotlib import image
import cv2
import pandas as pd
import debiasmedimg.settings as settings
from copy import deepcopy


def get_filenames(csv_file, domains, merge=False):
    """
    Extract the filenames of all images in the folders for all domains
    :param csv_file: Path to the csv files containing info about the images
    :param domains: List of domains names
    :param merge: Whether to return one array of all images
    :return: filenames
    """
    csv_df = pd.read_csv(csv_file)
    number_of_domains = len(domains)
    files = [[] for _ in range(0, number_of_domains)]
    for index, row in csv_df.iterrows():
        # Only use images without issues
        if pd.isna(row["issues"]) and not pd.isna(row["img_path"]):
            # Find the id of the corresponding domain the current sample belongs to
            if row["origin"] in domains:
                domain_id = domains.index(row["origin"])
                files[domain_id].extend([settings.DB_DIR + row["img_path"]])
    if merge:
        files = [item for sublist in files for item in sublist]
    files = np.array([np.array(xi) for xi in files])
    return files

def get_filtered_filenames(csv_file, domain):
    """
    Extract the filenames of all images in the folders for all domains
    :param csv_file: Path to the csv files containing info about the images
    :param domain: Domain to return paths from
    :return: filenames
    """
    csv_df = pd.read_csv(csv_file)
    files = []
    for index, row in csv_df.iterrows():
        # Only use images without issues
        if pd.isna(row["issues"]) and row["origin"] == domain:
            # Find the id of the corresponding domain the current sample belongs to
            files.extend([settings.DB_DIR + row["img_path"]])
    files = np.array([np.array(xi) for xi in files])
    return files

def _downscale_img(img):
    """
    Function for downsampling to 256x256 pixels
    :param img: Image to downscale using a Gaussian pyramid
    :return: Downsampled images
    """
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
    return g


def minimal_pad(img, color=1):
    """
    Add minimal padding to the image to make sure the image is square-shaped
    :param img: Image to pad
    :param color: Color to pad in (default is white)
    """
    img_padded = deepcopy(img)
    if img.shape[0] < img.shape[1]:
        # Pad height
        padding_height = img.shape[1] - img.shape[0]
        padding_height_1 = int(padding_height / 2)
        padding_height_2 = int(padding_height / 2)
        if not padding_height % 2 == 0:
            padding_height_1 = int(padding_height / 2)
            padding_height_2 = int(padding_height / 2) + 1
        img_padded = np.pad(img, pad_width=((padding_height_1, padding_height_2), (0, 0),
                                            (0, 0)), mode='constant', constant_values=(color,))
    elif img.shape[1] < img.shape[0]:
        # Pad width
        padding_width = img.shape[0] - img.shape[1]
        padding_width_1 = int(padding_width / 2)
        padding_width_2 = int(padding_width / 2)
        if not padding_width % 2 == 0:
            padding_width_1 = int(padding_width / 2)
            padding_width_2 = int(padding_width / 2 + 1)
        img_padded = np.pad(img, pad_width=((0, 0), (padding_width_1, padding_width_2),
                                            (0, 0)), mode='constant', constant_values=(color,))
    return img_padded


def get_sample_from_path(path, color=1):
    """
    Get an image from the given path in original size and downscaled
    :param path: location of the image
    :param color: Color to use for padding image to a square shape (if necessary)
    :return: normalized image
    """
    # Read in image and downscale it
    x = image.imread(path)
    if x.shape[2] == 4:
        # If Image is read in as RGBA for some reason -> Can cut it off
        x = x[:, :, :-1]
    original_size = [deepcopy(x.shape[0]), deepcopy(x.shape[1])]
    if not x.shape[0] == x.shape[1]:
        x = minimal_pad(x, color=color)
    x_small = _downscale_img(x)
    # Normalize original size and small size
    x = normalize(x)
    x_small = normalize(x_small)
    # Add dimension to make samples out of individual images
    x = tf.expand_dims(x, axis=0)
    x_small = tf.expand_dims(x_small, axis=0)
    return x, x_small, original_size


def get_all_samples(file_list):
    """
    Read in all samples from a path for evaluation
    :param file_list: List of files to load
    :return: Np array of images
    """
    samples = []
    for file in file_list:
        img = image.imread(file)
        if img.shape[2] == 4:
            # Image is read in as RGBA for some reason, but all entries in A are 1 -> Can cut it off
            img = img[:, :, :-1]
        if not img.shape[0] == img.shape[1]:
            img = minimal_pad(img)
        samples.append(normalize_for_evaluation(img))
    samples = np.array([np.array(xi) for xi in samples])
    return samples


def get_domain_name(path, domains):
    """
    Given a path, extract the domain name
    :param path: Path to extract from
    :param domains: Possible domains the image could belong to
    :return:
    """
    domain_name = None
    for domain in domains:
        if domain in path:
            domain_name = domain
    assert domain_name is not None
    return domain_name


def get_real_samples(file_names, n_samples, batch_number, domains, return_domain_names=False, all_files=None):
    """
    Select a batch of random samples, returns images and their domain names
    :param file_names: Dataset to sample from
    :param n_samples: Number of samples
    :param batch_number: Current batch
    :param domains: List of domain names the samples could belong to
    :param return_domain_names: Whether to return the names of the domains of the images
    :param all_files: all files the samples are cut in a list of arrays with the order of the arrays reflecting the domains
    :return: Images and targets
    """
    # choose n instances
    # Images are shuffled randomly at the beginning of a new epoch
    ix = np.arange(batch_number * n_samples, (batch_number + 1) * n_samples)
    # ix = np.random.randint(0, len(file_names), n_samples)
    # retrieve selected images
    x = file_names[ix]
    # Replace paths with images
    samples = []
    domain_names = []
    for index, path in enumerate(x):
        # x[index] = _random_jitter(image)
        # Normalize images
        img = image.imread(path)
        if img.shape[2] == 4:
            # Image is read in as RGBA for some reason, but all entries in A are 1 -> Can cut it off
            img = img[:, :, :-1]
        if not img.shape[0] == img.shape[1]:
            img = minimal_pad(img)
        img = _downscale_img(img)
        samples.append(normalize(img))
        if return_domain_names and all_files is None:
            # get domain names based on path (domain has to be included in path)
            domain_name = get_domain_name(path, domains)
            domain_names.append(domain_name)
        elif return_domain_names:
            domain_idx = -1
            for idx, domain in enumerate(all_files):
                if path in domain:
                    domain_idx = idx
            assert not domain_idx == -1
            domain_name = domains[domain_idx]
            domain_names.append(domain_name)
    samples = np.array([np.array(xi) for xi in samples])
    if return_domain_names:
        # Return images and their domain names
        return samples, domain_names
    # Return images
    return samples


def create_patches(image_to_cut, patch_size=16):
    """
    This function takes an image and cuts it into a variable number of non-overlapping patches
    :param image_to_cut: a numpy array of an image
    :param patch_size: the size that the patches should have in the end
    Returns: an array of patches [n_patches x patch_size x patch_size x 3] and the number of patches
    """
    first_indices_horizontal = np.arange(0, image_to_cut.shape[0] - patch_size + 1, patch_size)
    first_indices_vertical = np.arange(0, image_to_cut.shape[1] - patch_size + 1, patch_size)
    # Calculate the number of patches
    number_resulting_patches = first_indices_horizontal.size * first_indices_vertical.size
    patches = np.zeros((number_resulting_patches, patch_size, patch_size, image_to_cut.shape[2]))
    patch_number = 0
    for idx_ver in first_indices_vertical:
        for idx_hor in first_indices_horizontal:
            patches[patch_number, ...] = np.array(image_to_cut[idx_ver:idx_ver + patch_size, idx_hor:idx_hor + patch_size, :])
            patch_number += 1
    return patches, patch_number


def normalize(img):
    """
    Normalize an image to range [-1,1]
    :param img: Image to normalize
    :return: Normalized image
    """
    old_min = np.amin(img)
    old_max = np.amax(img)
    # NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
    # NewMin = -1, NewMax = 1, NewRange = 1 - (-1)
    img = (img - old_min) * 2 / (old_max - old_min) - 1.0
    # Conv2D layers cast from float64 to float32, so make sure we have the correct type here
    img = tf.cast(img, tf.float32)
    return img


def normalize_for_display(img):
    """
    Normalize an image to range [0,1]
    :param img: Image to normalize
    :return: Normalized image
    """
    old_min = np.amin(img)
    old_max = np.amax(img)
    # NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
    # NewMin = 0, NewMax = 1, NewRange = 1 - 0
    img = (img - old_min) / (old_max - old_min)
    return img


def normalize_for_evaluation(img):
    """
    Normalize an image to range [0,255]
    :param img: Image to normalize
    :return: Normalized image
    """
    old_min = np.amin(img)
    old_max = np.amax(img)
    # NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
    # NewMin = 0, NewMax = 255, NewRange = 1 - 0
    img = (img - old_min) * 255 / (old_max - old_min)
    img = tf.cast(img, tf.int32)
    return np.asarray(img)


def save_to_csv(run_id, epoch, a_name, b_name, means, approach, dataset, validate=True, only_ab=False):
    """
    Save the evaluation results to a csv file
    :param run_id: Run id of the evaluated run
    :param epoch: Epoch that has been evaluated
    :param a_name: Name of domain A
    :param b_name: Name of domain b
    :param means: List of evaluation results
    :param approach: Which approach was used for transformation
    :param dataset: Name of the dataset that was bias transferred
    :param validate: Whether we are currently validating or testing
    :param only_ab: Whether we only evaluated a transformation from A to B, not the other way around
    """
    if only_ab:
        val_dict = [{"run_id": run_id, "epoch": epoch, "a": a_name, "b": b_name, "ssim_inout_a": means[0],
                    "fid_original": means[1], "fid_b": means[2]}]
        val_df = pd.DataFrame.from_dict(val_dict)
    else:
        val_dict = [{"run_id": run_id, "epoch": epoch, "a": a_name, "b": b_name,  "ssim_inout_a": means[0],
                    "ssim_inout_b": means[1], "fid_original": means[2], "fid_a": means[3], "fid_b": means[4]}]
        val_df = pd.DataFrame.from_dict(val_dict)

    if validate:
        if not os.path.isfile(settings.EVAL_DIR + "/" + dataset + "_validation_" + approach + ".csv"):
            val_df.to_csv(settings.EVAL_DIR + "/" + dataset + "_validation_" + approach + ".csv", header=True,
                          index=False, mode='w')
        else:
            val_df.to_csv(settings.EVAL_DIR + "/" + dataset + "_validation_" + approach + ".csv", header=False,
                          index=False, mode='a')

    else:
        if not os.path.isfile(settings.EVAL_DIR + "/" + dataset + "_test_" + approach + ".csv"):
            val_df.to_csv(settings.EVAL_DIR + "/" + dataset + "_test_" + approach + ".csv", header=True,
                          index=False, mode='w')
        else:
            val_df.to_csv(settings.EVAL_DIR + "/" + dataset + "_test_" + approach + ".csv", header=False,
                          index=False, mode='a')
