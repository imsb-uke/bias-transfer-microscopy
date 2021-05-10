import numpy as np


def cut_out_files(image_files, image_number):
    """
    Return a list of training file names with equal amount of images for each domain
    :param image_files: Image files to cut from
    :param image_number: Number of images per domain to use
    """
    files = [[] for _ in range(0, len(image_files))]
    for ix, domain in enumerate(image_files):
        files[ix].extend(domain[:image_number])
    files = [item for sublist in files for item in sublist]
    files = np.array([np.array(xi) for xi in files])
    return files
