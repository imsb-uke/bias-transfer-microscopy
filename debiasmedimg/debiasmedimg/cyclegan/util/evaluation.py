import numpy as np
from scipy.linalg import sqrtm
from skimage.metrics import structural_similarity as ssim
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input


def ssim_score(input1, input2):
    """
    Calculate the mean structural similarity index (SSIM) between two images
    :param input1: Image 1
    :param input2: Image 2 (must have the same shape as Image 1)
    :return: SSIM value
    """
    ssim_value = ssim(input1, input2, gaussian_weights=True, multichannel=True)
    return ssim_value


def scale_images(images, new_shape):
    """
    Scale images to a specified shape
    :param images: Images to re-scale
    :param new_shape: Shape to scale the images to
    """
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = tf.image.resize(image, new_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # store
        images_list.append(new_image.numpy())
    return np.asarray(images_list)


def get_fid(set_a, set_b):
    """
    Calculate the Fréchet Inception Distance (Evaluation metric conceptualised for GANs)
    :param set_a: Set of images of domain A
    :param set_b: Set of images of domain B
    """
    # prepare the inception v3 model
    model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))
    # Prepare images
    images_a = scale_images(set_a, [299, 299])
    images_b = scale_images(set_b, [299, 299])
    images_a = preprocess_input(images_a)
    images_b = preprocess_input(images_b)
    # calculate activations
    act1 = model.predict(images_a)
    act2 = model.predict(images_b)
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    score = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return score
