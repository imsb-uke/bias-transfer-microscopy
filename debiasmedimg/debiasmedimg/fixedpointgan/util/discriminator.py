import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, LeakyReLU, Activation, GlobalAveragePooling2D, \
    Flatten, Dense, Add
from tensorflow.keras.initializers import RandomNormal
from tensorflow_addons.layers import InstanceNormalization
from tensorflow import squeeze
from tensorflow.keras.layers import concatenate
from .generator import pad


def define_discriminator(image_shape, n_labels, n_hidden=6):
    """
    Create the discriminator model and compile it
    :param image_shape: Shape of the images used
    :param n_labels: How many labels exist
    :param n_hidden: Number of hidden layers
    :return: Discriminator model
    """
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # source image input
    in_image = Input(shape=image_shape)
    # Number of units in the first hidden layer
    n_filters = 64
    d = in_image
    for i in range(n_hidden):
        d = pad(d, 1)
        d = Conv2D(filters=n_filters, kernel_size=4, strides=2, input_shape=image_shape, padding='valid',
                   use_bias=False, kernel_initializer=init)(d)
        d = LeakyReLU(alpha=0.01)(d)
        # Number of hidden units doubles after each layer
        n_filters = n_filters * 2
    # ------------Output source-----------------------
    # (batch, h/64, w/64, 2048) ==> (batch, h/64, w/64, 1) #patch GAN
    patch_out = pad(d, 1)
    patch_out = Conv2D(filters=1, kernel_size=3, strides=1, padding='valid',
                       kernel_initializer=init)(patch_out)
    output_src = patch_out
    # ------------Output class------------------------
    # (batch, h/64, w/64, 2048) ==> (batch, 1, 1, num_classes)
    k_size = int(image_shape[0] / 64)
    label_out = Conv2D(filters=n_labels, kernel_size=k_size, strides=1, padding='valid', kernel_initializer=init)(d)
    output_cls = tf.reshape(label_out, [-1, n_labels])
    # define model
    model = Model(in_image, [output_src, output_cls])
    model.summary()
    return model


def define_cyclegan_discriminator(image_shape, n_labels):
    """
    Create the discriminator model and compile it
    :param image_shape: shape of the input image
    :param n_labels: Number of labels
    :return: Discriminator model
    """
    # Weight initialization
    init = RandomNormal(stddev=0.02)
    # Source image input
    in_image = Input(shape=image_shape)
    # C64
    d = Conv2D(64, (4, 4), input_shape=image_shape, strides=(2, 2), padding='same',
               kernel_initializer=init)(in_image)
    d = LeakyReLU(alpha=0.2)(d)
    # C128
    d = Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    # InstanceNormalization = normalized the values on each feature map,
    # the intent is to remove image-specific contrast information from the image
    # axis=-1 normalize features per feature map
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C256
    d = Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C512
    d = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # Second to last output layer
    d = Conv2D(512, (4, 4), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # ------------Output source-----------------------
    # Patch output
    output_src = Conv2D(1, (4, 4), padding='same', kernel_initializer=init)(d)
    # ------------Output class------------------------
    # (batch, h/64, w/64, 2048) ==> (batch, 1, 1, num_classes)
    k_size = int(image_shape[0] / 64)
    label_out = Conv2D(filters=n_labels, kernel_size=k_size, strides=1, padding='valid', kernel_initializer=init)(d)
    output_cls = tf.reshape(label_out, [-1, n_labels])
    # define model
    model = Model(in_image, [output_src, output_cls])
    return model
