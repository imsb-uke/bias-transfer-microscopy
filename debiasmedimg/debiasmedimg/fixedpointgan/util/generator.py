import numpy as np
import tensorflow as tf
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Activation, Concatenate, UpSampling2D, Add, Lambda, \
    RepeatVector, Reshape, concatenate
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.backend import expand_dims


def resnet_block(n_filters, input_layer, normalize_fn=True):
    """
    Generates a resNet block
    :param n_filters: Number of filters
    :param input_layer: Which layer this block follows
    :param normalize_fn: Whether to normalize inside the resnet blocks
    :return: resNet block
    """
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # first layer convolutional layer
    g = pad(input_layer, 1)
    g = Conv2D(n_filters, kernel_size=3, strides=1, padding='valid', kernel_initializer=init)(g)
    if normalize_fn:
        g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # second convolutional layer
    g = pad(g, 1)
    g = Conv2D(n_filters, kernel_size=3, strides=1, padding='valid', kernel_initializer=init)(g)
    if normalize_fn:
        g = InstanceNormalization(axis=-1)(g)
    g += input_layer
    return g


def _padding_arg(h, w, input_format):
    """Calculate the padding shape for tf.pad()
    :param h: padding of height dim
    :param w: padding of width dim
    :param input_format 'NHWC' or 'HWC'
    :return: padded image
    """
    if input_format == 'NHWC':
        return [[0, 0], [h, h], [w, w], [0, 0]]
    elif input_format == 'HWC':
        return [[h, h], [w, w], [0, 0]]
    else:
        raise ValueError('Input Format %s is not supported.' % input_format)


def pad(input_layer, padding_size):
    """
    Pad the input tensor with padding_size on height and width dimension
    :param input_layer: Tensor of dimension 3 or 4 ((batch_size), height, width, channels)
    :param padding_size: Size of padding
    :return: Padded tensor
    """
    if len(input_layer.shape) == 4:
        return tf.pad(
            tensor=input_layer,
            paddings=_padding_arg(padding_size, padding_size, 'NHWC'))
    elif len(input_layer.shape) == 3:
        return tf.pad(
            tensor=input_layer,
            paddings=_padding_arg(padding_size, padding_size, 'HWC'))
    else:
        raise ValueError('The input tensor need to be either 3D or 4D.')


def define_generator(image_shape, label_shape, target_units, out_channels, n_resnet):
    """
    Creates the generator
    :param image_shape: Shape of the input image (height,width,channels)
    :param label_shape: Shape of the condition (domain to translate to)
    :param n_resnet: Number of resnet blocks
    :param target_units: Number of hidden units of the final layers
    :param out_channels: How many channels the output should be
    :return: generator model
    """
    if target_units % 4 != 0:
        raise ValueError("Target shape must be divisible by 4")
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=image_shape)
    in_label = Input(shape=label_shape)
    # Prepare label for concatenation
    in_label_con = RepeatVector(image_shape[0] * image_shape[1])(in_label)
    in_label_con = Reshape((image_shape[0], image_shape[1], label_shape[-1]))(in_label_con)
    # c7s1-64
    # Encoder
    in_c_image = Concatenate(axis=-1)([in_image, in_label_con])
    g = pad(in_c_image, 3)
    g = Conv2D(filters=target_units // 4, kernel_size=7, strides=1, padding='valid', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # d128
    g = pad(g, 1)
    g = Conv2D(filters=target_units // 2, kernel_size=4, strides=2, padding='valid',
               kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # d256
    g = pad(g, 1)
    g = Conv2D(filters=target_units, kernel_size=4, strides=2, padding='valid', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # R256
    # Transformer
    for _ in range(n_resnet):
        g = resnet_block(target_units, g)
    # u128
    # Decoder
    g = Conv2DTranspose(filters=128, kernel_size=4, strides=2, padding='valid',
                        kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    g = g[:, 1:-1, 1:-1, :]
    # u64
    g = Conv2DTranspose(filters=64, kernel_size=4, strides=2, padding='valid', kernel_initializer=init)(
        g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    g = g[:, 1:-1, 1:-1, :]
    # c7s1-3
    g = pad(g, 3)
    g = Conv2D(filters=out_channels, kernel_size=7, strides=1, padding='valid',
               kernel_initializer=init)(g)
    out_image = Activation('tanh')(g)
    # define model
    model = Model([in_image, in_label], out_image)
    model.summary()
    return model
