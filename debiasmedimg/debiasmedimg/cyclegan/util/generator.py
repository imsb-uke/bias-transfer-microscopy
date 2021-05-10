import tensorflow as tf
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Activation, Add, concatenate
from tensorflow.keras.initializers import RandomNormal


def resnet_block(n_filters, input_layer):
    """
    Generates a resNet block
    :param n_filters: Number of filters
    :param input_layer: Which layer this block follows
    :return: resNet block
    """
    # Weight initialization
    init = RandomNormal(stddev=0.02)
    # First convolutional layer
    g = Conv2D(n_filters, (3, 3), strides=(1, 1), padding='same', kernel_initializer=init)(input_layer)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # Second convolutional layer
    g = Conv2D(n_filters, (3, 3), strides=(1, 1), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    # Add up the original input and the convolved output
    g = Add()([g, input_layer])
    return g


def define_generator(image_shape, n_resnet):
    """
    Creates the generator
    :param image_shape: shape of the input image
    :param n_resnet: Number of resnet blocks
    :return: generator model
    """

    # Weight initialization
    init = RandomNormal(stddev=0.02)
    # Image input
    in_image = Input(shape=image_shape)
    # c7s1-64
    # Encoder
    g = Conv2D(64, (7, 7), padding='same', kernel_initializer=init)(in_image)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # d128
    g = Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # d256
    g = Conv2D(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # R256
    # Transformer
    for _ in range(n_resnet):
        g = resnet_block(256, g)
    # u128
    # Decoder
    g = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # u64
    g = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # c7s1-3
    g = Conv2D(3, (7, 7), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    out_image = Activation('tanh')(g)
    # Define model
    model = Model(in_image, out_image)
    return model
