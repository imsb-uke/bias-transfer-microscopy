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


def define_unet_generator(image_shape, n_resnet):
    """
    Creates the generator with skip-connections between the encoder and decoder
    :param image_shape: Shape of the input image
    :param n_resnet: Number of resnet blocks
    :return: generator model
    """
    # Weight initialization
    init = RandomNormal(stddev=0.02)
    # Image input
    in_image = Input(shape=image_shape)
    # Encoder
    # c7s1-64
    e1 = Conv2D(64, (7, 7), padding='same', kernel_initializer=init)(in_image)
    e1 = InstanceNormalization(axis=-1)(e1)
    e1 = Activation('relu')(e1)
    # d128
    e2 = Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(e1)
    e2 = InstanceNormalization(axis=-1)(e2)
    e2 = Activation('relu')(e2)
    # d256
    e3 = Conv2D(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(e2)
    e3 = InstanceNormalization(axis=-1)(e3)
    e3 = Activation('relu')(e3)
    # R256
    # Transformer
    g = resnet_block(256, e3)
    for _ in range(n_resnet - 1):
        g = resnet_block(256, g)
    # Decoder
    # u128
    # Original 256 + 256 channels from skip-connection
    u1 = concatenate([g, e3], axis=3)
    d1 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(u1)
    d1 = InstanceNormalization(axis=-1)(d1)
    d1 = Activation('relu')(d1)
    # u64
    # Original 128 + 128 channels from skip-connection
    u2 = concatenate([d1, e2], axis=3)
    d2 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(u2)
    d2 = InstanceNormalization(axis=-1)(d2)
    d2 = Activation('relu')(d2)
    # c7s1-3
    # Original 64 + 64 channels from skip-connection
    u3 = concatenate([d2, e1], axis=3)
    # Final convolution layer to reduce channels to 3
    d3 = Conv2D(3, (7, 7), padding='same', kernel_initializer=init)(u3)
    d3 = InstanceNormalization(axis=-1)(d3)
    # Want the output to be between -1 and 1
    out_image = Activation('tanh')(d3)
    # Define model
    model = Model(in_image, out_image)
    model.summary()
    return model
