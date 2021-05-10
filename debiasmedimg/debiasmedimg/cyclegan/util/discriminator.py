from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, LeakyReLU
from tensorflow.keras.initializers import RandomNormal
from tensorflow_addons.layers import InstanceNormalization


def define_discriminator(image_shape):
    """
    Create the discriminator model and compile it
    :param image_shape: Shape of the input image
    :return: Discriminator Keras model
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
    # Second last output layer
    d = Conv2D(512, (4, 4), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # Patch output
    patch_out = Conv2D(1, (4, 4), padding='same', kernel_initializer=init)(d)
    # Define model
    model = Model(in_image, patch_out)
    return model
