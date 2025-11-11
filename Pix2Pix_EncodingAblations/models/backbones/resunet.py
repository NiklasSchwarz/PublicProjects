import tensorflow as tf
import numpy as np

from .__base__ import unet_segment as unet
from ..transforms import HadamardCoder
from ..transforms import BaseDecoder

def GeneratorResUNet(num_classes: int = 19, code_size: int = 32, resunet: bool = True, shape: tuple = (512, 512, 3), hadamard: bool = False):
    '''
    Generator Res-U-Net
    '''
    filters_per_block = np.array([32, 64, 128, 128, 256])
    
    if resunet:
        resunet = unet.resunet(filters_per_block=filters_per_block,
                              output_channels=code_size,
                              img_size=shape,
                              droprate=0.25)
    else:
        resunet = unet.incepresunet_segment(filters_per_block=filters_per_block,
                                            num_classes=code_size,
                                            img_size=shape,
                                            droprate=0.25)

    initializer = tf.random_normal_initializer(0., 0.02)

    convT = tf.keras.layers.Conv2DTranspose(code_size, 4,
                                            strides=1,
                                            padding='same',
                                            activation='tanh',
                                            kernel_initializer=initializer)

    x_input = tf.keras.layers.Input(shape=shape)
    
    x = x_input
    x = resunet(x)
    x = convT(x)

    if hadamard:
        mask, alpha, codes = HadamardCoder(num_classes, code_size)(x)
    else:
        mask, alpha, codes = BaseDecoder()(x)
    
    return tf.keras.Model(inputs=x_input, outputs=[mask, alpha, codes])