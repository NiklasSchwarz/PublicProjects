import tensorflow as tf
import numpy as np

from .__base__ import unet_segment as unet
from ..transforms import HadamardCoder
from ..transforms import BaseDecoder

def GeneratorUNet3Plus(num_classes: int = 19, code_size: int = 32, shape: tuple = (512, 512, 3), hadamard: bool = False):
    '''
    Generator U-Net 3 Plus
    '''
    filters_per_block = np.array([64, 128, 256, 512, 1024])
    
    unet_3plus = unet.unet_3plus_segment(filters_per_block = filters_per_block,  
                                         num_classes       = code_size,
                                         img_size          = shape) 
                    

    initializer = tf.random_normal_initializer(0., 0.02)

    convT = tf.keras.layers.Conv2DTranspose(code_size, 4,
                                            strides = 1,
                                            padding = 'same',
                                            activation='tanh',
                                            kernel_initializer = initializer)

    
    x_input = tf.keras.layers.Input(shape=shape)
    
    x = x_input
    x = unet_3plus(x)
    x = convT(x)

    if hadamard:
        mask, alpha, codes = HadamardCoder(num_classes, code_size)(x)
    else:
        mask, alpha, codes = BaseDecoder()(x)

    return tf.keras.Model(inputs=x_input, outputs=[mask, alpha, codes])