import tensorflow as tf
import numpy as np

from .__base__ import downsample, upsample
from ..transforms import HadamardCoder
from ..transforms import BaseDecoder


def GeneratorUNet(num_classes: int = 19, code_size: int = 32, shape: tuple = (512, 512, 3), hadamard: bool = False):
    '''
    Generator U-Net
    '''
    x_inputs = tf.keras.layers.Input(shape=shape)

    down_stack = [downsample(64,  4, apply_batchnorm=False), 
                  downsample(128, 4),  
                  downsample(256, 4),  
                  downsample(512, 4),  
                  downsample(512, 4),  
                  downsample(512, 4),  
                  downsample(512, 4)]  

    up_stack   = [upsample(512, 4, apply_dropout=True),  
                  upsample(512, 4, apply_dropout=True),  
                  upsample(512, 4, apply_dropout=True), 
                  upsample(512, 4),  
                  upsample(256, 4),  
                  upsample(128, 4)]  


    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(code_size, 4,
                                           strides=2,
                                           padding='same',
                                           activation='tanh',
                                           kernel_initializer=initializer)

    x = x_inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    if hadamard:
        mask, alpha, codes = HadamardCoder(num_classes, code_size)(x)
    else:
        mask, alpha, codes = BaseDecoder()(x)

    return tf.keras.Model(inputs=x_inputs, outputs=[mask, alpha, codes])