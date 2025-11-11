import tensorflow as tf

from ..backbones.__base__ import downsample

def Discriminator_70x70(input_shape: tuple = (256,256,3), output_shape: tuple = (256,256,1)):
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=input_shape, name='input_image')
    tar = tf.keras.layers.Input(shape=output_shape, name='target_image')

    x = tf.keras.layers.concatenate([inp, tar])
    
    # More downsampling steps to handle higher resolution
    down1 = downsample(64, 4, apply_batchnorm=False)(x)   # 256x512
    down2 = downsample(128, 4)(down1)     # 128x256
    down3 = downsample(256, 4)(down2)     # 64x128   
    down4 = downsample(512, 4, 1)(down3)     # 32x64

    last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(down4)
    return tf.keras.Model(inputs=[inp, tar], outputs=last)
