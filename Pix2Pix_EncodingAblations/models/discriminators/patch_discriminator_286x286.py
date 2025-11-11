import tensorflow as tf

from ..backbones.__base__ import downsample

def Discriminator_286x286(input_shape: tuple = (256,256,3), output_shape: tuple = (256,256,1)):
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=input_shape, name='input_image')
    tar = tf.keras.layers.Input(shape=output_shape, name='target_image')

    x = tf.keras.layers.concatenate([inp, tar])
    
    # More downsampling steps to handle higher resolution
    down1 = downsample(64, 4, False)(x)   # 256x512
    down2 = downsample(128, 4)(down1)     # 128x256
    down3 = downsample(256, 4)(down2)     # 64x128   
    down4 = downsample(512, 4)(down3)     # 32x64
    down5 = downsample(512, 4)(down4)     # 16x32

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down5)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1, 
                                 kernel_initializer=initializer, 
                                 use_bias=False)(zero_pad1)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)
    last = tf.keras.layers.Conv2D(1, 4, strides=1, 
                                 kernel_initializer=initializer)(zero_pad2)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)
