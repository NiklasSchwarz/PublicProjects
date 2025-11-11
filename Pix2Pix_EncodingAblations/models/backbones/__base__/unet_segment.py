#Version 1.0 M. Rivera: multiples modelos de segmentación 
#Version 2.0 A. Hoyos: implementación del modelo UNet 3+ 

import numpy as np

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D, Dropout, UpSampling2D, Activation
from tensorflow.keras.layers import Input, Dense, Activation, Conv2D,  SeparableConv2D, LeakyReLU


def unet_segment(filters_per_block,  num_classes, img_size, droprate=0.25):
    '''
    parameters
    filters_per_block   :   list of channels after each block-level.
                            filters_per_block = np.array([num_channels, 256, 128, 64, 32])
                            where num_channels is the number of channels in the input image.
    num_classes         :   int, number of channels in the output
    img_size            :   (rows, cols, channels) ints,  input dimension
    droprate            :   float, dropout rate
    
    return
    A unet model for segmenting images into num_classes models
    
    --- Example ---          
    img_size = (160,160,3)
    filters_per_block = np.array([img_size[2], 256, 128, 64, 32])
    model_unet = Unet_segment(filters_per_block = filters_per_block,  
                              num_classes       = 3,
                              img_size          = img_size,
                              droprate          = 0.25)  
    AJHI & MJJRM - 08/2024
    '''
    
    num_blocks  = len(filters_per_block)   
    drop        = droprate*tf.ones(num_blocks, tf.float32)
    kernel_size = (3,3)
    
    #- - - - - - - - - 
    # Encoder
    #- - - - - - - - - 
    Xdicc = {}
    Xin   = tf.keras.layers.Input(shape=img_size, name="x_true")
    
    
    # head
    Xdicc[0] = Xin
    X = Conv2D(10, kernel_size=kernel_size, padding='same', activation='relu', name='encoder-conv0_0')(Xin) 
        
    numFilters=filters_per_block[0]
    for i in range(1,num_blocks):
        numFilters=filters_per_block[i]
        
        X = Conv2D(numFilters, kernel_size=kernel_size, padding='same', activation='relu', name='encoder-conv1_{}'.format(str(i)))(X) 
        X = Conv2D(numFilters, kernel_size=kernel_size, padding='same', activation='relu', name='encoder-conv2_{}'.format(str(i)))(X)
        X = Dropout(rate=drop[i], name='encoder-drop_{}'.format(str(i)))(X)
        X = MaxPooling2D(pool_size=(2,2), padding='valid', name='encoder-maxpool_{}'.format(str(i)))(X)
        
        Xdicc[i] = X

    #- - - - - - - - - 
    # Decoder
    #- - - - - - - - - 
    Y=X
    for i in range(num_blocks-1,0,-1):
        if i>1:
            numFilters = filters_per_block[i] 
        else:
            numFilters = 128
        
        Y = UpSampling2D(size=2, name='decoder-up_{}'.format(str(i)))(Y)  
        Y = Concatenate(name='decoder-concat_{}'.format(str(i)))([Y, Xdicc[i-1]])
        Y = Conv2D(numFilters, kernel_size=(3,3), padding='same', activation='relu', name='decoder-conv2_{}'.format(str(i)))(Y)
        Y = Conv2D(numFilters, kernel_size=(3,3), padding='same', activation='relu', name='decoder-conv3_{}'.format(str(i)))(Y)
        Y = Dropout(rate=drop[i], name='decoder-drop_{}'.format(str(i)))(Y)

    # Tail 
    Y = Conv2D(32, kernel_size=(3,3), padding='same', activation=None, name='tail-2xch')(Y)

    Yout = Conv2D(num_classes, kernel_size=(1,1), padding='same', activation='softmax', name='tail-last')(Y)

    return Model(inputs=Xin, outputs=[Yout], name='UNet_segment')


def resunet_down_segment(filters_per_block,  num_classes, img_size, droprate=0.25):
    '''
    parameters
    filters_per_block   :   list of channels after each block-level.
                            filters_per_block = np.array([num_channels, 256, 128, 64, 32])
                            where num_channels is the number of channels in the input image.
                            
    num_classes         :   int, number of channels in the output
    
    img_size            :   (rows, cols, channels) ints,  input dimension
    droprate            :   float, dropout rate
    
    return
    A unet model for segmenting images into num_classes models
    
    
    --- Example ---          
    
    img_size = (160,160,3)
    filters_per_block = np.array([img_size[2], 256, 128, 64, 32])
    
    model_resunet_down = resunet_down_segment(filters_per_block = filters_per_block,  
                                              num_classes       = 3,
                                              img_size          = img_size,
                                              droprate          = 0.25)  
    
    AJHI & MJJRM - 08/2024
    '''
    
    num_blocks  = len(filters_per_block)   
    drop        = droprate*tf.ones(num_blocks, 'float32')
    kernel_size = (3,3)
    
    #- - - - - - - - - 
    # Encoder
    #- - - - - - - - - 
    Xdicc = {}
    Xin   = Input(shape=img_size, name="x_true")
    
    # head
    Xdicc[0] = Xin
    X_skip   = Conv2D(10, kernel_size=kernel_size, padding='same', activation='relu', name='encoder-conv0_0')(Xin) 
        
    numFilters=filters_per_block[0]
    for i in range(1,num_blocks):
        Xdicc[i] = X_skip
        numFilters = filters_per_block[i]
        
        X = SeparableConv2D(numFilters, kernel_size=kernel_size, padding='same', name='encoder-conv1_{}'.format(str(i)))(X_skip) 
        X = BatchNormalization()(X)
        X = Activation("relu")(X) 
        
        X = SeparableConv2D(numFilters, kernel_size=kernel_size, padding='same', name='encoder-conv2_{}'.format(str(i)))(X) 
        X = BatchNormalization()(X)
        X = Activation("relu")(X)          
        
        X = Dropout(rate=drop[i], name='encoder-drop_{}'.format(str(i)))(X)
        
        # residual aggregation 
        X = Concatenate(name='encoder-concat-residual-{}'.format(str(i)))([X,X_skip])
        X = Conv2D(numFilters, kernel_size=(1,1), padding='same', name='encoder-conv3_{}'.format(str(i)))(X) 
        X = BatchNormalization()(X)
        X = Activation("relu")(X)
        
        X_skip = MaxPooling2D(pool_size=(2,2), padding='valid', name='encoder-maxpool_{}'.format(str(i)))(X)

    #- - - - - - - - - 
    # Decoder
    #- - - - - - - - - 
    Y = X_skip
    for i in range(num_blocks-1,0,-1):
        if i>1:
            numFilters = filters_per_block[i-1] 
        else:
            numFilters = 64
            
        Y = UpSampling2D(size=2, name='decoder-up_{}'.format(str(i)))(Y)  
        Y = Concatenate(name='decoder-concat_{}'.format(str(i)))([Y, Xdicc[i]])
        
        Y = SeparableConv2D(numFilters, kernel_size=(3,3), padding='same', name='decoder-conv2_{}'.format(str(i)))(Y)
        Y = BatchNormalization()(Y)
        Y = Activation("relu")(Y) 
    
        Y = SeparableConv2D(numFilters, kernel_size=(3,3), padding='same', name='decoder-conv3_{}'.format(str(i)))(Y)
        Y = BatchNormalization()(Y)
        Y = Activation("relu")(Y) 
    
        Y = Dropout(rate=drop[i], name='decoder-drop_{}'.format(str(i)))(Y)

    # Tail 
    Y = SeparableConv2D(32, 
                        kernel_size = (3,3), 
                        padding     = 'same', 
                        activation  = None,
                        name        = 'tail-2xch')(Y)
    
    Yout = SeparableConv2D(num_classes, 
                           kernel_size = (1,1), 
                           padding     = 'same', 
                           activation  = 'softmax', 
                           name        = 'tail-last')(Y)
            
    return Model(inputs=Xin, outputs=[Yout], name='ResUNet_down_segment')

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -         

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -         

def BlockConv(filters, kernel_size, drop, lastname, padding='same', dilation=(1,1)):
    blockdown = Sequential(name='conv_block_'+lastname)

    blockdown.add(SeparableConv2D(filters       = filters, 
                                  kernel_size   = kernel_size, 
                                  padding       = padding,
                                  dilation_rate = dilation[0]))
                                   
    blockdown.add(BatchNormalization())
    blockdown.add(LeakyReLU())

    blockdown.add(SeparableConv2D(filters       = filters, 
                                  kernel_size   = kernel_size, 
                                  padding       = padding,
                                  dilation_rate = dilation[1]))
    
    blockdown.add(BatchNormalization())
    
    return blockdown

def BlockBlendDown(filters, kernel_size, drop, lastname):
    block_blenddown = Sequential(name='blenddown_'+lastname)
    
    block_blenddown.add(SeparableConv2D(filters, 
                                    kernel_size = (3,3), 
                                    strides     = 2,         
                                    padding     = 'same'))  
    
    block_blenddown.add(BatchNormalization())
    block_blenddown.add(LeakyReLU())
    block_blenddown.add(Dropout(drop))
    return block_blenddown
        
def BlockBlend(filters, kernel_size, drop, lastname):
    block_blend = Sequential(name='blend_'+lastname)
    
    block_blend.add(SeparableConv2D(filters, 
                                    kernel_size = (3,3), 
                                    padding     = 'same'))
    
    block_blend.add(BatchNormalization())
    block_blend.add(LeakyReLU())
    block_blend.add(Dropout(drop))
    return block_blend
        
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -         
def resunet_segment(filters_per_block, num_classes, img_size, droprate=0.25, logits=False):
    '''
    parameters
    filters_per_block   :   list of channels after each block-level.
                            filters_per_block = np.array([num_channels, 256, 128, 64, 32])
                            where num_channels is the number of channels in the input image.
                            
    num_classes         :   int, number of channels in the output
    
    img_size            :   (rows, cols, channels) ints,  input dimension
    droprate            :   float, dropout rate
    
    return
    A unet model for segmenting images into num_classes models
    
    
    --- Example ---          
    
    img_size = (160,160,3)
    filters_per_block = np.array([img_size[2], 256, 128, 64, 32])
    
    model_resunet_down = resunet_segment(filters_per_block = filters_per_block,  
                              num_classes       = 3,
                              img_size          = img_size,
                              droprate          = 0.25,
                              )  
    
    AJHI & MJJRM - 08/2024
    '''
    
    num_blocks  = len(filters_per_block)   
    drop        = droprate * np.ones(num_blocks)
    kernel_size = (3,3)
    
    #- - - - - - - - - 
    # Encoder
    #- - - - - - - - - 
    Xdicc = {}
    Xin   = Input(shape=img_size, name="x_true")
    
    # head
    Xdicc[0] = Xin
    X = Conv2D(32, kernel_size=kernel_size, padding='same', activation='relu', name='encoder-conv0_0')(Xin) 
    
    for i in range(1,num_blocks):
        Xdicc[i]=X
        filters=filters_per_block[i]
        
        FX = BlockConv      (filters=filters, kernel_size=kernel_size, drop=drop[i], lastname='enc'+str(i))(X)
        X  = Concatenate    (name='encoder_concat_residual_{}'.format(str(i)))([X, FX])
        X  = BlockBlendDown (filters=filters, kernel_size=kernel_size,drop=drop[i], lastname= 'enc'+str(i))(X)
    
    #- - - - - - - - - 
    # Decoder
    #- - - - - - - - - 
    Y = X
    for i in range(num_blocks-1,0,-1):
        if i>1:
            filters = filters_per_block[i-1] 
        else:
            filters = 64
            
        Y  = UpSampling2D(size=2, name='decoder-up_{}'.format(str(i)))(Y)  
        Y  = Concatenate (name='decoder-concat_{}'.format(str(i)))([Y, Xdicc[i]])
        FY = BlockConv   (filters=filters, kernel_size=kernel_size, drop=drop[i],lastname='dec'+str(i))(Y)
        Y  = Concatenate (name='decoder-concat-residual-{}'.format(str(i)))([Y,FY])
        Y  = BlockBlend  (filters=filters, kernel_size=kernel_size, drop=drop[i],lastname='dec'+str(i))(Y)
        
    # Tail 
    Y = SeparableConv2D(32, 
                        kernel_size = (3,3), 
                        padding     = 'same', 
                        activation  = None,
                        name        = 'tail-2xch')(Y)
    
    Y = Dropout(drop[0])(Y)
    
    Yout = SeparableConv2D(num_classes, 
                           kernel_size = (1,1), 
                           padding     = 'same', 
                           activation  = None, 
                           name        = 'tail-last')(Y)
    
    if not logits:
        Yout = Activation("softmax")(Yout)
                    
    return Model(inputs=Xin, outputs=[Yout], name='ResUNet_segment')

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -         

def resunet(filters_per_block, output_channels, img_size, droprate=0.25):
    '''
    parameters
    filters_per_block   :   list of channels after each block-level.
                            filters_per_block = np.array([num_channels, 256, 128, 64, 32])
                            where num_channels is the number of channels in the input image.
                            
    output_channels     :   int, number of channels in the output
    
    img_size            :   (rows, cols, channels) ints,  input dimension
    droprate            :   float, dropout rate
    
    return
    A unet model for segmenting images into num_classes models
    
    
    --- Example ---          
    
    img_size = (160,160,3)
    filters_per_block = np.array([img_size[2], 256, 128, 64, 32])
    
    resunet_down = resunet(filters_per_block = filters_per_block,  
                           num_classes       = 3,
                           img_size          = img_size,
                           droprate          = 0.25,
                           )  
    
    AJHI & MJJRM - 08/2024
    '''
    
    num_blocks  = len(filters_per_block)   
    drop        = droprate*np.ones(num_blocks) #tf.ones(num_blocks, 'float32')
    kernel_size = (3,3)
    
    #- - - - - - - - - 
    # Encoder
    #- - - - - - - - - 
    Xdicc ={}
    Xin   = Input(shape=img_size, name="x_true")
    
    # head
    Xdicc[0] = Xin
    X = Conv2D(32, kernel_size=kernel_size, padding='same', activation='relu', name='encoder-conv0_0')(Xin) 
    
    for i in range(1,num_blocks):
        Xdicc[i]=X
        filters=filters_per_block[i]
        
        FX = BlockConv     (filters=filters, kernel_size=kernel_size, drop=drop[i], lastname='enc'+str(i))(X)
        X  = Concatenate   (name='encoder_concat_residual_{}'.format(str(i)))([X, FX])
        X  = BlockBlendDown(filters=filters, kernel_size=kernel_size,drop=drop[i], lastname= 'enc'+str(i))(X)
   
    #- - - - - - - - - 
    # Decoder
    #- - - - - - - - - 
    Y = X
    for i in range(num_blocks-1,0,-1):
        if i>1:
            filters = filters_per_block[i-1] 
        else:
            filters = 64
            
        Y  = UpSampling2D(size=2, name='decoder-up_{}'.format(str(i)))(Y)  
        Y  = Concatenate (name='decoder-concat_{}'.format(str(i)))([Y, Xdicc[i]])
        FY = BlockConv   (filters=filters, kernel_size=kernel_size, drop=drop[i],lastname='dec'+str(i))(Y)
        Y  = Concatenate (name='decoder-concat-residual-{}'.format(str(i)))([Y,FY])
        Y  = BlockBlend  (filters=filters, kernel_size=kernel_size, drop=drop[i],lastname='dec'+str(i))(Y)
        
    # Tail 
    Y = SeparableConv2D(output_channels, 
                        kernel_size = (3,3), 
                        padding     = 'same', 
                        activation  = None,
                        name        = 'tail-2xch')(Y)
                       
    Yout = Dropout(drop[0])(Y)

    return Model(inputs=Xin, outputs=[Yout], name='ResUnet')

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -         

def incepresunet_segment(filters_per_block,  num_classes, img_size, droprate=0.25):
    '''
    parameters
    filters_per_block   :   list of channels after each block-level.
                            filters_per_block = np.array([num_channels, 256, 128, 64, 32])
                            where num_channels is the number of channels in the input image.
                            
    num_classes         :   int, number of channels in the output
    
    img_size            :   (rows, cols, channels) ints,  input dimension
    droprate            :   float, dropout rate
    
    return
    A unet model for segmenting images into num_classes models
    
    
    --- Example ---          
    
    img_size = (160,160,3)
    filters_per_block = np.array([img_size[2], 256, 128, 64, 32])
    
    model_resunet_down = incepresunet_segment(filters_per_block = filters_per_block,  
                                              num_classes       = 3,
                                              img_size          = img_size,
                                              droprate          = 0.25,
                                              )  
    
    AJHI & MJJRM - 08/2024
    '''
    
    num_blocks  = len(filters_per_block)   
    drop        = droprate*np.ones(num_blocks) #tf.ones(num_blocks, 'float32')
    kernel_size = (3,3)
    
    #- - - - - - - - - 
    # Encoder
    #- - - - - - - - - 
    Xdicc = {}
    Xin   = Input(shape=img_size, name="x_true")
    
    # head
    Xdicc[0] = Xin
    X = Conv2D(32, kernel_size=kernel_size, padding='same', activation='relu', name='encoder-conv0_0')(Xin) 
    
    for i in range(1,num_blocks):
        Xdicc[i] = X
        filters  = filters_per_block[i]
        
        FX1 = BlockConv     (filters=filters//2, kernel_size=kernel_size, drop=drop[i], lastname='enc1_'+str(i), dilation=(1,1))(X)
        FX2 = BlockConv     (filters=filters//2, kernel_size=kernel_size, drop=drop[i], lastname='enc2_'+str(i), dilation=(1,2))(X)
        X   = Concatenate   (name='encoder_concat_residual_{}'.format(str(i)))([X, FX1, FX2])
        X   = BlockBlendDown(filters=filters, kernel_size=kernel_size,drop=drop[i], lastname= 'enc_'+str(i))(X)
    
    #- - - - - - - - - 
    # Decoder
    #- - - - - - - - - 
    Y = X
    for i in range(num_blocks-1,0,-1):
        if i>1:
            filters = filters_per_block[i-1] 
        else:
            filters = 64
            
        Y  = UpSampling2D(size=2, name='decoder-up_{}'.format(str(i)))(Y)  
        Y  = Concatenate (name='decoder-concat_{}'.format(str(i)))([Y, Xdicc[i]])
        FY = BlockConv   (filters=filters, kernel_size=kernel_size, drop=drop[i],lastname='dec_'+str(i))(Y)
        Y  = Concatenate (name='decoder-concat-residual-{}'.format(str(i)))([Y,FY])
        Y  = BlockBlend  (filters=filters, kernel_size=kernel_size, drop=drop[i],lastname='dec_'+str(i))(Y)
        
    # Tail 
    Y = SeparableConv2D(32, 
                        kernel_size = (3,3), 
                        padding     = 'same', 
                        activation  = None,
                        name        = 'tail-2xch')(Y)
   
    Y = Dropout(drop[0])(Y)

    Yout = SeparableConv2D(num_classes, 
                           kernel_size = (1,1), 
                           padding     = 'same', 
                           activation  = None,  
                           name        = 'tail-last')(Y)
                               
    return Model(inputs=Xin, outputs=[Yout], name='IncepResUnet_segment')


def conv_block(x, kernels, kernel_size=(3, 3), strides=(1, 1), padding='same', is_bn=True, is_relu=True, is_tanh=False, n=2):
    for i in range(1, n + 1):
        x = keras.layers.Conv2D(filters     = kernels, 
                                kernel_size = kernel_size, 
                                padding     = padding, 
                                strides     = strides,
                                kernel_regularizer = tf.keras.regularizers.l2(1e-4),
                                kernel_initializer = keras.initializers.he_normal(seed=5))(x)
        
        if is_bn:
            x = keras.layers.BatchNormalization()(x)
        
        if is_relu:
            x = keras.activations.relu(x)
            
        if is_tanh:
            x = keras.activations.tanh(x)

    return x


def unet_3plus_segment(filters_per_block,  num_classes, img_size):
    '''
    source: https://github.com/hamidriasat/UNet-3-Plus
    
    parameters
    filters_per_block   :   list of channels after each block-level.
                            filters_per_block = [64, 128, 256, 512, 1024]
                            where num_channels is the number of channels in the input image.
                            
    num_classes         :   int, number of channels in the output
    
    img_size            :   (rows, cols, channels) ints,  input dimension
    
    return
    A unet model for segmenting images into num_classes models
    
    
    --- Example ---          
    
    img_size = (256, 256, 3)
    filters_per_block = [64, 128, 256, 512, 1024]
    
    model_unet_3plus = unet_3plus_segment(filters_per_block = filters_per_block,  
                                          num_classes       = 19,
                                          img_size          = img_size)
    
    AJHI & MJJRM - 08/2024
    '''
    
    input_layer = Input(shape=img_size, name="input_layer")
    
    #- - - - - - - - - 
    # Encoder
    #- - - - - - - - - 
    # block 1
    e1 = conv_block(input_layer, filters_per_block[0])        

    # block 2
    e2 = keras.layers.MaxPool2D(pool_size=(2, 2))(e1)  
    e2 = conv_block(e2, filters_per_block[1])                 

    # block 3
    e3 = keras.layers.MaxPool2D(pool_size=(2, 2))(e2)  
    e3 = conv_block(e3, filters_per_block[2])                

    # block 4
    e4 = keras.layers.MaxPool2D(pool_size=(2, 2))(e3)  
    e4 = conv_block(e4, filters_per_block[3])                

    # block 5
    # bottleneck layer
    e5 = keras.layers.MaxPool2D(pool_size=(2, 2))(e4)  
    e5 = conv_block(e5, filters_per_block[4])                

    #- - - - - - - - - 
    # Decoder
    #- - - - - - - - - 
    
    cat_channels = filters_per_block[0]
    cat_blocks = len(filters_per_block)
    upsample_channels = cat_blocks * cat_channels

    """ d4 """
    e1_d4 = keras.layers.MaxPool2D(pool_size=(8, 8))(e1)                        
    e1_d4 = conv_block(e1_d4, cat_channels, n=1)                       

    e2_d4 = keras.layers.MaxPool2D(pool_size=(4, 4))(e2)                        
    e2_d4 = conv_block(e2_d4, cat_channels, n=1)                       

    e3_d4 = keras.layers.MaxPool2D(pool_size=(2, 2))(e3)                        
    e3_d4 = conv_block(e3_d4, cat_channels, n=1)                       

    e4_d4 = conv_block(e4, cat_channels, n=1)                          

    e5_d4 = keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(e5)  
    e5_d4 = conv_block(e5_d4, cat_channels, n=1)                        

    d4 = keras.layers.concatenate([e1_d4, e2_d4, e3_d4, e4_d4, e5_d4])
    d4 = conv_block(d4, upsample_channels, n=1)                         

    """ d3 """
    e1_d3 = keras.layers.MaxPool2D(pool_size=(4, 4))(e1)    
    e1_d3 = conv_block(e1_d3, cat_channels, n=1)        

    e2_d3 = keras.layers.MaxPool2D(pool_size=(2, 2))(e2)    
    e2_d3 = conv_block(e2_d3, cat_channels, n=1)        

    e3_d3 = conv_block(e3, cat_channels, n=1)           

    d4_d3 = keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(d4)      
    d4_d3 = conv_block(d4_d3, cat_channels, n=1)        

    e5_d3 = keras.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(e5)      
    e5_d3 = conv_block(e5_d3, cat_channels, n=1)        

    d3 = keras.layers.concatenate([e1_d3, e2_d3, e3_d3, d4_d3, e5_d3])
    d3 = conv_block(d3, upsample_channels, n=1)         

    """ d2 """
    e1_d2 = keras.layers.MaxPool2D(pool_size=(2, 2))(e1)    
    e1_d2 = conv_block(e1_d2, cat_channels, n=1)        

    e2_d2 = conv_block(e2, cat_channels, n=1)           

    d3_d2 = keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(d3)      
    d3_d2 = conv_block(d3_d2, cat_channels, n=1)        

    d4_d2 = keras.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(d4)      
    d4_d2 = conv_block(d4_d2, cat_channels, n=1)        

    e5_d2 = keras.layers.UpSampling2D(size=(8, 8), interpolation='bilinear')(e5)      
    e5_d2 = conv_block(e5_d2, cat_channels, n=1)        

    d2 = keras.layers.concatenate([e1_d2, e2_d2, d3_d2, d4_d2, e5_d2])
    d2 = conv_block(d2, upsample_channels, n=1)         

    """ d1 """
    e1_d1 = conv_block(e1, cat_channels, n=1)      

    d2_d1 = keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(d2)      
    d2_d1 = conv_block(d2_d1, cat_channels, n=1)        

    d3_d1 = keras.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(d3)      
    d3_d1 = conv_block(d3_d1, cat_channels, n=1)        

    d4_d1 = keras.layers.UpSampling2D(size=(8, 8), interpolation='bilinear')(d4)      
    d4_d1 = conv_block(d4_d1, cat_channels, n=1)        

    e5_d1 = keras.layers.UpSampling2D(size=(16, 16), interpolation='bilinear')(e5)    
    e5_d1 = conv_block(e5_d1, cat_channels, n=1)        

    d1 = keras.layers.concatenate([e1_d1, d2_d1, d3_d1, d4_d1, e5_d1, ])
    d1 = conv_block(d1, upsample_channels, n=1)
    
    # last layer does not have batchnorm and relu
    d = conv_block(d1, num_classes, n=1, is_bn=False, is_relu=False)
    
    return Model(inputs=input_layer, outputs=[d], name='UNet_3Plus_segment')