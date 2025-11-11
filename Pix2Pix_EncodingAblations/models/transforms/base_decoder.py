import numpy as np
import tensorflow as tf

class BaseDecoder(tf.keras.layers.Layer):
    def __init__(self, num_classes: int = 19):
        super(BaseDecoder, self).__init__()
        colormap = np.array(range(num_classes)) #/(NUM_CLASSES-1))-0.5)*2
        colormap = tf.constant(colormap,  dtype=tf.float32)
        self.colormap = tf.reshape(colormap, [1, 1, 1, num_classes])

    def call(self, inputs):
        alpha = tf.keras.layers.Softmax()(inputs)
        index = tf.keras.layers.Lambda(lambda alpha: alpha*self.colormap)(alpha)
        mask  = tf.keras.backend.sum(index, axis=-1, keepdims=True)    
        return mask, alpha, inputs

