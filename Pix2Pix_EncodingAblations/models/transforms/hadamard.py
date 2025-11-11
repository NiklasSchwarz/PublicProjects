import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from scipy.linalg import hadamard 


class HadamardCoder(tf.keras.layers.Layer):
    def __init__(self, num_classes: int = 19, code_size: int = 32):
        super(HadamardCoder, self).__init__()

        self.codes_matrix = hadamard(code_size, dtype=np.float32)
        self.codes_matrix = self.codes_matrix[:num_classes, :code_size]
        self.codes_transpose = self.codes_matrix.transpose()

        self.codes_tensor = tf.convert_to_tensor(self.codes_matrix)
        self.codes_transpose = tf.convert_to_tensor(self.codes_transpose)

        self.identity_matrix = np.identity(num_classes)
        self.identity_matrix = tf.convert_to_tensor(self.identity_matrix)

        colormap = np.array(range(num_classes))
        colormap = tf.constant(colormap,  dtype=tf.float32)
        self.colormap = tf.reshape(colormap, [1, 1, 1, num_classes])

    def call(self, inputs):
        inputs_untouched = inputs

        inputs = tf.einsum('bmni, ij->bmnj', inputs, self.codes_transpose)
        
        alpha = tf.keras.layers.Softmax()(inputs)
        index = tf.keras.layers.Lambda(lambda alpha: alpha*self.colormap)(alpha)
        mask  = tf.keras.backend.sum(index, axis=-1, keepdims=True)    

        #mask = tf.argmax(alpha, axis=-1, output_type=tf.int32)
        return mask, alpha, inputs_untouched

    def encode(self, mask):
        IGNORE = 255
        codes = tf.cast(self.codes_tensor, tf.float32)        # [19, 32]

        # (H,W) oder (H,W,1)  -> int32
        mask = tf.cast(mask, tf.int32)

        # Ignore -> eigener Index (19), sonst trainId
        ignore_index = tf.shape(codes)[0]                      # 19
        indices = tf.where(tf.equal(mask, IGNORE),
                        tf.fill(tf.shape(mask), ignore_index),
                        mask)                                # int32!

        # Codebook um Nullzeile erweitern (für Ignore)
        zero_row = tf.zeros([1, tf.shape(codes)[1]], dtype=codes.dtype)
        codebook = tf.concat([codes, zero_row], axis=0)        # [20, 32]

        gathered = tf.gather(codebook, indices)                # (H,W,1,32)
        out = tf.squeeze(gathered, axis=-2)                    # (H,W,32)
        return out
