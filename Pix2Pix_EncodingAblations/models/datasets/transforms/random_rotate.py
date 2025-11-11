import tensorflow as tf
import tensorflow_addons as tfa

def random_rotate_img_label(img, lbl, max_deg=30.0):
    deg = tf.random.uniform([], -max_deg, max_deg)
    rad = deg * (3.14159265 / 180.0)
    img = tfa.image.rotate(img, rad, interpolation='BILINEAR', fill_mode='nearest')
    lbl = tfa.image.rotate(lbl, rad, interpolation='NEAREST', fill_mode='nearest')
    return img, lbl
