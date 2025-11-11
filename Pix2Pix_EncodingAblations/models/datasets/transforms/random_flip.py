import tensorflow as tf

def random_flip_img_label(img, label):
    # Eine Zufallsentscheidung treffen
    flip_cond = tf.random.uniform(()) > 0.5

    # Wenn flip_cond True ist, horizontal flippen
    img_flipped = tf.cond(flip_cond,
                          lambda: tf.image.flip_left_right(img),
                          lambda: img)
    label_flipped = tf.cond(flip_cond,
                            lambda: tf.image.flip_left_right(label),
                            lambda: label)
    return img_flipped, label_flipped
