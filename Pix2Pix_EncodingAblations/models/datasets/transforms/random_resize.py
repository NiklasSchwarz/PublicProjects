import tensorflow as tf

def random_resize_img_label(img, label, min_scale=0.5, max_scale=2.0):
    # Zufälligen Skalierungsfaktor wählen
    scale = tf.random.uniform([], min_scale, max_scale)

    # Neue Größe berechnen (als int)
    orig_height = tf.cast(tf.shape(img)[0], tf.float32)
    orig_width = tf.cast(tf.shape(img)[1], tf.float32)

    new_height = tf.cast(orig_height * scale, tf.int32)
    new_width = tf.cast(orig_width * scale, tf.int32)

    # Bild: bilineare Interpolation
    img_resized = tf.image.resize(img, (new_height, new_width), method="bilinear")

    # Label: nearest-neighbor (damit Klassen-IDs erhalten bleiben)
    label_resized = tf.image.resize(label, (new_height, new_width), method="nearest")

    return img_resized, label_resized