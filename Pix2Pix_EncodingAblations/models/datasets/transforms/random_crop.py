import tensorflow as tf


def ensure_min_size(img, mask, crop_size):
    IGNORE_LABEL = tf.constant(255, tf.int32)
    h = tf.shape(img)[0]; w = tf.shape(img)[1]
    pad_h = tf.maximum(0, crop_size[0] - h)
    pad_w = tf.maximum(0, crop_size[1] - w)

    paddings = [[0, pad_h], [0, pad_w], [0, 0]]  # unten/rechts

    # Bild normal mit 0 padden
    img  = tf.pad(img, paddings, constant_values=0)

    # Maske mit IGNORE_LABEL padden
    mask = tf.cast(mask, tf.int32)
    mask  = tf.pad(mask, paddings, constant_values=int(IGNORE_LABEL))
    mask  = tf.cast(mask, tf.uint8)  # zurück, falls du uint8 nutzt

    return img, mask

def random_crop_img_label(img, label, crop_size):

    img, label = ensure_min_size(img, label, crop_size)
    # Stapel mit Bild und Label an der letzten Dimension
    combined = tf.concat([img, tf.cast(label, img.dtype)], axis=-1)
    
    # Gemeinsamen Crop durchführen
    combined_cropped = tf.image.random_crop(
        combined, size=[crop_size[0], crop_size[1], tf.shape(combined)[-1]]
    )
    
    # Bild und Label wieder trennen
    img_cropped = combined_cropped[..., :tf.shape(img)[-1]]
    label_cropped = tf.cast(combined_cropped[..., tf.shape(img)[-1]:], label.dtype)
    
    return img_cropped, label_cropped

