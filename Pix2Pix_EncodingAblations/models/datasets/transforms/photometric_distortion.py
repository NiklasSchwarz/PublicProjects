import tensorflow as tf

def photometric_distortion(
    img,
    brightness=(0.875, 1.125),   # multiplicative factor range
    contrast=(0.5, 1.5),
    saturation=(0.5, 1.5),
    hue=(-0.05, 0.05),           # delta in [-0.5, 0.5]
    p=0.5
):

    def maybe_apply(op, prob):
        return tf.cond(tf.random.uniform([], 0, 1) < prob, op, lambda: img)
 
    # --- Brightness (multiplikativ) ---
    b_low, b_high = brightness
    b_factor = tf.random.uniform([], minval=b_low, maxval=b_high)
    def apply_brightness():
        return tf.clip_by_value(img * b_factor, 0.0, 1.0)
    img = maybe_apply(apply_brightness, p)

    # --- Random order for contrast/saturation/hue ---
    order = tf.random.shuffle(tf.constant([0, 1, 2]))  # 0=contrast, 1=saturation, 2=hue

    def apply_contrast(x):
        return tf.image.random_contrast(x, lower=contrast[0], upper=contrast[1])

    def apply_saturation(x):
        return tf.image.random_saturation(x, lower=saturation[0], upper=saturation[1])

    def apply_hue(x):
        h_delta = tf.random.uniform([], minval=hue[0], maxval=hue[1])
        return tf.image.adjust_hue(x, h_delta)

    def apply_by_idx(x, idx):
        return tf.switch_case(
            idx,
            branch_fns=[
                lambda: apply_contrast(x),
                lambda: apply_saturation(x),
                lambda: apply_hue(x),
            ],
            default=lambda: x
        )

    # apply each with prob p in random order
    for t in tf.unstack(order):
        def do_op():
            return apply_by_idx(img, t)
        img = tf.cond(tf.random.uniform([], 0, 1) < p, do_op, lambda: img)

    return tf.clip_by_value(img, 0.0, 1.0)
