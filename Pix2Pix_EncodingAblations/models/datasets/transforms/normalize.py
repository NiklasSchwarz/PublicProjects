import tensorflow as tf

def normalize_img(img, mean, std):
    #img = (img - mean) / std
    img = img / 127.5 - 1
    return img

