import os
import tensorflow as tf

from glob import glob
from models.transforms import HadamardCoder
from .transforms import random_flip_img_label, \
                       random_crop_img_label, \
                       photometric_distortion, \
                       random_resize_img_label, \
                       normalize_img, \
                       loading_img_label

def CityscapesDataset(path: str, 
                      batch_size: int = 4, 
                      image_shape: tuple[int, int] = (1024, 2048), 
                      crop_size: tuple[int, int] = (512, 1024), 
                      validation_shape: tuple[int, int] = (512, 1024), 
                      hadamard: bool =False, 
                      num_classes: int = 19, 
                      codes_size: int = 32,
                      shuffle: bool = False):

    dataset_dict = {}
    splits = ['train', 'val', 'test']
    
    if hadamard:
        hadamard_coder = HadamardCoder(num_classes=num_classes, code_size=codes_size)

    CITYSCAPES_MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
    CITYSCAPES_STD  = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

    for split in splits:
        img_pattern  = os.path.join(path, 'leftImg8bit', split, '*', '*.png')
        mask_pattern = os.path.join(path, 'gtFine',      split, '*', '*_labelTrainIds.png')

        img_files  = sorted(glob(img_pattern))
        mask_files = sorted(glob(mask_pattern))

        ds = tf.data.Dataset.from_tensor_slices((img_files, mask_files))
        if shuffle:
            ds = ds.shuffle(len(img_files), reshuffle_each_iteration=True)

        if split == 'train':
            ds = ds.map(lambda x, y: loading_img_label(x, y, image_shape), num_parallel_calls=tf.data.AUTOTUNE)
            #ds = ds.map(lambda x, y: random_resize_img_label(x, y), num_parallel_calls=tf.data.AUTOTUNE)
            ds = ds.map(lambda x, y: random_crop_img_label(x, y, crop_size), num_parallel_calls=tf.data.AUTOTUNE) #(768,768) // (1024,1024)
            ds = ds.map(lambda x, y: random_flip_img_label(x, y), num_parallel_calls=tf.data.AUTOTUNE)
            #ds = ds.map(lambda x, y: (photometric_distortion(x), y), num_parallel_calls=tf.data.AUTOTUNE)
            ds = ds.map(lambda x, y: (normalize_img(x, CITYSCAPES_MEAN, CITYSCAPES_STD), y), num_parallel_calls=tf.data.AUTOTUNE)
            if hadamard:
                ds = ds.map(lambda x, y: (x, y, hadamard_coder.encode(y)), num_parallel_calls=tf.data.AUTOTUNE)
            ds = ds.batch(batch_size)

        else:
            ds = ds.map(lambda x, y: loading_img_label(x, y, validation_shape), num_parallel_calls=tf.data.AUTOTUNE)
            ds = ds.map(lambda x, y: (normalize_img(x, CITYSCAPES_MEAN, CITYSCAPES_STD), y), num_parallel_calls=tf.data.AUTOTUNE)
            ds = ds.batch(1)

        ds = ds.prefetch(tf.data.AUTOTUNE)
        dataset_dict[split] = ds

    return dataset_dict