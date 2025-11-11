import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from matplotlib.colors import LinearSegmentedColormap

def GenerateImages(model, x_input, y_input, save_path, filename):
    colors_01 = [
        (128/255,  64/255, 128/255),  # road
        (244/255,  35/255, 232/255),  # sidewalk
        ( 70/255,  70/255,  70/255),  # building
        (102/255, 102/255, 156/255),  # wall
        (190/255, 153/255, 153/255),  # fence
        (153/255, 153/255, 153/255),  # pole
        (250/255, 170/255,  30/255),  # traffic light
        (220/255, 220/255,   0/255),  # traffic sign
        (107/255, 142/255,  35/255),  # vegetation
        (152/255, 251/255, 152/255),  # terrain
        ( 70/255, 130/255, 180/255),  # sky
        (220/255,  20/255,  60/255),  # person
        (255/255,   0/255,   0/255),  # rider
        (  0/255,   0/255, 142/255),  # car
        (  0/255,   0/255,  70/255),  # truck
        (  0/255,  60/255, 100/255),  # bus
        (  0/255,  80/255, 100/255),  # train
        (  0/255,   0/255, 230/255),  # motorcycle
        (119/255,  11/255,  32/255),  # bicycle
        (  0/255,   0/255,   0/255)   # background

    ]
    
    seg_cmap = LinearSegmentedColormap.from_list('seg_colors', colors_01)
    
    # con training=True se obtienen las metricas sobre el Lote. 
    # En otro caso, no se evaluan y se regresan las del entrenamiento.
    y_pred, probs, __ = model(x_input, training=True)  
    plt.figure(figsize=(18, 5))
    display_list = [ y_input[0], x_input[0], y_pred[0], probs[0]]

    plt.subplot(1, 4, 1)
    im = display_list[1]
    m, M = tf.reduce_min(im).numpy(), tf.reduce_max(im).numpy()
    im = (im-m)/(M-m)
    plt.imshow(im) 
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    img = display_list[0].numpy()
    img = np.squeeze(img, axis=-1)
    plt.imshow(img, cmap=seg_cmap, interpolation='nearest', vmin=0, vmax=18) 
    plt.axis('off')

    plt.subplot(1, 4, 3)
    mask = np.argmax(display_list[3].numpy(), axis=-1)         
    mask = np.where(img == 255, 255, mask)
    plt.imshow(mask, cmap=seg_cmap, interpolation='nearest', vmin=0, vmax=18)
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(im)
    plt.imshow(mask, cmap=seg_cmap, alpha=.5, interpolation='nearest', vmin=0, vmax=18)
    plt.axis('off')
    
    # plt.show()
    plt.tight_layout()
    plt.savefig(save_path + filename)
    return