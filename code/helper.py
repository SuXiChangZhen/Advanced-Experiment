# -*- coding: utf-8 -*-
# some useful function
import numpy as np 
from matplotlib import pyplot as plt 

def perturbation(pert, img):
    pixel_count = pert.shape[1] // 5
    pert = pert.reshape((pert.shape[0], pixel_count, 5))
    result = []
    for p in pert:
        image = np.copy(img)
        for pixel in p:
            x, y, *rgb = pixel
            image[:, int(x), int(y)] = rgb
        result.append(image)
    return np.array(result)

def img_show(img):
    if len(img.shape) == 4:
        image = np.squeeze(img)
    else:
        image = img.copy()
    image = image.transpose((1,2,0))
    plt.figure();plt.imshow(image)
 
