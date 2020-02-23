# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 02:33:47 2020

@author: Kosmas Pinitas
"""
from skimage.transform import resize
from skimage.color import rgb2gray
import numpy as np

# 210*160*3(rgb) to 84*84(grayscale) and floats to ints for memory size reduction
def preprocessing(next_frame, curr_frame):
    max_frame = np.maximum(next_frame, curr_frame) 
    gray_frame = rgb2gray(max_frame)
    frame = np.uint8(resize(gray_frame, (84, 84), mode='constant') * 255)
    return frame

