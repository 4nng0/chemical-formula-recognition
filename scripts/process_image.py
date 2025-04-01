#Processes images from scratch to detect arrow heads and lines in the image.

import numpy as np
import cv2
import tensorflow as tf
import os
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from remove_text import remove_text
from arrow_line_recognition import get_result



if __name__ == '__main__' :


    script_path = os.getcwd()
    base_path = os.path.dirname(script_path)
    test_images = 'test_images'
    notext_images = 'notext_images'
    result_images = 'result_images'
    image_name = '8.jpeg'
    model_name = 'unet_model_512.keras'
    
    image_path = os.path.join(base_path, test_images, image_name)
    notext_image_path = os.path.join(base_path, notext_images, image_name)
    result_path = os.path.join(base_path, result_images, image_name)
    model_path = os.path.join(base_path, 'saved_models', model_name)
    
    remove_text(image_path, notext_image_path)

    print('Text removed from image')

    get_result(notext_image_path, model_path, result_path)



