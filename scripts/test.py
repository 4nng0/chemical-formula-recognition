import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import matplotlib.pyplot as plt
import detect
import visualisation

# Enable GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Enabled GPU memory growth.")
    except RuntimeError as e:
        print(f"Error enabling GPU memory growth: {e}")


if __name__ == "__main__":
    # Get the paths of all images in the specified directory
    script_path = os.getcwd()
    base_path = os.path.dirname(script_path)
    image_dir = os.path.join(base_path, 'test_images')
    model_path = os.path.join(base_path, 'saved_models/unet_model_512.keras')


    image_path = os.path.join(base_path, 'data/images/', '9.jpg')
        # Call detect_arrow_heads function
    binary_map = detect.arrow_heads(image_path, model_path)

        # Visualize the results
    pic = visualisation.display_image_with_mask(image_path, binary_map)
    pic.show()
