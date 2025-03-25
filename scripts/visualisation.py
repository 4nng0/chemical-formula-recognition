import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import matplotlib.pyplot as plt



def display_image_with_mask(image_path, binary_mask):
    overlay = np.zeros((*binary_mask.shape, 3), dtype=np.uint8)
    overlay[binary_mask > 0] = [255, 30, 0]  # Rote Maske

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    print("Image shape:", image_rgb.shape)
    print("Overlay shape:", overlay.shape)

    # Stelle sicher, dass die Maske nicht auf Weiß unsichtbar wird
    overlayed_image = image_rgb.copy()
    overlayed_image[binary_mask > 0] = overlay[binary_mask > 0]  # Direkte Überlagerung

    # Anzeige des Bildes mit überlagerter Maske
    plt.imshow(overlayed_image)
    plt.title("Arrow Heads Detected with Binary Mask Overlay")
    plt.axis("off")
    plt.show()