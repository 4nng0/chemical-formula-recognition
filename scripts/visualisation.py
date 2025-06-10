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

    #print("Image shape:", image_rgb.shape)
    #print("Overlay shape:", overlay.shape)

    # Stelle sicher, dass die Maske nicht auf Weiß unsichtbar wird
    overlayed_image = image_rgb.copy()
    overlayed_image[binary_mask > 0] = overlay[binary_mask > 0]  # Direkte Überlagerung

    # Anzeige des Bildes mit überlagerter Maske
    plt.imshow(overlayed_image)
    plt.title("Arrow Heads Detected with Binary Mask Overlay")
    plt.axis("off")
    plt.show()
    return plt

def visualize_results(image, centroids, intersecting_lines, info=None):

    """
    Visualize the centroids and intersecting lines on the image.
    """
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Mark centroids
    for cx, cy in centroids[1:]:
        cv2.circle(output_image, (int(cx), int(cy)), 10, (0, 255, 0), -1)  # Green dots

    # Draw lines
    for line_list in intersecting_lines:
        start = line_list[0]
        start = (int(start[0]), int(start[1]))
        for i in range(1, len(line_list)):
            next = line_list[i]
            cv2.line(output_image, start, next, (0, 0, 255), 2)  # Red lines
            start = next

    #
    #         plt.imshow(binary_mask, cmap='gray')
    #         for info in axes.values():
    #             cx, cy = info['center']
    #             dx, dy = info['axis']
    #             plt.arrow(cx - dx * 50, cy - dy * 50, dx * 100, dy * 100, color='red', head_width=5)
    #         plt.title("Beste Symmetrieachsen (PCA + Hausdorff)")
    #         plt.axis('off')
    #         plt.show()

    plt.figure(figsize=(8, 8))
    if info is not None:
        plt.title(info)
    else:
        plt.title("Centroids and Intersecting Lines")

    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    """for line_list in intersecting_lines:
        start = line_list[0]
        start = (int(start[0]), int(start[1]))
        for i in range(1, len(line_list)):
            next = line_list[i]
            plt.arrow(start[0], start[1], next[0] - start[0], next[1] -start[1], color='red', head_width=15 ) # Red lines
            start = next"""

    plt.axis("off")
    plt.show()
    return output_image


