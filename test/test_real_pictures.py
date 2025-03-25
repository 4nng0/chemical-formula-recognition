import os

import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from torch.ao.nn.quantized.functional import threshold

from scripts import detect, visualisation


def testPixelAccuracy():
    script_path = os.getcwd()
    base_path = os.path.dirname(script_path)
    model_path = os.path.join(base_path, 'saved_models/unet_model_512.keras')

    total = 0
    numberOfPictures = 1

    for i in range(1, numberOfPictures + 1):
        picture = i
        imagePath = "realPictures/" + str(picture) + ".jpg"
        predicted = detect.arrow_heads(imagePath, model_path)
        image = cv2.imread("binaryMasks/" + str(picture) + ".png", cv2.IMREAD_GRAYSCALE)  # Graustufenbild laden
        real = (image > 127).astype(np.uint8)

        # predicted = np.array([[1, 0, 0, 0],[1, 1, 0, 0]])
        # real = np.array([[1, 1, 0, 0],[1, 1, 1, 0]])

        # visualisation.display_image_with_mask(imagePath, predicted)

        temp = real - predicted
        temp = np.abs(temp)
        correct_pixels = temp.size - np.sum(temp)

        # visualisation.display_image_with_mask(imagePath, temp)
        total_pixels = real.size

        print("correct pixels: " + str(correct_pixels))
        accuracy = correct_pixels / total_pixels
        total += accuracy
        print("pixel accuracy for picture " + str(picture) + ": " + str(accuracy))

        correct_pixels = np.sum(real == predicted)
        accuracy = correct_pixels / total_pixels
        print("correct pixels: " + str(correct_pixels))
        print("pixel accuracy for picture " + str(picture) + ": " + str(accuracy))
        # TODO: calculate pixel accuracy FIND MISTAKE

    total = total / numberOfPictures
    print("average pixel accuracy: " + str(total))


def testCentroidAccuracy():
    script_path = os.getcwd()
    base_path = os.path.dirname(script_path)
    model_path = os.path.join(base_path, 'saved_models/unet_model_512.keras')

    total = 0
    numberOfPictures = 13

    for i in range(1, numberOfPictures + 1):
        picture = i
        imagePath = "realPictures/" + str(picture) + ".jpg"
        predicted = detect.arrow_heads(imagePath, model_path)
        image = cv2.imread("binaryMasks/" + str(picture) + ".png", cv2.IMREAD_GRAYSCALE)  # Graustufenbild laden
        real = (image > 127).astype(np.uint8)

        p_num_labels, p_labels, p_stats, p_centroids = cv2.connectedComponentsWithStats(predicted)
        r_num_labels, r_labels, r_stats, r_centroids = cv2.connectedComponentsWithStats(real)

        #print(r_centroids)
        #print(p_centroids)

        #makes a circle around the centroid to the width and height of the object
        #assumes that the arrows are all the same size
        threshold = max(p_stats[1, cv2.CC_STAT_WIDTH], p_stats[1, cv2.CC_STAT_HEIGHT])
        min_idx = -1

        list_matches = []
        list_real_without_partner = []

        centroid_list = [tuple(centroid) for centroid in p_centroids[1:]] # Skip background centroid


        for r_idx, (cx, cy) in enumerate(r_centroids[1:], start=1):  # Skip background centroid
            min_diff = 2 * threshold


            for  (x1, y1) in centroid_list:
                x_diff = cx - x1
                y_diff = cy - y1
                this_diff = np.sqrt(x_diff ** 2 + y_diff ** 2)

                if this_diff < min_diff:
                    min_diff = this_diff
                    best_match = (x1, y1)

            if min_diff < threshold:
                list_matches.append(((cx, cy), best_match))
                centroid_list.remove(best_match)
            else:
                list_real_without_partner.append((cx, cy))

        list_predictedWithoutPartner = centroid_list

        image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)

        output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Mark centroids
        for (cx, cy) in list_predictedWithoutPartner:
            cv2.circle(output_image, (int(cx), int(cy)), 5, (255, 0, 0), -1)  # Red dots


        for (cx, cy) in list_real_without_partner:
            cv2.circle(output_image, (int(cx), int(cy)), 5, (0, 0, 255), -1)  # Blue dots

        for (rx, ry),(px, py) in list_matches:
            cv2.circle(output_image, (int(rx), int(ry)), 5, (0, 255, 0), -1)  # Green dots
            cv2.circle(output_image, (int(px), int(py)), 5, (0, 255, 0), -1)


        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()



if __name__ == "__main__":

    #testPixelAccuracy()

    testCentroidAccuracy()