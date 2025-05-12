import os
import cv2
import numpy as np

from scripts import detect

model_location = "saved_models/unet_model_512_version_1.keras"

def testPixelAccuracy(numberOfPictures):
    script_path = os.getcwd()
    base_path = os.path.dirname(script_path)
    model_path = os.path.join(base_path, model_location)

    precision = 0
    recall = 0


    for i in range(1, numberOfPictures + 1):
        picture = i
        imagePath = "realPictures/" + str(picture) + ".jpg"
        predicted = detect.arrow_heads(imagePath, model_path)
        image = cv2.imread("binaryMasks/" + str(picture) + ".png", cv2.IMREAD_GRAYSCALE)  # Graustufenbild laden
        real = (image > 127).astype(np.uint8)

        tp = np.sum(predicted * real)  # TP
        fp = np.sum(predicted  * (1 - real))  # FP
        fn = np.sum((1 - predicted ) * real)  # FN
        tn = np.sum((1 - predicted ) * (1 - real))  # TN

        precision += tp / (tp + fp )
        recall += tp / (tp + fn)

        print("pixel precision for picture " + str(picture) + ": " + str(tp / (tp + fp )))
        print("pixel recall for picture " + str(picture) + ": " + str(tp / (tp + fn)))


    precision = precision / numberOfPictures
    recall = recall / numberOfPictures
    f1 = 2 * (precision * recall) / (precision + recall)
    print("average pixel precision: " + str(precision))
    print("average pixel recall: " + str(recall))
    print("pixel f1: " + str(f1))


def testCentroidAccuracy(numberOfPictures):
    script_path = os.getcwd()
    base_path = os.path.dirname(script_path)
    model_path = os.path.join(base_path, model_location)


    numberOfPictures
    recall = 0
    accuracy = 0
    precision = 0


    for i in range(1, numberOfPictures + 1):
        picture = i
        imagePath = "realPictures/" + str(picture) + ".jpg"
        result_path = os.path.join(script_path, 'test_results', 'centroid_' + str(picture) + "_result.png")
        predicted = detect.arrow_heads(imagePath, model_path)
        image = cv2.imread("binaryMasks/" + str(picture) + ".png", cv2.IMREAD_GRAYSCALE)  # Graustufenbild laden
        real = (image > 127).astype(np.uint8)

        p_num_labels, p_labels, p_stats, p_centroids = cv2.connectedComponentsWithStats(predicted)
        r_num_labels, r_labels, r_stats, r_centroids = cv2.connectedComponentsWithStats(real)


        #makes a circle around the centroid to the width and height of the object
        #assumes that the arrows are all the same size

        list_matches = []
        list_real_without_partner = []

        centroid_list = [tuple(centroid) for centroid in p_centroids[1:]] # Skip background centroid
        i = 0

        for r_idx, (cx, cy) in enumerate(r_centroids[1:], start=1):  # Skip background centroid
            i += 1
            threshold = max(r_stats[i, cv2.CC_STAT_WIDTH], r_stats[i, cv2.CC_STAT_HEIGHT])
            min_diff = 2 * threshold

            for (x1, y1) in centroid_list:
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



        cv2.imwrite(result_path, output_image)

        recall += (list_matches.__len__()) / r_centroids.__len__()
        accuracy += list_matches.__len__() /(list_matches.__len__() + list_real_without_partner.__len__() + list_predictedWithoutPartner.__len__())
        precision += (list_matches.__len__()) / p_centroids.__len__()


    recall = recall / numberOfPictures
    accuracy = accuracy / numberOfPictures
    precision = precision / numberOfPictures
    f1 = 2 * (precision * recall) / (precision + recall)
    print("recall: " + str(recall))
    print("accuracy: " + str(accuracy))
    print("precision: " + str(precision))
    print("F1: " + str(f1))



if __name__ == "__main__":

    numberOfPictures = 13

    testPixelAccuracy(numberOfPictures)

    testCentroidAccuracy(numberOfPictures)