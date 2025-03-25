import json
import os

import numpy as np
import cv2
import scripts.visualisation as visualisation


if __name__ == "__main__":
    import numpy as np
    import cv2
    import json
    import matplotlib.pyplot as plt

    picture = 2
    # Lade die JSON-Daten
    annotation_path = "binaryMasks/Total.JSON"
    with open(annotation_path) as f:
        data = json.load(f)

    # Extrahiere Bildgröße aus der ersten Annotation
    original_width = data[picture - 1]["annotations"][0]["result"][0]["original_width"]
    original_height = data[picture - 1]["annotations"][0]["result"][0]["original_height"]

    # Erstelle eine leere Maske (schwarz = 0, weiß = 255 für annotierte Bereiche)
    binary_mask = np.zeros((original_height, original_width), dtype=np.uint8)

    # Durchlaufe alle Annotationen und zeichne die Polygone in die Maske
    for annotation in data[picture - 1]["annotations"][0]["result"]:
        if annotation["type"] == "polygonlabels":
            points = np.array(annotation["value"]["points"])  # Polygon-Punkte extrahieren

            # Skalierung der relativen Koordinaten in absolute Pixel
            points[:, 0] = (points[:, 0] / 100) * original_width
            points[:, 1] = (points[:, 1] / 100) * original_height
            points = points.astype(np.int32)  # Integer-Koordinaten für OpenCV

            # Zeichne das Polygon auf die Maske (weiß = 255)
            cv2.fillPoly(binary_mask, [points], 255)

    # Optional: Maske als Bild speichern
    cv2.imwrite("binaryMasks/" + str(picture) + ".png", binary_mask)

    base_path = os.getcwd()
    image_path = os.path.join(base_path, "realPictures/" + str(picture) + ".jpg")

    visualisation.display_image_with_mask(image_path, binary_mask)