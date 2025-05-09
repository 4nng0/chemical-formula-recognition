import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the image
image_name = '3.jpeg'
image_path = os.path.join(os.getcwd(), 'notext_images', image_name)
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

# Edge detection (Canny Edge Detection)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Line detection using Hough Transform
lines = cv2.HoughLines(edges, 1, np.pi / 180, 180)

# Draw detected lines
if lines is not None:
    for rho, theta in lines[:, 0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red lines

# Display the image
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Detected Lines")
plt.axis("off")
plt.show()