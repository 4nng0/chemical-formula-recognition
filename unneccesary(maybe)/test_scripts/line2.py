import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
# Read image
image_name = '3.jpeg'
image_path = os.path.join(os.getcwd(), 'notext_images', image_name)
image = cv2.imread(image_path)

# Convert image to grayscale
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# Use canny edge detection
edges = cv2.Canny(gray,50,150,apertureSize=5)

# Apply HoughLinesP method to 
# to directly obtain line end points
lines_list =[]
lines = cv2.HoughLinesP(
            edges, # Input edge image
            1, # Distance resolution in pixels
            np.pi/180, # Angle resolution in radians
            threshold=60, # Min number of votes for valid line
            minLineLength=5, # Min allowed length of line
            maxLineGap=8 # Max allowed gap between line for joining them
            )

# Iterate over points
for points in lines:
      # Extracted points nested in the list
    x1,y1,x2,y2=points[0]
    # Draw the lines joing the points
    # On the original image
    cv2.line(image,(x1,y1),(x2,y2),(0,255,0),2)
    # Maintain a simples lookup list for points
    lines_list.append([(x1,y1),(x2,y2)])

plt.figure(figsize=(10, 10))
plt.title("Detected Lines")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Renkleri dönüştür
plt.axis("off")
plt.show()
# Save the result image
cv2.imwrite('detectedLines.png',image)