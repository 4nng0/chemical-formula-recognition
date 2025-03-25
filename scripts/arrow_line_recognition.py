import numpy as np
import cv2
import tensorflow as tf
import os
import matplotlib.pyplot as plt

from scripts import detect

# Enable GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Enabled GPU memory growth.")
    except RuntimeError as e:
        print(f"Error enabling GPU memory growth: {e}")


def detect_lines(image):
    edges = cv2.Canny(image, 50, 150, apertureSize=5)

    lines_list = []
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=10, maxLineGap=8)

    for points in lines:
        x1, y1, x2, y2 = points[0]
        lines_list.append([(x1, y1), (x2, y2)])
    return lines_list


def visualize_results(image, centroids, intersecting_lines):
    """
    Visualize the centroids and intersecting lines on the image.
    """
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Mark centroids
    for cx, cy in centroids[1:]:
        cv2.circle(output_image, (int(cx), int(cy)), 5, (0, 255, 0), -1)  # Green dots

    # Draw lines
    for (x1, y1), (x2, y2) in intersecting_lines:
        cv2.line(output_image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red lines

    plt.figure(figsize=(8, 8))
    plt.title("Centroids and Intersecting Lines")
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()
    return output_image

# Bounding box calculation
def get_rotated_bounding_box(x1, y1, x2, y2, threshold=20):
    # Convert the line to a list of points
    points = np.array([[x1, y1], [x2, y2]], dtype=np.float32)

    # Get the minimum area rectangle (rotated bounding box)
    rect = cv2.minAreaRect(points)

    # Expand rectangle dimensions (add flexibility by threshold)
    center, (width, height), angle = rect
    rect = (center, (width + threshold, height + threshold), angle)

    # Get corners of the rectangle
    box = cv2.boxPoints(rect)
    box = np.int0(box)  # Convert corner points to integers

    return box, rect


def find_lines_intersecting_components(detected_lines, labels, centroids, threshold=10):
    """
    Find intersecting lines for detected components and remove centroids with no intersecting lines.
    """
    intersecting_lines = []
    valid_centroids = [centroids[0]]  # Keep the background centroid (index 0)
    visited_lines = set()

    for idx, (cx, cy) in enumerate(centroids[1:], start=1):  # Skip background centroid
        has_intersecting_line = False
        for (x1, y1), (x2, y2) in detected_lines:
            # Check proximity of line to the centroid
            x_min = min(x1, x2) - threshold
            x_max = max(x1, x2) + threshold
            y_min = min(y1, y2) - threshold
            y_max = max(y1, y2) + threshold

            if x_min <= cx <= x_max and y_min <= cy <= y_max:
                intersecting_lines.append(((x1, y1), (x2, y2)))
                visited_lines.add(((x1, y1), (x2, y2)))
                # Backtrack to find more intersecting lines
                find_intersecting_lines(detected_lines, intersecting_lines, visited_lines, x1, x2, y1, y2, threshold)
                has_intersecting_line = True
                break  # Add only one line per centroid

        if has_intersecting_line:
            valid_centroids.append((cx, cy))  # Only keep centroids with intersecting lines

    return intersecting_lines, valid_centroids


def find_intersecting_lines(detected_lines, intersecting_lines, visited_lines, x1, x2, y1, y2, threshold=10):
    # Calculate bounding box of the line
    x_min = min(x1, x2) - threshold
    x_max = max(x1, x2) + threshold
    y_min = min(y1, y2) - threshold
    y_max = max(y1, y2) + threshold

    for (X1, Y1), (X2, Y2) in detected_lines:
        if ((X1, Y1), (X2, Y2)) not in visited_lines:
            if x_min <= X1 <= x_max and y_min <= Y1 <= y_max or x_min <= X2 <= x_max and y_min <= Y2 <= y_max:
                intersecting_lines.append(((X1, Y1), (X2, Y2)))
                visited_lines.add(((X1, Y1), (X2, Y2)))
                find_intersecting_lines(detected_lines, intersecting_lines, visited_lines, X1, X2, Y1, Y2)


def filter_short_lines(lines, min_length=30):
    """
    Filter out lines shorter than the specified minimum length.
    """
    filtered_lines = []
    for line in lines:
        (x1, y1), (x2, y2) = line
        # Calculate the length of the line
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if length >= min_length:
            filtered_lines.append(line)
    return filtered_lines


def get_result(image_path, model_path, result_path):
    binary_mask = detect.arrow_heads(image_path, model_path)
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask)
    print(f"Total connected components (excluding background): {num_labels - 1}")

    detected_lines = detect_lines(original_image)

    # Find intersecting lines
    intersecting_lines, centroids = find_lines_intersecting_components(detected_lines, labels, centroids)

    # Filter short lines
    intersecting_lines = filter_short_lines(intersecting_lines)
    intersecting_lines, centroids = find_lines_intersecting_components(intersecting_lines, labels, centroids)
    
    output_image = visualize_results(original_image, centroids, intersecting_lines)
    cv2.imwrite(result_path, output_image)


if __name__ == "__main__":
    # Image path
    script_path = os.getcwd()
    base_path = os.path.dirname(script_path)
    test_images = 'test_images'
    image_name = '8.jpeg'
    model_name = 'unet_model_512.keras'
    image_path = os.path.join(base_path, test_images, image_name)
    model_path = os.path.join(base_path,'saved_models', model_name)
    
    get_result(image_path, model_path, os.path.join(os.getcwd(), 'result_image.jpg'))