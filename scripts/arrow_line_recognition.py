import numpy as np
import cv2
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from skimage.metrics import hausdorff_distance

from scripts import detect

#TODO: get the threshold to a better value,
#TODO: somehow support - - - > arrows.
#TODO: get a way to test the accuracy
#TODO: only one line per centroid
#TODO: splitting lines with more points of interest
#TODO: some arrows are to short, whats up with that?

model_name = 'unet_model_512_version_1.keras'

# Enable GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Enabled GPU memory growth.")
    except RuntimeError as e:
        print(f"Error enabling GPU memory growth: {e}")


def reflect_points(points, center, direction):
    # Normiere Richtungsvektor
    direction = direction / np.linalg.norm(direction)

    # Vektor von center zu jedem Punkt
    vecs = points - center

    # Projektion auf Achse
    proj = np.dot(vecs, direction[:, None]) * direction

    # Spiegelung: p' = p - 2 * (p_proj - center)
    reflected = center + proj - (vecs - proj)
    return reflected

def points_to_mask(points, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    for y, x in np.round(points).astype(int):
        if 0 <= y < shape[0] and 0 <= x < shape[1]:
            mask[y, x] = 1
    return mask

def compute_best_symmetry_axes(binary_mask, show=False):
    axes = {}
    details_show = False

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask)
    for label in range(1, num_labels):
        component_mask = (labels == label).astype(np.uint8)
        points = np.column_stack(np.where(component_mask > 0))
        if len(points) < 5:
            continue

        # PCA zur Bestimmung der Hauptachsen
        # am ende war es besser einfach durch alles durchzugehen aber vielleicht kann man das verbessern damit es schneller wird
        mean = np.mean(points, axis=0)
        centered = points - mean
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eig(cov)
        axis1 = eigvecs[:, np.argmax(eigvals)]
        axis2 = np.array([-axis1[1], axis1[0]])


        best_score = float('inf')
        best_info = None
        original_mask = points_to_mask(points, component_mask.shape)

        for angle in np.arange(0, 180, 15):
            theta = np.deg2rad(angle)
            direction = np.array([ np.cos(theta), np.sin(theta)])

            reflected = reflect_points(points, mean, direction)
            mask = points_to_mask(reflected, component_mask.shape)

            if details_show:
                # Normiere die Masken, falls nötig (auf 0–1 Bereich)
                original_mask = original_mask.astype(np.float32)
                mask = mask.astype(np.float32)

                # Erstelle ein leeres RGB-Bild
                overlay = np.zeros((*original_mask.shape, 3), dtype=np.float32)

                # Weist die Farben zu
                overlay[..., 0] = original_mask  # Rotkanal
                overlay[..., 2] = mask  # Blaukanal

                # Optional: Clip Werte auf 0–1
                overlay = np.clip(overlay, 0, 1)

                # Anzeige
                plt.figure(figsize=(6, 6))
                plt.imshow(overlay)
                plt.title(f'Overlay: Red = Original, Blue = Reflected, angle = {angle}')
                plt.axis('off')
                plt.tight_layout()
                plt.show()

                plt.figure(figsize=(10, 5))

            # Konturen vergleichen (original und gespiegelt)
            d = hausdorff_distance(mask, original_mask)


            #print(f"Label {label}: Hausdorff-Distanz = {d:.2f} angle = {angle}")

            if d < best_score:
                best_score = d
                best_info = {
                    'center': tuple(mean[::-1]),
                    'axis': tuple(direction[::-1]),
                    'hausdorff': d
                }

        if best_info:
            axes[label] = best_info

    # Optional: Visualisierung
    if show:
        plt.figure(figsize=(8, 8))
        plt.imshow(binary_mask, cmap='gray')
        for info in axes.values():
            cx, cy = info['center']
            dx, dy = info['axis']
            plt.arrow(cx - dx * 50, cy - dy * 50, dx * 100, dy * 100, color='red', head_width=5)
        plt.title("Beste Symmetrieachsen (PCA + Hausdorff)")
        plt.axis('off')
        plt.show()

    return num_labels, labels, stats, centroids, axes

def detect_lines(image):
    show = False

    edges = cv2.Canny(image, 50, 150, apertureSize=5)

    if show:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(edges, cmap='gray')
        plt.title('Canny Edges')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 5))

    lines_list = []
    #lsd methode
    lines = []

    gray_image = (image * 255).astype(np.uint8)
    lsd = cv2.createLineSegmentDetector()

    # Linien detektieren
    lines = lsd.detect(gray_image)[0]
    #lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=5, maxLineGap=8)

    for points in lines:
        x1, y1, x2, y2 = points[0]
        lines_list.append([(int(x1), int(y1)), (int(x2), int(y2))])

    return lines_list


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

def get_distance_norm_with_axis(line, centroid, axis):
    (x1, y1), (x2, y2) = line
    (cx, cy) = centroid
    x_diff_1 = cx - x1
    y_diff_1 = cy - y1
    x_diff_2 = cx - x2
    y_diff_2 = cy - y2
    diff_1 = np.sqrt(x_diff_1 ** 2 + y_diff_1 ** 2)
    diff_2 = np.sqrt(x_diff_2 ** 2 + y_diff_2 ** 2)

    line_vec = np.array([x2 - x1, y2 - y1])
    axis_vec = np.array(axis.get('axis'))

    cos_theta = np.dot(line_vec, axis_vec) / (np.linalg.norm(line_vec) * np.linalg.norm(axis_vec))

    return min(diff_1, diff_2) + (1 - cos_theta) * 10 # Add a large penalty for non-parallel lines


def get_distance_norm(line, centroid):
    (x1, y1), (x2, y2) = line
    (cx, cy) = centroid
    x_diff_1 = cx - x1
    y_diff_1 = cy - y1
    x_diff_2 = cx - x2
    y_diff_2 = cy - y2
    diff_1 = np.sqrt(x_diff_1 ** 2 + y_diff_1 ** 2)
    diff_2 = np.sqrt(x_diff_2 ** 2 + y_diff_2 ** 2)

    # returns the smaller distance and the point of the line that is farther away from the centroid
    if diff_1 <  diff_2:
        return diff_1, (x2, y2)
    else:
        return diff_2, (x1, y1)

def find_lines_that_point_to_centroids(detected_lines, stats, centroids,  threshold=10):
    """
    Find lines intersecting with centroids and remove centroids with no intersecting lines.
    """
    intersecting_lines = []
    valid_centroids = [centroids[0]]  # Keep the background centroid (index 0)
    element = 0

    #go over the list of lines and match ist with best matching centroid
    for r_idx, (cx, cy) in enumerate(centroids[1:], start=1):  # Skip background centroid
        element += 1
        threshold = max(stats[element, cv2.CC_STAT_WIDTH], stats[element, cv2.CC_STAT_HEIGHT]) * 2
        min_diff = 2 * threshold

        for line in detected_lines:

            this_diff, point = get_distance_norm(line, (cx, cy))

            if this_diff < min_diff:
                min_diff = this_diff
                best_match = line
                this_line = [(cx, cy), point]


        if min_diff < threshold:

            intersecting_lines.append(this_line)
            detected_lines.remove(best_match)
            valid_centroids.append((cx, cy))



    # Filter out lines that are too short
    return intersecting_lines , valid_centroids, detected_lines


def backtrack_lines(original_image, detected_lines, intersecting_lines, threshold=10):
    queue = list(range(len(intersecting_lines)))

    for i in queue:
        this_line = intersecting_lines[i]
        (x, y) = this_line [-1]
        min_diff = 2 * threshold

        for line in detected_lines:
            this_diff, point = get_distance_norm(line, (x, y))

            if this_diff < min_diff:
                min_diff = this_diff
                best_match = line
                best_point = point

        if min_diff < threshold:

            intersecting_lines[i].append(best_point)
            diff = get_distance_norm(best_match, (x, y))
            #visualize_results(original_image, [], [intersecting_lines[i]], info= f"min-diff = {min_diff}, diff {diff} \n this line{this_line} \n best_match {best_match} \n {x}, {y} ")
            detected_lines.remove(best_match)
            queue.append(i)


    return intersecting_lines

def find_lines_intersecting_components(detected_lines, centroids, threshold=10):
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
    binary_mask = cv2.imread(model_path, cv2.IMREAD_GRAYSCALE)
    #binary_mask =detect.arrow_heads(image_path, model_path)
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    detected_lines = detect_lines(original_image)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask)

    print(f"Total connected components (excluding background): {num_labels - 1}")

    detected_lines = filter_short_lines(detected_lines)
    visualize_results(binary_mask, centroids, [])
    # Find intersecting lines
    # TODO maybe pre process the lines to make them more correct
    # TODO here we need to make it so we can only match one line with one point of interest and in a certain distance
    intersecting_lines, new_centroids = find_lines_that_point_to_centroids(detected_lines, stats, centroids, axes={})
    visualize_results(binary_mask, new_centroids, [])
    return 0


    # Filter short lines
    #TODO: what does this do?
    #intersecting_lines = filter_short_lines(intersecting_lines)
    #intersecting_lines, new_centroids = find_lines_intersecting_components(intersecting_lines, labels, new_centroids)
    
    output_image = visualize_results(original_image, new_centroids, intersecting_lines)

    cv2.imwrite(result_path, output_image)

def overlaps(line1, line2, dist_thresh):
    (l1_x1, l1_y1), (l1_x2, l1_y2) = line1
    (l2_x1, l2_y1), (l2_x2, l2_y2) = line2
    x_min = min(l1_x1, l1_x2) - dist_thresh
    x_max = max(l1_x1, l1_x2) + dist_thresh
    y_min = min(l1_y1, l1_y2) - dist_thresh
    y_max = max(l1_y1, l1_y2) + dist_thresh

    return (x_min <= l2_x1 <= x_max and y_min <= l2_y1 <= y_max) or (x_min <= l2_x2 <= x_max and y_min <= l2_y2 <= y_max)


def angles_are_similar(line1, line2, angle_thresh=10):
    (l1_x1, l1_y1), (l1_x2, l1_y2) = line1
    (l2_x1, l2_y1), (l2_x2, l2_y2) = line2
    angle1 = np.degrees(np.arctan2(l1_y2 - l1_y1, l1_x2 - l1_x1)) % 180
    angle2 = np.degrees(np.arctan2(l2_y2 - l2_y1, l2_x2 - l2_x1)) % 180

    # Winkelähnlichkeit prüfen
    return abs(angle1 - angle2) <= angle_thresh



def combine_cluster_lines(original_image, lines, target_lines, angle_thresh=12, dist_thresh=10):
    clusters = []
    used = [False] * len(lines)

    for target_line in target_lines:
        cluster = [target_line]

        angles_match = []

        for j in range(len(lines)):
            if not used[j] and angles_are_similar(target_line, lines[j], angle_thresh):
                angles_match.append(j)

        #print(f'targetline : {target_line} \n angles_match: {angles_match}')

        for j in range(len(angles_match)):
            if not used[j] and (overlaps(target_line, lines[j], dist_thresh) or overlaps(lines[j], target_line, dist_thresh)):
                cluster.append(lines[j])

                #print(f"Clustered line {lines[j]} with target line {target_line}")
                used[j] = True
                lines.remove(lines[j])

        clusters.append(cluster)


    intersecting_lines = []

    for cluster in clusters:
        corner1 = []
        corner2 = []
        angles = []
        for line in cluster:
            (x1, y1), (x2, y2) = line
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if angle >= 0:
                corner1.append((x2, y2))
                corner2.append((x1, y1))
            else:
                corner1.append((x1, y1))
                corner2.append((x2, y2))

            angles.append(angle % 180)

        angle = np.mean(angles)
        #print(angles)
        line = (0,0), (0,0)
        #print(corner1)
        #visualize_results(original_image, corner1, cluster, info= f'number of lines: {len(cluster)} \n angle: {angle}')
        if angle < (22.5):
            x1 = max(x for x, y in corner1)
            y1 = np.mean([y for x, y in corner1])
            x2 = min(x for x, y in corner2)
            y2 = np.mean([y for x, y in corner2])
            line = (int(x1), int(y1)), (int(x2), int(y2))
        elif angle < (67.5):
            x1 = min(min(x for x, y in corner1), min(x for x, y in corner2))
            x2 = max( max(x for x, y in corner1), max(x for x, y in corner2))
            y1 = min(min(y for x, y in corner1), min(y for x, y in corner2))
            y2 = max(max(y for x, y in corner1), max(y for x, y in corner2))
            line = (int(x1), int(y1)), (int(x2), int(y2))
        elif angle < (112.5):
            y1 = max(y for x, y in corner1)
            x1 = np.mean([x for x, y in corner1])
            y2 = min(y for x, y in corner2)
            x2 = np.mean([x for x, y in corner2])
            line = (int(x1), int(y1)), (int(x2), int(y2))
        elif angle < (157.5):
            x2 = min(min(x for x, y in corner1), min(x for x, y in corner2))
            x1 = max(max(x for x, y in corner1), max(x for x, y in corner2))
            y1= min(min(y for x, y in corner1), min(y for x, y in corner2))
            y2 = max(max(y for x, y in corner1), max(y for x, y in corner2))
            line = (int(x1), int(y1)), (int(x2), int(y2))
        else:
            x1 = max(x for x, y in corner2)
            y1 = np.mean([y for x, y in corner2])
            x2 = min(x for x, y in corner2)
            y2 = np.mean([y for x, y in corner2])
            line = (int(x1), int(y1)), (int(x2), int(y2))
        intersecting_lines.append(line)

    print(intersecting_lines)

    return intersecting_lines, lines


def try_stuff(image_path, mask_path):
    binary_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    #binary_mask =detect.arrow_heads(image_path, model_path)
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    #without_arrow = original_image.copy()

    #without_arrow[binary_mask == 255] = 255


    detected_lines = detect_lines(original_image)
    detected_lines = filter_short_lines(detected_lines)



    visualize_results(original_image, [], detected_lines)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask)
    #num_labels, labels, stats, centroids, axes = compute_best_symmetry_axes(binary_mask)


    # Find intersecting lines
    intersecting_lines, valid_centroids, detected_lines = find_lines_that_point_to_centroids(detected_lines, stats, centroids)

    intersecting_lines, detected_lines= combine_cluster_lines(original_image, detected_lines, intersecting_lines)

    intersecting_lines = match_line(valid_centroids, intersecting_lines)
    #visualize_results(original_image, valid_centroids, intersecting_lines)
    #intersecting_lines = backtrack_lines(original_image, detected_lines, intersecting_lines, threshold = 10)

    #intersecting_lines, new_centroids = find_lines_intersecting_components(detected_lines, centroids)

    visualize_results(original_image, valid_centroids, intersecting_lines)


if __name__ == "__main__":
    # Image path


    for i in range(12, 13):
        script_path = os.getcwd()
        base_path = os.path.dirname(script_path)
        test_images = 'test/realPictures'
        binary_mask = 'test/binaryMasks'
        image_path = os.path.join(base_path, test_images,  f"{i}.jpg")
        #image_path = os.path.join(base_path, f'datasetGeneration/data/i{i}.jpg')
        model_path = os.path.join(base_path,'saved_models', model_name)
        mask_path = os.path.join(base_path, binary_mask, f"{i}.png")
        #mask_path = os.path.join(base_path, f'datasetGeneration/data/m{i}.jpg')
        result_path = os.path.join(base_path, 'test_results', f"centroid_{i}_result.png")
        try_stuff(image_path, mask_path)
        # right now we use the right binary mask and not the one from the model
        #get_result(image_path, mask_path, result_path)


