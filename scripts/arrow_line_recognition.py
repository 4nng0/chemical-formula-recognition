import numpy as np
import cv2
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from visualisation import visualize_results, visualize_simple_graph
from shapely.geometry import LineString
from shapely.ops import nearest_points
from reaction_graph import Graph, draw_graph_ignoring_position
from scripts import detect

# Enable GPU memory growth to prevent TensorFlow from allocating all GPU memory at once
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Enabled GPU memory growth.")
    except RuntimeError as e:
        print(f"Error enabling GPU memory growth: {e}")


def detect_lines(image):
    """
    Detects lines in an image using OpenCV's LineSegmentDetector.
    Returns a list of lines as point pairs [(x1, y1), (x2, y2)].
    """
    show = False  # Toggle for visual debugging

    edges = cv2.Canny(image, 50, 150, apertureSize=5)

    if show:
        # Visual debug output
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

    lines_list = []

    gray_image = (image * 255).astype(np.uint8)  # Convert to 8-bit image for OpenCV
    lsd = cv2.createLineSegmentDetector()  # Create Line Segment Detector

    # Detect lines using LSD
    lines = lsd.detect(gray_image)[0]
    if lines is None:
        return []

    # Convert detected lines to tuple format
    for points in lines:
        x1, y1, x2, y2 = points[0]
        lines_list.append([(int(x1), int(y1)), (int(x2), int(y2))])

    return lines_list


def get_distance_norm(line, centroid):
    """
    Computes distance between a centroid and a line.
    Returns the shortest distance and the closest endpoint of the line.
    """
    (x1, y1), (x2, y2) = line
    (cx, cy) = centroid

    # Compute distances from centroid to both endpoints
    diff_1 = np.sqrt((cx - x1) ** 2 + (cy - y1) ** 2)
    diff_2 = np.sqrt((cx - x2) ** 2 + (cy - y2) ** 2)

    return (diff_1, (x1, y1)) if diff_1 < diff_2 else (diff_2, (x2, y2))


def find_lines_that_point_to_centroids(detected_lines, stats, centroids, threshold=10):
    """
    Matches lines to the nearest centroids.
    Removes lines not pointing to any centroids and invalid centroids.
    """
    intersecting_lines = []
    valid_centroids = [centroids[0]]  # Background centroid

    for r_idx, (cx, cy) in enumerate(centroids[1:], start=1):
        # Dynamic threshold based on object size
        threshold = max(stats[r_idx, cv2.CC_STAT_WIDTH], stats[r_idx, cv2.CC_STAT_HEIGHT]) * 2
        min_diff = 2 * threshold

        for line in detected_lines:
            this_diff, _ = get_distance_norm(line, (cx, cy))
            if this_diff < min_diff:
                min_diff = this_diff
                best_match = line

        if min_diff < threshold:
            intersecting_lines.append(best_match)
            detected_lines.remove(best_match)
            valid_centroids.append((cx, cy))

    return intersecting_lines, valid_centroids, detected_lines


def backtrack_lines(original_image, lines, graph, threshold=10):
    """
    Backtracks from graph edges to expand the graph by finding spatially close lines.
    Uses geometric distance to connect nearby line segments.
    """
    queue = list(graph.edges())  # Initialize queue with existing edges

    for v, u in queue:
        line1 = LineString([v, u])  # Line from graph edge
        min_diff = 2 * threshold

        # Try to match current edge with unused line segments
        for line in lines:
            (x1, y1), (x2, y2) = line
            line2 = LineString([(x1, y1), (x2, y2)])

            if line1.distance(line2) < min_diff:
                min_diff = line1.distance(line2)
                best_match = line

        # Look for closest edge in the graph
        for v2, u2 in graph.edges():
            if Graph.are_connected(graph, v, v2):
                continue
            line2 = LineString([v2, u2])
            if line1.distance(line2) < min_diff:
                min_diff = line1.distance(line2)
                best_match = (v2, u2)

        if min_diff < threshold:
            best_match, lines = combine_cluster_lines_targeted(original_image, lines, [best_match])
            (x1, y1), (x2, y2) = best_match[0]
            line2 = LineString([(x1, y1), (x2, y2)])

            p1, p2 = nearest_points(line1, line2)
            p1, p2 = (int(p1.x), int(p1.y)), (int(p2.x), int(p2.y))

            # Skip if we are at the origin of the arrow
            if Graph.is_beginning_node(graph, p1):
                continue

            # Add matched line to the graph
            if p2 == (x1, y1) or p2 == (x2, y2):
                graph.add_node((x1, y1), (x1, y1), "line")
                graph.add_node((x2, y2), (x2, y2), "line")
                graph.add_edge((x1, y1), (x2, y2))
                queue.append(((x1, y1), (x2, y2)))
            else:
                graph.add_node((x1, y1), (x1, y1), "line")
                graph.add_node((x2, y2), (x2, y2), "line")
                graph.add_node(p2, p2, "line")
                graph.add_edge((x1, y1), p2)
                graph.add_edge((x2, y2), p2)
                queue.append(((x1, y1), p2))
                queue.append(((x2, y2), p2))

            # Add edge from p1 to p2 or replace original edge with split
            if p1 == u or p1 == v:
                graph.add_edge(p1, p2)
                queue.append((p1, p2))
                queue.append((u, v))
            else:
                graph.add_node(p1, p1, "line")
                graph.remove_edge(u, v)
                graph.add_edge(p1, u)
                graph.add_edge(p1, v)
                graph.add_edge(p2, p1)
                queue.append((p1, u))
                queue.append((p1, v))

    # Final step: connect very close nodes
    for u in graph.nodes:
        for v in graph.nodes:
            if u != v and np.linalg.norm(np.subtract(u, v)) < threshold:
                graph.add_edge(u, v)

    return graph


def filter_short_lines(lines, min_length=30):
    """
    Removes lines shorter than a minimum length threshold.
    """
    filtered_lines = []
    for line in lines:
        (x1, y1), (x2, y2) = line
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if length >= min_length:
            filtered_lines.append(line)
    return filtered_lines


def overlaps(line1, line2, leng_thresh=20, width_thresh=10):
    """
    Determines if line2 lies within a band around line1 (extended rectangle).
    """
    (l1_x1, l1_y1), (l1_x2, l1_y2) = line1
    (l2_x1, l2_y1), (l2_x2, l2_y2) = line2

    # Compute direction and length
    dx = l1_x2 - l1_x1
    dy = l1_y2 - l1_y1
    length = np.hypot(dx, dy)

    if length == 0:
        raise ValueError("wrong line coordinates: length is zero")

    unit_vector = (dx / length, dy / length)
    normalvector = (-unit_vector[1], unit_vector[0])  # Perpendicular vector

    # Create a rectangle (thickened line) around line1
    longline1 = ((l1_x1 - unit_vector[0] * leng_thresh + normalvector[0] * width_thresh,
                  l1_y1 - unit_vector[1] * leng_thresh + normalvector[1] * width_thresh),
                 (l1_x2 + unit_vector[0] * leng_thresh + normalvector[0] * width_thresh,
                  l1_y2 + unit_vector[1] * leng_thresh + normalvector[1] * width_thresh))

    longline2 = ((l1_x1 - unit_vector[0] * leng_thresh - normalvector[0] * width_thresh,
                  l1_y1 - unit_vector[1] * leng_thresh - normalvector[1] * width_thresh),
                 (l1_x2 + unit_vector[0] * leng_thresh - normalvector[0] * width_thresh,
                  l1_y2 + unit_vector[1] * leng_thresh - normalvector[1] * width_thresh))

    # Build polygon corners
    corners = np.array([longline1[0], longline1[1], longline2[1], longline2[0]], dtype=np.float32)

    # Test if endpoints of line2 lie within the polygon
    return cv2.pointPolygonTest(corners, (l2_x1, l2_y1), False) > 0 or \
           cv2.pointPolygonTest(corners, (l2_x2, l2_y2), False) > 0


def combine_clusters(clusters, angle_thresh=6):
    """
    Combines clusters of similar lines into a single representative line per cluster.
    """
    intersecting_lines = []
    all_found = True  # Flag to indicate if every cluster had only one line

    for cluster in clusters:
        if len(cluster) > 1:
            all_found = False
        else:
            intersecting_lines.append(cluster[0][0])
            continue

        # Separate endpoints by directionality
        corner1, corner2, angles = [], [], []
        for line, ang in cluster:
            (x1, y1), (x2, y2) = line
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
            if angle - angle_thresh >= 0:
                corner1.append((x2, y2))
                corner2.append((x1, y1))
            else:
                corner1.append((x1, y1))
                corner2.append((x2, y2))
            angles.append(angle)

        # Estimate average angle and construct combined line
        angle = np.mean(angles)
        line = []

        if angle < 10 or angle > 170:
            # Horizontal line
            x1 = max(x for x, y in corner2)
            x2 = min(x for x, y in corner1)
            y = int(np.mean([y for x, y in corner1 + corner2]))
            line = [(x1, y), (x2, y)]
        elif angle < 80 or angle > 100:
            # Diagonal line
            x1 = min(x for x, y in corner1 + corner2)
            y1 = min(y for x, y in corner1 + corner2)
            x2 = max(x for x, y in corner1 + corner2)
            y2 = max(y for x, y in corner1 + corner2)
            line = [(x1, y1), (x2, y2)]
        else:
            # Vertical line
            y1 = max(y for x, y in corner1)
            y2 = min(y for x, y in corner2)
            x1 = int(np.mean([x for x, y in corner1]))
            x2 = int(np.mean([x for x, y in corner2]))
            line = [(x1, y1), (x2, y2)]

        intersecting_lines.append(line)

    return intersecting_lines, all_found


def angles_are_similar(line1, line2, angle_thresh):
    """
    Compares two lines and returns True if their angles differ by less than angle_thresh degrees.
    for these comparisons angles that have a difference of 180 degrees are considered equal.
    """
    (x1, y1), (x2, y2) = line1
    angle1 = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180

    (x1, y1), (x2, y2) = line2
    angle2 = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180

    return abs(angle1 - angle2) < angle_thresh or 180 - abs(angle1 - angle2) < angle_thresh

def combine_cluster_lines_targeted(original_image, lines, target_lines, angle_thresh=6, dist_thresh=10, show=False):
    """
    Clusters lines based on angular and spatial similarity with a set of target lines.

    Parameters:
        original_image (ndarray): The original image for optional visualization.
        lines (list): List of detected lines [(pt1, pt2), ...].
        target_lines (list): Reference lines to compare against.
        angle_thresh (float): Maximum allowed angle difference for clustering.
        dist_thresh (float): Maximum allowed spatial distance for clustering.
        show (bool): Whether to visualize results.

    Returns:
        tuple: (intersecting_lines, remaining_lines) if all clusters have size 1,
               else recursively clusters again.
    """
    clusters = []
    used = [False] * len(lines)

    for target_line in target_lines:
        cluster = [target_line]
        angles_match = []

        # Find lines with similar angle to target
        for j in range(len(lines)):
            if not used[j] and angles_are_similar(target_line, lines[j], angle_thresh):
                angles_match.append(j)

        # Check for spatial overlap and add to cluster
        for j in range(len(angles_match)):
            i = angles_match[j]
            if not used[i]:
                if overlaps(target_line, lines[i], dist_thresh) or overlaps(lines[i], target_line, dist_thresh):
                    cluster.append(lines[i])
                    used[i] = True

        clusters.append(cluster)

    # Remove clustered lines from original lines list
    result = []
    for i in range(len(lines)):
        if not used[i]:
            result.append(lines[i])
    lines = result

    # Optional visualization
    if show:
        for cluster in clusters:
            visualize_results(original_image, [], cluster, info=f'number of lines {len(cluster)} \n taget_line: {cluster[0]}')

    intersecting_lines = []
    all_found = True

    for cluster in clusters:
        if len(cluster) > 1:
            all_found = False

        corner1 = []
        corner2 = []
        angles = []

        # Compute average angle and determine start/end points for merged line
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
        line = []

        # Heuristically determine endpoints of merged line based on angle
        if angle < 10:
            x1 = max(x for x, y in corner2)
            y1 = np.mean([y for x, y in corner1])
            x2 = min(x for x, y in corner1)
            y2 = np.mean([y for x, y in corner2])
        elif angle < 80:
            x1 = min(min(x for x, y in corner1), min(x for x, y in corner2))
            x2 = max(max(x for x, y in corner1), max(x for x, y in corner2))
            y1 = min(min(y for x, y in corner1), min(y for x, y in corner2))
            y2 = max(max(y for x, y in corner1), max(y for x, y in corner2))
        elif angle < 100:
            y1 = max(y for x, y in corner1)
            x1 = np.mean([x for x, y in corner1])
            y2 = min(y for x, y in corner2)
            x2 = np.mean([x for x, y in corner2])
        elif angle < 170:
            x1 = max(max(x for x, y in corner1), max(x for x, y in corner2))
            y1 = min(min(y for x, y in corner1), min(y for x, y in corner2))
            x2 = min(min(x for x, y in corner1), min(x for x, y in corner2))
            y2 = max(max(y for x, y in corner1), max(y for x, y in corner2))
        else:
            x1 = max(x for x, y in corner2)
            y1 = np.mean([y for x, y in corner2])
            x2 = min(x for x, y in corner1)
            y2 = np.mean([y for x, y in corner1])

        line.append((int(x1), int(y1)))
        line.append((int(x2), int(y2)))
        intersecting_lines.append(line)

    # Recursively re-cluster if not all clusters are single-line
    if all_found:
        return intersecting_lines, lines
    else:
        return combine_cluster_lines_targeted(original_image, lines, intersecting_lines, angle_thresh=angle_thresh, dist_thresh=dist_thresh)


def combine_cluster_lines(original_image, lines, angle_thresh=6, dist_thresh=10, show=False):
    """
    Clusters lines based on angle and spatial overlap into larger connected segments.

    Parameters:
        original_image (ndarray): The original image for optional visualization.
        lines (list): List of detected lines [(pt1, pt2), ...].
        angle_thresh (float): Maximum allowed angle difference for clustering.
        dist_thresh (float): Maximum allowed spatial distance for clustering.
        show (bool): Whether to visualize the clusters.

    Returns:
        list: List of merged/intersecting lines after clustering.
    """
    clusters = []
    used = [False] * len(lines)

    lines_with_angle = []
    for line in lines:
        (x1, y1), (x2, y2) = line
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        lines_with_angle.append((line, angle))

    lines_with_angle = sorted(lines_with_angle, key=lambda x: x[1])

    for idx, (target_line, target_angle) in enumerate(lines_with_angle):
        if used[idx]:
            continue

        cluster = [(target_line, target_angle)]
        used[idx] = True  # Mark starting line as used

        # Search backward in sorted list
        next = idx - 1
        while next >= 0 and min(abs(target_angle - lines_with_angle[next][1]), 180 - abs(target_angle - lines_with_angle[next][1])) <= angle_thresh:
            if not used[next]:
                if overlaps(target_line, lines_with_angle[next][0], dist_thresh) or overlaps(lines_with_angle[next][0], target_line, dist_thresh):
                    cluster.append(lines_with_angle[next])
                    used[next] = True
            next -= 1

        # Search forward in sorted list
        next = idx + 1
        while next < len(lines_with_angle) and min(abs(target_angle - lines_with_angle[next][1]), 180 - abs(target_angle - lines_with_angle[next][1])) <= angle_thresh:
            if not used[next]:
                if overlaps(target_line, lines_with_angle[next][0], dist_thresh) or overlaps(lines_with_angle[next][0], target_line, dist_thresh):
                    cluster.append(lines_with_angle[next])
                    used[next] = True
            next += 1

        clusters.append(cluster)

    if show:
        for cluster in clusters:
            visualize_results(original_image, [], [x[0] for x in cluster], info=f'number of lines {len(cluster)} \n taget_line: {cluster[0][0]}')

    # Merge clustered lines
    intersecting_lines, all_found = combine_clusters(clusters, angle_thresh=angle_thresh)

    if all_found:
        return intersecting_lines
    else:
        return combine_cluster_lines(original_image, intersecting_lines, angle_thresh=angle_thresh, dist_thresh=dist_thresh)


def match_lines(centroids, lines):
    """
    Matches detected lines to nearby centroids and builds a graph.

    Parameters:
        centroids (ndarray): Array of centroid coordinates.
        lines (list): List of detected lines [(pt1, pt2), ...].

    Returns:
        Graph: A graph connecting centroids and lines based on proximity.
    """
    arrow_graph = Graph()

    for r_idx, (cx, cy) in enumerate(centroids[1:], start=1):  # Skip background centroid
        threshold = 30
        min_diff = 2 * threshold

        for line in lines:
            this_diff, point = get_distance_norm(line, (cx, cy))
            if this_diff < min_diff:
                min_diff = this_diff
                best_match = line
                this_line = [(cx, cy), point]

        if min_diff < threshold:
            cx, cy = int(cx), int(cy)
            arrow_graph.add_node((cx, cy), (cx, cy), "centroid")
            (x1, y1), (x2, y2) = best_match
            arrow_graph.add_node((x1, y1), (x1, y1), "line")
            arrow_graph.add_node((x2, y2), (x2, y2), "line")
            arrow_graph.add_edge((x1, y1), (x2, y2))

            _, point = get_distance_norm(best_match, (cx, cy))
            arrow_graph.add_edge((cx, cy), point)

    return arrow_graph


def applies_pipline(image_path, model_path):
    """
    Runs the full arrow detection pipeline including clustering and graph construction.

    Parameters:
        image_path (str): Path to the input grayscale image.
        model_path (str): Path to the model for line detection.

    Returns:
        Graph: The final graph representation of lines and centroids.
    """

    cluster_before = True

    binary_mask = detect.arrow_heads_py(image_path, model_path)
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    without_arrow = original_image.copy()
    without_arrow[binary_mask == 255] = 255

    detected_lines = detect_lines(without_arrow)
    visualize_results(original_image, [], detected_lines, info="", path="2")

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask)
    visualize_results(original_image, centroids, [], info="", path="1")

    if cluster_before:
        detected_lines = combine_cluster_lines(original_image, detected_lines)
        detected_lines = filter_short_lines(detected_lines)
        visualize_results(original_image, [], detected_lines, info="", path="3")
        intersecting_lines, valid_centroids, detected_lines = find_lines_that_point_to_centroids(detected_lines, stats, centroids)
        visualize_results(original_image, [], intersecting_lines, info="", path="4")
    else:
        detected_lines = detect_lines(without_arrow)
        detected_lines = filter_short_lines(detected_lines)
        intersecting_lines, valid_centroids, detected_lines = find_lines_that_point_to_centroids(detected_lines, stats, centroids)
        intersecting_lines, detected_lines = combine_cluster_lines_targeted(original_image, detected_lines, intersecting_lines)

    graph = match_lines(valid_centroids, intersecting_lines)
    visualize_simple_graph(original_image, graph, info="", path="5")
    graph = backtrack_lines(original_image, detected_lines, graph, threshold=10)

    Graph.merge_close_nodes(graph, threshold=5)

    visualize_simple_graph(original_image, graph)
    visualize_simple_graph(original_image, graph, info="", path="")

    return graph


if __name__ == "__main__":
    for i in range(6, 7):
        script_path = os.getcwd()
        base_path = os.path.dirname(script_path)
        test_images = 'test/realPictures'
        binary_mask = 'test/binaryMasks'
        image_path = os.path.join(base_path, test_images, f"{i}.jpg")
        model_path = os.path.join(base_path, 'saved_models', model_name)
        mask_path = os.path.join(base_path, binary_mask, f"{i}.png")
        applies_pipline(image_path, model_path)




