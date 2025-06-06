import numpy as np
import cv2
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from skimage.metrics import hausdorff_distance
from visualisation import visualize_results
from shapely.geometry import LineString
from shapely.ops import nearest_points
import decimer_segmentation as ds


from scripts import detect

class Node:
    def __init__(self, node_id, position, typ):
        self.id = node_id
        self.position = position  # z. B. (x, y)
        self.typ = typ            # z. B. "line", "arrowhead"
        self.neighbors = set()    # IDs der Nachbarn (ungerichtet)

    def add_neighbor(self, other_id):
        self.neighbors.add(other_id)


class Graph:
    def __init__(self):
        self.nodes = {}  # ID → Node

    def add_node(self, node_id, position, typ):
        if node_id not in self.nodes:
            self.nodes[node_id] = Node(node_id, position, typ)

    def add_edge(self, id1, id2):
        if id1 in self.nodes and id2 in self.nodes:
            self.nodes[id1].add_neighbor(id2)
            self.nodes[id2].add_neighbor(id1)  # ungerichtet!

    def get_neighbors(self, node_id):
        return self.nodes[node_id].neighbors

    def __getitem__(self, node_id):
        return self.nodes[node_id]

    def edges(self):
        seen = set()
        for node in self.nodes.values():
            for neighbor in node.neighbors:
                edge = tuple(sorted([node.id, neighbor]))
                if edge not in seen:
                    seen.add(edge)
                    yield edge

    def remove_edge(self, id1, id2):
        if id1 in self.nodes and id2 in self.nodes[id1].neighbors:
            self.nodes[id1].neighbors.remove(id2)
        if id2 in self.nodes and id1 in self.nodes[id2].neighbors:
            self.nodes[id2].neighbors.remove(id1)


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

def visualize_simple_graph(image, graph, info=None):
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


    for v, u in graph.edges():
        position1 = graph.nodes[v].position
        position2 = graph.nodes[u].position
        cv2.line(output_image, position1, position2, (255, 0, 0), 4)
    # Mark centroids
    for node in graph.nodes.values():
        if node.typ == "centroid":
            cv2.circle(output_image, (int(node.position[0]), int(node.position[1])), 10, (0, 255, 0), -1)
        if node.typ == "line":
            cv2.circle(output_image, (int(node.position[0]), int(node.position[1])), 5, (0, 0, 255), -1)



    plt.figure(figsize=(8, 8))
    if info is not None:
        plt.title(info)
    else:
        plt.title("Graph Visualization")
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()
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
    """returns the distance of the centroid to the line and the point of the line that is closer the centroid"""
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
        return diff_1, (x1, y1)
    else:
        return diff_2, (x2, y2)

def find_lines_that_point_to_centroids(detected_lines, stats, centroids,  threshold=10):
    """
    Find lines intersecting with centroids and remove centroids with no intersecting lines.
    """
    intersecting_lines = []
    valid_centroids = [centroids[0]]  # Keep the background centroid (index 0)
    element = 0

    #go over the list of lines and match ist with best matching centroid
    for r_idx, (cx, cy) in enumerate(centroids[1:], start=1):  # Skip background centroid
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



    # Filter out lines that are too short
    return intersecting_lines , valid_centroids, detected_lines

def backtrack_lines(original_image, lines, graph, threshold=5):

    queue = []

    for v, u in graph.edges():
        queue.append((v, u))

    for v, u in queue:
        line1 = LineString([v, u])

        min_diff = 2 * threshold

        for line in lines:
            (x1, y1), (x2, y2) = line
            line2 = LineString([(x1, y1), (x2, y2)])

            if line1.distance(line2) < min_diff:
                min_diff = line1.distance(line2)
                best_match = line

        if min_diff < threshold:

            best_match, lines = combine_cluster_lines_targeted(original_image, lines, [best_match])
            (x1, y1), (x2, y2) = best_match[0]
            line2 = LineString([(x1, y1), (x2, y2)])

            p1, p2 = nearest_points(line1, line2)
            p1, p2 = (int(p1.x), int(p1.y)), (int(p2.x), int(p2.y))


            if p2 == (x1, y1) or p2 == (x1, y1):
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

            if p1 == u or p1 == v:
                graph.add_edge(p1, p2)
                queue.append((p1, p2))
            else:
                graph.add_node(p1,p1,"line")
                graph.remove_edge(u, v)
                graph.add_edge(p1, u)
                graph.add_edge(p1, v)
                graph.add_edge(p2, p1)
                queue.append((p1, u))
                queue.append((p1, v))


    for u in graph.nodes:
        for v in graph.nodes:

            x_diff = u[0] - v[0]
            y_diff=  u[1] - v[1]
            if np.sqrt(x_diff ** 2 + y_diff ** 2) < threshold and u != v:
                graph.add_edge(u, v)

    return graph

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

def overlaps(line1, line2, leng_thresh = 10, width_thresh=10):
    (l1_x1, l1_y1), (l1_x2, l1_y2) = line1
    (l2_x1, l2_y1), (l2_x2, l2_y2) = line2

    dx = l1_x2 - l1_x1
    dy = l1_y2 - l1_y1

    length = np.hypot(dx, dy)

    if length != 0:
        unit_vector = (dx / length, dy / length)
    else:
        raise ValueError("Ungültiger Wert!")

    longline1 = (l1_x1 - unit_vector[0] * leng_thresh, l1_y1 - unit_vector[1] * leng_thresh), (l1_x2 + unit_vector[0] * leng_thresh, l1_y2 + unit_vector[1] * leng_thresh)
    longline2 = (l1_x1 - unit_vector[0] * leng_thresh, l1_y1 - unit_vector[1] * leng_thresh), (l1_x2 + unit_vector[0] * leng_thresh, l1_y2 + unit_vector[1] * leng_thresh)


    normalvector = (-unit_vector[1], unit_vector[0])


    longline1 = ((longline1[0][0] + normalvector[0] * width_thresh, longline1[0][1] + normalvector[1] * width_thresh), (longline1[1][0] + normalvector[0] * width_thresh, longline1[1][1] + normalvector[1] * width_thresh))

    longline2 = ((longline2[0][0] + normalvector[0] * -width_thresh, longline2[0][1] + normalvector[1] * -width_thresh), (longline2[1][0] + normalvector[0] * -width_thresh, longline2[1][1] + normalvector[1] * -width_thresh))

    ecken = np.array([longline1[0], longline1[1], longline2[1], longline2[0]], dtype=np.float32)

    return cv2.pointPolygonTest(ecken, (l2_x1, l2_y1), measureDist=False) > 0 or cv2.pointPolygonTest(ecken, (l2_x2, l2_y2), measureDist=False) > 0

def angles_are_similar(line1, line2, angle_thresh=10):
    (l1_x1, l1_y1), (l1_x2, l1_y2) = line1
    (l2_x1, l2_y1), (l2_x2, l2_y2) = line2
    angle1 = np.degrees(np.arctan2(l1_y2 - l1_y1, l1_x2 - l1_x1)) % 180
    angle2 = np.degrees(np.arctan2(l2_y2 - l2_y1, l2_x2 - l2_x1)) % 180

    # Winkelähnlichkeit prüfen
    return abs(angle1 - angle2) <= angle_thresh

def combine_cluster_lines_targeted(original_image, lines, target_lines, angle_thresh=6, dist_thresh=10, show=False):
    clusters = []
    used = [False] * len(lines)

    for target_line in target_lines:
        cluster = [target_line]

        angles_match = []



        for j in range(len(lines)):
            if not used[j] and angles_are_similar(target_line, lines[j], angle_thresh):
                angles_match.append(j)


        #print(f"target line: {target_line} \n angles_match: {angles_match}")
        #for angle in angles_match:
        #    print(f"angle {lines[angle]} with target line {target_line}")


        for j in range(len(angles_match)):
            i = angles_match[j]
            if not used[i]:
                if overlaps(target_line, lines[i], dist_thresh) or overlaps(lines[i], target_line, dist_thresh):
                    cluster.append(lines[i])
                    #print(f"Clustered line {lines[i]} with target line {target_line}")
                    used[i] = True


        clusters.append(cluster)


    result = []
    for i in range(len(lines)):
        if not used[i]:
            result.append(lines[i])
    lines = result

    if show:

        for cluster in clusters:
            visualize_results(original_image, [], cluster, info= f'number of lines {len(cluster)} \n taget_line: {cluster[0]}')

    intersecting_lines = []

    all_found = True
    #mistakes in [(1161,1379),(1375,1307)] [(955,941), (908,954)]

    for cluster in clusters:
        if len(cluster) > 1:
            all_found = False
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
        line = list()
        #print(corner1)
        #visualize_results(original_image, corner1, cluster, info= f'number of lines: {len(cluster)} \n angle: {angle}')
        if angle < (10):
            x1 = max(x for x, y in corner2)
            y1 = np.mean([y for x, y in corner1])
            x2 = min(x for x, y in corner1)
            y2 = np.mean([y for x, y in corner2])
            line.append((int(x1), int(y1)))
            line.append((int(x2), int(y2)))
        elif angle < (80):
            x1 = min(min(x for x, y in corner1), min(x for x, y in corner2))
            x2 = max( max(x for x, y in corner1), max(x for x, y in corner2))
            y1 = min(min(y for x, y in corner1), min(y for x, y in corner2))
            y2 = max(max(y for x, y in corner1), max(y for x, y in corner2))
            line.append((int(x1), int(y1)))
            line.append((int(x2), int(y2)))
        elif angle < (100):
            y1 = max(y for x, y in corner1)
            x1 = np.mean([x for x, y in corner1])
            y2 = min(y for x, y in corner2)
            x2 = np.mean([x for x, y in corner2])
            line.append((int(x1), int(y1)))
            line.append((int(x2), int(y2)))
        elif angle < (170):
            x1 = max(max(x for x, y in corner1), max(x for x, y in corner2))
            y1= min(min(y for x, y in corner1), min(y for x, y in corner2))
            x2 = min(min(x for x, y in corner1), min(x for x, y in corner2))
            y2 = max(max(y for x, y in corner1), max(y for x, y in corner2))
            line.append((int(x1), int(y1)))
            line.append((int(x2), int(y2)))
        else:
            x1 = max(x for x, y in corner2)
            y1 = np.mean([y for x, y in corner2])
            x2 = min(x for x, y in corner1)
            y2 = np.mean([y for x, y in corner1])
            line.append((int(x1), int(y1)))
            line.append((int(x2), int(y2)))
        intersecting_lines.append(line)

    #return intersecting_lines, lines
    if all_found:
        return intersecting_lines, lines
    else:
        return combine_cluster_lines_targeted(original_image, lines, intersecting_lines, angle_thresh=angle_thresh, dist_thresh=dist_thresh)

def combine_cluster_lines(original_image, lines, angle_thresh=6, dist_thresh=20, show=False):
    clusters = []
    used = [False] * len(lines)

    for idx, target_line in enumerate(lines):
        if used[idx]:
            continue

        cluster = [target_line]
        used[idx] = True  # markiere die Startlinie sofort als verwendet

        for j in range(len(lines)):
            if not used[j] and angles_are_similar(target_line, lines[j], angle_thresh):
                if overlaps(target_line, lines[j], dist_thresh) or overlaps(lines[j], target_line, dist_thresh):
                    cluster.append(lines[j])
                    used[j] = True

        clusters.append(cluster)

    if show:

        for cluster in clusters:
            visualize_results(original_image, [], cluster, info= f'number of lines {len(cluster)} \n taget_line: {cluster[0]}')

    intersecting_lines = []

    all_found = True

    for cluster in clusters:
        if len(cluster) > 1:
            all_found = False
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
        line = list()
        #print(corner1)
        #visualize_results(original_image, corner1, cluster, info= f'number of lines: {len(cluster)} \n angle: {angle}')
        if angle < (10):
            x1 = max(x for x, y in corner2)
            y1 = np.mean([y for x, y in corner1])
            x2 = min(x for x, y in corner1)
            y2 = np.mean([y for x, y in corner2])
            line.append((int(x1), int(y1)))
            line.append((int(x2), int(y2)))
        elif angle < (80):
            x1 = min(min(x for x, y in corner1), min(x for x, y in corner2))
            x2 = max( max(x for x, y in corner1), max(x for x, y in corner2))
            y1 = min(min(y for x, y in corner1), min(y for x, y in corner2))
            y2 = max(max(y for x, y in corner1), max(y for x, y in corner2))
            line.append((int(x1), int(y1)))
            line.append((int(x2), int(y2)))
        elif angle < (100):
            y1 = max(y for x, y in corner1)
            x1 = np.mean([x for x, y in corner1])
            y2 = min(y for x, y in corner2)
            x2 = np.mean([x for x, y in corner2])
            line.append((int(x1), int(y1)))
            line.append((int(x2), int(y2)))
        elif angle < (170):
            x1 = max(max(x for x, y in corner1), max(x for x, y in corner2))
            y1= min(min(y for x, y in corner1), min(y for x, y in corner2))
            x2 = min(min(x for x, y in corner1), min(x for x, y in corner2))
            y2 = max(max(y for x, y in corner1), max(y for x, y in corner2))
            line.append((int(x1), int(y1)))
            line.append((int(x2), int(y2)))
        else:
            x1 = max(x for x, y in corner2)
            y1 = np.mean([y for x, y in corner2])
            x2 = min(x for x, y in corner1)
            y2 = np.mean([y for x, y in corner1])
            line.append((int(x1), int(y1)))
            line.append((int(x2), int(y2)))
        intersecting_lines.append(line)

    if all_found:
        return intersecting_lines
    else:
        return combine_cluster_lines(original_image, intersecting_lines, angle_thresh=angle_thresh, dist_thresh=dist_thresh)

def match_lines(centroids, lines):
    arrow_graph = Graph()

    #go over the list of lines and match ist with best matching centroid
    for r_idx, (cx, cy) in enumerate(centroids[1:], start=1):  # Skip background centroid
        threshold = 30
        min_diff = 2 * threshold

        for line in lines:

            this_diff, point = get_distance_norm(line, (cx, cy))

            if this_diff < min_diff:
                min_diff = this_diff
                best_match = line
                this_line = list()
                this_line.append((cx, cy))
                this_line.append(point)

        if min_diff < threshold:

            arrow_graph.add_node((int(cx), int(cy)), (int(cx), int(cy)), "centroid")
            (x1, y1), (x2, y2) = best_match
            arrow_graph.add_node((x1, y1), (x1, y1), "line")
            arrow_graph.add_node((x2, y2), (x2, y2), "line")
            arrow_graph.add_edge((x1, y1), (x2, y2))

            diff, point = get_distance_norm(best_match, (cx, cy))
            arrow_graph.add_edge((cx, cy), point)

    return arrow_graph

def get_result(image_path, mask_path):
    """
    binary_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    #binary_mask =detect.arrow_heads(image_path, model_path)
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    without_arrow = original_image.copy()

    without_arrow[binary_mask == 255] = 255


    detected_lines = detect_lines(without_arrow)
    visualize_results(original_image, [], detected_lines)
    detected_lines = combine_cluster_lines(original_image, detected_lines)
    visualize_results(original_image, [], detected_lines)
    detected_lines = filter_short_lines(detected_lines)
    visualize_results(original_image, [], detected_lines)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask)
    graph = match_lines(centroids, detected_lines)

    visualize_simple_graph(original_image, graph)

    graph = backtrack_lines(original_image, detected_lines, graph, threshold=10)

    visualize_simple_graph(original_image, graph)




    """
    binary_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    #binary_mask =detect.arrow_heads(image_path, model_path)
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    without_arrow = original_image.copy()

    without_arrow[binary_mask == 255] = 255

    detected_lines = detect_lines(without_arrow)
    detected_lines = filter_short_lines(detected_lines)


    visualize_results(original_image, [], detected_lines)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask)
    #num_labels, labels, stats, centroids, axes = compute_best_symmetry_axes(binary_mask)


    # Find intersecting lines
    intersecting_lines, valid_centroids, detected_lines = find_lines_that_point_to_centroids(detected_lines, stats, centroids)

    #visualize_results(original_image, valid_centroids, intersecting_lines, info="after detection")

    intersecting_lines, detected_lines= combine_cluster_lines_targeted(original_image, detected_lines, intersecting_lines)

    #visualize_results(original_image, valid_centroids, intersecting_lines, info="after clustering")

    graph = match_lines(valid_centroids, intersecting_lines)

    #visualize_simple_graph(original_image, graph)

    graph = backtrack_lines(original_image, detected_lines, graph, threshold = 10)

    visualize_simple_graph(original_image, graph)
    return graph


def find_structure_boxes(image_path):
    # Lade Bild
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Bild nicht gefunden: {image_path}")

    # Wandle BGR -> RGB (DECIMER erwartet RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Strukturen segmentieren
    segments = segment_chemical_structures(image_rgb)

    # Extrahiere Bounding Boxes
    bounding_boxes = []
    for segment in segments:
        x, y, w, h = segment['bbox']
        bounding_boxes.append((x, y, x + w, y + h))  # (x1, y1, x2, y2)

    return bounding_boxes


if __name__ == "__main__":
    # Image path


    for i in range(1, 14):
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
        get_result(image_path, mask_path)



