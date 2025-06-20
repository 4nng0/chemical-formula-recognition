import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import hausdorff_distance


def reflect_points(points, center, direction):
    """
    Reflects 2D points across a line defined by a center point and a direction vector.

    Parameters:
        points (ndarray): Array of shape (N, 2) containing 2D points.
        center (ndarray): The point through which the reflection axis passes.
        direction (ndarray): Direction vector of the reflection axis.

    Returns:
        ndarray: Reflected 2D points, same shape as input.
    """
    direction = direction / np.linalg.norm(direction)
    vecs = points - center
    proj = np.dot(vecs, direction[:, None]) * direction
    reflected = center + proj - (vecs - proj)
    return reflected

def points_to_mask(points, shape):
    """
    Converts a set of (y, x) coordinates into a binary mask.

    Parameters:
        points (ndarray): Array of shape (N, 2) with (y, x) pixel coordinates.
        shape (tuple): Shape of the mask (height, width).

    Returns:
        ndarray: Binary mask with ones at point locations.
    """
    mask = np.zeros(shape, dtype=np.uint8)
    for y, x in np.round(points).astype(int):
        if 0 <= y < shape[0] and 0 <= x < shape[1]:
            mask[y, x] = 1
    return mask

def compute_best_symmetry_axes(binary_mask, show=False):
    """
    Finds the best symmetry axis for each connected component in a binary mask
    by minimizing the Hausdorff distance between the original and reflected shapes.

    Parameters:
        binary_mask (ndarray): Binary image (0/1 or 0/255) with foreground objects.
        show (bool): If True, visualizes the best axes on the image.

    Returns:
        tuple: (num_labels, labels, stats, centroids, axes)
            - num_labels: Number of connected components + background
            - labels: Label image of same shape as input
            - stats: Component bounding boxes and areas
            - centroids: Component centroids
            - axes: Dictionary mapping label -> dict with 'center', 'axis', 'hausdorff'
    """
    axes = {}
    details_show = False

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask)
    for label in range(1, num_labels):
        component_mask = (labels == label).astype(np.uint8)
        points = np.column_stack(np.where(component_mask > 0))
        if len(points) < 5:
            continue

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
                original_mask = original_mask.astype(np.float32)
                mask = mask.astype(np.float32)

                overlay = np.zeros((*original_mask.shape, 3), dtype=np.float32)

                overlay[..., 0] = original_mask  # Rotkanal
                overlay[..., 2] = mask  # Blaukanal

                overlay = np.clip(overlay, 0, 1)

                plt.figure(figsize=(6, 6))
                plt.imshow(overlay)
                plt.title(f'Overlay: Red = Original, Blue = Reflected, angle = {angle}')
                plt.axis('off')
                plt.tight_layout()
                plt.show()

            d = hausdorff_distance(mask, original_mask)

            if d < best_score:
                best_score = d
                best_info = {
                    'center': tuple(mean[::-1]),
                    'axis': tuple(direction[::-1]),
                    'hausdorff': d
                }

        if best_info:
            axes[label] = best_info

    if show:
        plt.figure(figsize=(8, 8))
        plt.imshow(binary_mask, cmap='gray')
        for info in axes.values():
            cx, cy = info['center']
            dx, dy = info['axis']
            plt.arrow(cx - dx * 50, cy - dy * 50, dx * 100, dy * 100, color='red', head_width=5)
        plt.title("Best Symmetrieachsen (Hausdorff)")
        plt.axis('off')
        plt.show()

    return num_labels, labels, stats, centroids, axes

def get_distance_norm_with_axis(line, centroid, axis):
    """
    Measures how close and aligned a line is to a symmetry axis.

    Parameters:
        line (tuple): ((x1, y1), (x2, y2)) line endpoints.
        centroid (tuple): (cx, cy) center point to measure distance from.
        axis (dict): Dictionary with key 'axis' → direction vector.

    Returns:
        float: A score based on distance and angular deviation from the axis.
               Lower is better (more aligned and close).
    """
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