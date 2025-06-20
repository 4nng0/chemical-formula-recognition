import numpy as np
import cv2
import matplotlib.pyplot as plt
from reaction_graph import Graph


def visualize_simple_graph(image, graph, info=None, path=None):
    """
    Visualizes a graph on a grayscale image by drawing edges and nodes.

    Args:
        image (np.ndarray): Grayscale input image.
        graph (Graph): Custom graph object with nodes and edges.
        info (str, optional): Title for the plot.
        path (str, optional): If provided, saves the visualization as a PNG.
    """
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Draw edges as red lines
    for v, u in graph.edges():
        position1 = graph.nodes[v].position
        position2 = graph.nodes[u].position
        cv2.line(output_image, position1, position2, (0, 0, 255), 4)

    # Mark centroids (green) and line nodes (blue)
    for node in graph.nodes.values():
        if node.typ == "centroid":
            cv2.circle(output_image, (int(node.position[0]), int(node.position[1])), 10, (0, 255, 0), -1)
        elif node.typ == "line":
            cv2.circle(output_image, (int(node.position[0]), int(node.position[1])), 5, (255, 0, 0), -1)

    plt.figure(figsize=(8, 8))
    plt.title(info if info else "Graph Visualization")
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    if path is not None:
        plt.savefig(f"{path}.png")


def display_image_with_mask(image_path, binary_mask):
    """
    Displays an image with a binary mask overlay highlighting detected features.

    Args:
        image_path (str): Path to the grayscale image.
        binary_mask (np.ndarray): Binary mask to overlay (same size as image).

    Returns:
        matplotlib.pyplot: The plot object displaying the overlay.
    """
    overlay = np.zeros((*binary_mask.shape, 3), dtype=np.uint8)
    overlay[binary_mask > 0] = [255, 30, 0]  # red overlay

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Overlay the mask on the image
    overlayed_image = image_rgb.copy()
    overlayed_image[binary_mask > 0] = overlay[binary_mask > 0]

    plt.imshow(overlayed_image)
    plt.title("Arrow Heads Detected with Binary Mask Overlay")
    plt.axis("off")
    plt.show()
    return plt


def visualize_results(image, centroids, intersecting_lines, info=None, show=False, path=None):
    """
    Visualizes centroids and their connecting intersecting lines on the image.

    Args:
        image (np.ndarray): Grayscale image.
        centroids (list): List of centroid coordinates (x, y).
        intersecting_lines (list): List of lists of points representing lines.
        info (str, optional): Title for the plot.
        show (bool, optional): If True, prints line point lists.
        path (str, optional): If provided, saves the visualization as a PNG.

    Returns:
        np.ndarray: The image with drawn centroids and lines in color.
    """
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Draw centroids as green circles
    for cx, cy in centroids[1:]:
        cv2.circle(output_image, (int(cx), int(cy)), 10, (0, 255, 0), -1)

    # Draw intersecting lines in red
    for line_list in intersecting_lines:
        if show:
            print(line_list)
        start = (int(line_list[0][0]), int(line_list[0][1]))
        for i in range(1, len(line_list)):
            next_point = (int(line_list[i][0]), int(line_list[i][1]))
            cv2.line(output_image, start, next_point, color=(0, 0, 255), thickness=2)
            start = next_point

    plt.figure(figsize=(8, 8))
    plt.title(info if info else "Centroids and Intersecting Lines")
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    if path is not None:
        plt.savefig(f"{path}.png")
    return output_image
