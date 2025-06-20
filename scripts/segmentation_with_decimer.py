import cv2
import decimer_segmentation as ds
import os
import matplotlib.pyplot as plt


def find_structure_boxes(image_path):
    """
    Loads an image and detects structure bounding boxes using Mask R-CNN.

    Args:
        image_path (str): Path to the image file.

    Returns:
        list of tuples: List of bounding boxes, each as (y1, x1, y2, x2).

    Raises:
        FileNotFoundError: If the image cannot be loaded.
    """
    # Load image from disk
    image = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if image is None:
        raise FileNotFoundError(f"image not found: {image_path}")

    # Convert image from BGR (OpenCV default) to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply Mask R-CNN model to detect masks, bounding boxes, and scores
    masks, bboxes, scores = ds.get_mrcnn_results(image_rgb)

    return bboxes


if __name__ == "__main__":
    i = 1

    # Build the path to the image
    script_path = os.getcwd()
    base_path = os.path.dirname(script_path)
    test_images = 'test/realPictures'
    image_path = os.path.join(base_path, test_images, f"{i}.jpg")

    # Detect structure bounding boxes
    boxes = find_structure_boxes(image_path)

    # Load the image again for visualization
    image = cv2.imread(image_path)

    # Create a matplotlib figure
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap='gray')

    # Draw each bounding box as a red rectangle on the image
    for (y1, x1, y2, x2) in boxes:
        plt.gca().add_patch(
            plt.Rectangle(
                (x1, y1),              # Rectangle start point (top-left)
                x2 - x1,               # Width
                y2 - y1,               # Height
                edgecolor='red',       # Red border
                facecolor='none',      # Transparent fill
                linewidth=2            # Border thickness
            )
        )

    plt.axis('off')
    plt.show()