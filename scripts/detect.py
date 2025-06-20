import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

from torchvision.transforms import transforms
from scripts import visualisation
import matplotlib.pyplot as plt
import torch
from scripts.remove_text import remove_text
from unet import unet


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Enabled GPU memory growth.")
    except RuntimeError as e:
        print(f"Error enabling GPU memory growth: {e}")


def arrow_heads_py(image_path, model_path, device='cpu'):
    """
    Detects arrow heads in an image using a PyTorch UNet model.

    Args:
        image_path (str): Path to the input image.
        model_path (str): Path to the saved PyTorch model (.pth file).
        device (str): Device to run inference on ('cpu' or 'cuda').

    Returns:
        np.ndarray: Binary mask of predicted arrow head locations.
    """
    gs = transforms.Grayscale()

    # Load PyTorch UNet model
    model = unet()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    no_text = False  # Toggle for using pre-processed image without text

    if no_text:
        image = remove_text(image_path, None)
    else:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image not found at path: {image_path}")

    # Resize and binarize image for model input
    image_resized = cv2.resize(image, (1024, 1024))
    image_resized = (image_resized > 127) * 255.0  # Binarization

    # Convert to tensor and format for grayscale input
    image_tensor = torch.tensor(image_resized, dtype=torch.float32)
    image_tensor = image_tensor.squeeze()  # Remove redundant dimension
    image_tensor = image_tensor.permute(2, 0, 1)  # Rearrange to [C, H, W]
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    image_tensor = gs(image_tensor)  # Convert to grayscale

    image_tensor = image_tensor.to(device)

    try:
        with torch.no_grad():
            prediction = model(image_tensor)  # Forward pass
            prediction = prediction.squeeze().cpu().numpy()
        print("Prediction completed successfully!")
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

    # Resize prediction back to original image size
    prediction_resized = cv2.resize(prediction, (image.shape[1], image.shape[0]))

    # Threshold to obtain binary mask
    threshold = 0.2
    binary_mask = (prediction_resized > threshold).astype(np.uint8)

    return binary_mask


def arrow_heads(image_path, model_path):
    """
    Detects arrow heads in an image using a TensorFlow Keras model.

    Args:
        image_path (str): Path to the input image.
        model_path (str): Path to the saved Keras model (.keras file).

    Returns:
        np.ndarray: Binary mask of predicted arrow head locations.
    """
    # Load Keras model
    model = tf.keras.models.load_model(model_path)

    no_text = False
    if no_text:
        image = remove_text(image_path, None)
    else:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Image not found at path: {image_path}")

    # Enhance contrast slightly for better predictions
    image = cv2.convertScaleAbs(image, alpha=1.3, beta=0)

    print("Image loaded successfully. Resizing...")
    image_resized = cv2.resize(image, (512, 512))
    print("Image resized successfully!")

    # Prepare image for model input
    image_array = np.expand_dims(image_resized, axis=[0, -1]) / 255.0  # Normalize and add batch/channel dims

    try:
        prediction = model.predict(image_array)[0, :, :, 0]  # Forward pass
        print("Prediction completed successfully!")
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

    # Resize prediction to original image size
    prediction_resized = cv2.resize(prediction, (image.shape[1], image.shape[0]))

    # Threshold to create binary mask
    threshold = 0.4
    binary_mask = (prediction_resized > threshold).astype(np.uint8)

    return binary_mask


if __name__ == "__main__":
    # Example usage: define paths and visualize result

    script_path = os.getcwd()
    base_path = os.path.dirname(script_path)

    test_image_dir = 'test/realPictures'
    image_name = '13.jpg'
    model_name = "scripts/seg_model_ng3.pth"  # Change to .keras for TensorFlow

    image_path = os.path.join(base_path, test_image_dir, image_name)
    model_path = os.path.join(base_path, model_name)

    # Call PyTorch-based arrow head detector
    arrow_heads = arrow_heads_py(image_path, model_path)

    # Visualize prediction mask
    plt.imshow(arrow_heads)
    plt.title("Predicted Arrow Head Mask")
    plt.axis("off")
    plt.show()

    # Display image with overlayed prediction
    visualisation.display_image_with_mask(image_path, arrow_heads)


