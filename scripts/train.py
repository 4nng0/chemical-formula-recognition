import numpy as np
import tensorflow as tf
from unetModel import *  # Import the U-Net model definition
import os

def load_preprocessed_data():
    """
    Loads preprocessed image and mask datasets.
    
    Returns:
        tuple: A tuple containing the preprocessed images and masks as NumPy arrays.
    """
    script_path = os.getcwd()
    base_path = os.path.dirname(script_path)
    images = np.load(os.path.join(base_path, 'data/preprocessed_images_512.npy'))
    masks = np.load(os.path.join(base_path, 'data/preprocessed_masks_512.npy'))
    
    return images, masks

if __name__ == "__main__":
    print("Training U-Net model for arrow detection...")

    # Load preprocessed data
    images, masks = load_preprocessed_data()
    images = tf.convert_to_tensor(images)
    masks = tf.convert_to_tensor(masks)
    
    # Verify the dimensions of the data
    assert images.shape[1:] == (512, 512, 1), f"Expected image shape (512, 512, 1), but got {images.shape[1:]}"
    assert masks.shape[1:] == (512, 512, 1), f"Expected mask shape (512, 512, 1), but got {masks.shape[1:]}"

    print("Data loaded successfully!")
    
    # Create the U-Net model
    model = unet(input_size=(512, 512, 1))

    print("Model created successfully!")
    
    # Display the model summary to verify the architecture
    model.summary()
    
    # Train the model
    model.fit(images, masks, batch_size=32, epochs=50, validation_split=0.1)
    
    # Save the trained model
    model.save(os.path.join(os.getcwd(), 'saved_models/unet_model_512.keras'))