import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
from scripts import visualisation

# Enable GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Enabled GPU memory growth.")
    except RuntimeError as e:
        print(f"Error enabling GPU memory growth: {e}")
        


def arrow_heads(image_path, model_path):
    print(image_path)
    model = tf.keras.models.load_model(model_path)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image not found at path: {image_path}")


    # Increase contrast
    image = cv2.convertScaleAbs(image, alpha=1.3, beta=0)

    print("Image loaded successfully. Resizing...")
    image_resized = cv2.resize(image, (512, 512))
    print("Image resized successfully!") 
    image_array = np.expand_dims(image_resized, axis=[0, -1]) / 255.0

    # Prediction
    print(type(model))
    print(dir(model))
    try:
        prediction = model.predict(image_array)[0, :, :, 0]
        print("Prediction completed successfully!")
    except Exception as e:
        print(f"Error during prediction: {e}")
    prediction_resized = cv2.resize(prediction, (image.shape[1], image.shape[0]))

   # Create binary mask
    threshold = 0.4
    binary_mask = (prediction_resized > threshold).astype(np.uint8)

    return binary_mask



if __name__ == "__main__":
    # Specify a test image path
    script_path = os.getcwd()
    base_path = os.path.dirname(script_path)
    test_image_dir = 'test_images'
    image_name = '2.jpg'
    model_name = 'saved_models/unet_model_512_version_1.keras'
    image_path = os.path.join(base_path, test_image_dir, image_name)
    model_path = os.path.join(base_path, model_name)
    
    # Process image
    
    # Call detect_arrow_heads function
    arrow_heads = arrow_heads(image_path, model_path)
    
    # Check the output
    #print("Arrow heads detected at positions:", arrow_heads)
    
    visualisation.display_image_with_mask(image_path, arrow_heads)

