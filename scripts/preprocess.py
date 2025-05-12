import numpy as np
import cv2
import os

def load_data(image_dir, mask_dir, img_size=(512, 512)):
    images = []
    masks = []
    for filename in os.listdir(image_dir):
        if (filename != '.DS_Store'):
            print(filename)
            # Reading images and masks
            image = cv2.imread(os.path.join(image_dir, filename), cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(os.path.join(mask_dir, filename), cv2.IMREAD_GRAYSCALE)
            
            # Applying thresholding to obtain binary masks
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            
            # Resizing images and masks
            image = cv2.resize(image, img_size)
            mask = cv2.resize(mask, img_size)
            images.append(image)
            masks.append(mask)

            # Uncomment these lines to visualize images and masks during debugging
            # cv2.imshow('image', image)
            # cv2.imshow('mask', mask)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
    
    # Normalizing images
    images = np.array(images).astype('float32') / 255.0 
    masks = np.array(masks).astype('float32') / 255.0
    
    # Adding an additional dimension to make images compatible with the model
    images = np.expand_dims(images, axis=-1)
    masks = np.expand_dims(masks, axis=-1)

    return images, masks

if __name__ == "__main__": 
    
    script_path = os.getcwd()
    base_path = os.path.dirname(script_path)
    image_dir = os.path.join(base_path, 'datasetGeneration/data/images')
    mask_dir = os.path.join(base_path, 'datasetGeneration/data/masks')
    images, masks = load_data(image_dir, mask_dir)
    
    # Saving preprocessed images and masks into numpy files
    np.save(os.path.join(base_path, 'data/preprocessed_images_512.npy'), images)
    np.save(os.path.join(base_path, 'data/preprocessed_masks_512.npy'), masks)

"""
# Optional: Splitting the dataset into smaller chunks for easier handling
num_files = 20
split_images = np.array_split(images, num_files)
split_masks = np.array_split(masks, num_files)

for i, split in enumerate(split_images):
    np.save(f'data/preprocessed_images_part_{i}.npy', split)
    np.save(f'data/preprocessed_masks_part_{i}.npy', split_masks[i])
"""