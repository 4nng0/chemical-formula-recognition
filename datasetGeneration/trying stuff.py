import numpy as np
import cv2
from datasetGeneration.generation import create_random_box

if __name__ == "__main__":
    for k in range(10):

        # List of image paths
        image_paths = []
        nbMolecules = np.random.randint(5, 15)
        for j in range(nbMolecules):
            #print("b")
            i = np.random.randint(0, 40000)
            image_paths.append(f"chemicalStructureSource/{i}.png")

        # Dimensions of the image
        collage_width = 1500
        collage_height = 1500

        print(image_paths)

        # make the image
        # try to create a collage, if it returns an error, try again

        collage, masque = create_random_box(image_paths, collage_width, collage_height)
        """
        while True:
            try:
                collage, masque = create_random_box(image_paths, collage_width, collage_height)
                break
            except Exception as e:
                print(f"Error making the picture: {e}")
                break
                """  #continue

        # Save collage
        cv2.imwrite(f'data/images/{k}.jpg', collage)
        cv2.imwrite(f'data/masks/{k}.jpg', masque)
