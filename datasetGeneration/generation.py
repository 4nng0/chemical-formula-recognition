import os

import cv2
import numpy as np

formules = [
    "BH3",
    "BH3NH3",
    "Bi2Te3",
    "Bi4Ge3O12",
    "Bi12GeO20",
    "BN",
    "BNH6",
    "Br2",
    "C2Ag2O4",
    "C2Cl2O2",
    "C2Cl3O2",
    "C2Cl4",
    "C2ClF5",
    "C2H2",
    "C2H2AsCl3",
    "C2H2Cl2O2",
    "C2H3AgO2",
    "C2H3Cl",
    "C2H3Cl2F",
    "C2H3Cl3",
    "C2H3Cl3O2",
    "C2H3ClO",
    "C2H3F3",
    "C2H3LiO2",
    "C2H3NaO2",
    "C2H3NO5",
    "C2H4Cl2O",
    "C2H4O2S",
    "C2H5Cl",
    "C2H5NaO",
    "C2H5NaO3S2",
    "C2H6",
    "C2H6ClO3P",
    "C2H6N2O",
    "C2H6O2",
    "C2H7AsO2",
    "C2H7NO2",
    "C2H7NO2S",
    "C2H7NO3S",
    "C2H7NS",
    "C2H7O4P",
    "C2H8Si",
    "C2HAg",
    "C2HBrClF3",
    "C2HCl3",
    "C2HCl3O2",
    "C2HF3O2",
    "C2HF5",
    "C2N2",
    "C3Cl2N3NaO3",
    "C3F8",
    "C3H3Cl",
    "C3H3N3O3",
    "C3H3NS",
    "C3H4Cl2F2O",
    "C3H4Cl2O2",
    "C3H4N2O2",
    "C3H5Cl",
    "C3H5O6P",
    "C3H6N6",
    "C3H6N6O6",
    "C3H6O2S",
    "C3H7NO2S",
    "C3H7NO2Se",
    "C3H7NO3",
    "C3H7O4P",
    "C3H8",
    "C3H8NO5P",
    "C3H8O3S3",
    "C3H8OS2",
    "C3H8S2",
    "C3H9As",
    "C3H9Ga",
    "C3H9NO",
    "C3H10Si",
    "C4H2O3",
    "C4H3F7O",
    "C4H3FN2O2",
    "C4H4BrNO2",
    "C4H4ClNO2",
    "C4H4KNaO6",
    "C4H4KNO4S",
    "C4H4N2O3",
    "C4H4N2O5",
    "C4H4O5",
    "C4H4S",
    "C4H5KO6",
    "C4H5N",
    "C4H6CaO4",
    "C4H6CuO4",
    "C4H6N2S4Zn",
    "C4H6N4O3",
    "C4H6N4O3S2",
    "C4H6O4Pb",
    "C4H6O4S2",
    "C4H6O6",
    "C4H7Br2Cl2O4P",
    "C4H7Cl2O4P",
    "C4H7N3O",
    "C4H8Cl2S",
    "C4H8Cl3O4P",
    "C4H9N",
    "C4H9N3O2",
    "C4H9NO2S",
    "C4H10FO2P"
]

directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

transformations = {
    (0, 1): 0,  # Droite
    (1, 0): 270,  # Bas
    (0, -1): 180,  # Gauche
    (-1, 0): 90,  # Haut
    (1, 1): 315,  # Bas-Droite
    (1, -1): 225,  # Bas-Gauche
    (-1, 1): 45,  # Haut-Droite
    (-1, -1): 135  # Haut-Gauche
}

# TODO LIST
# different arrow styles
# different style of chemical formulas
# text at all different places and in diffent styles
# random bits that are not arrows
# plus where arrows could be
# arrows coming together
# funnel thing

# DONE
# more complex boxes
#

def draw_arrow(collage_image, collage_masque, numFleche, angle, e, yi, xi, hi, wi, direction):
    # load the random arrow in the right direction
    defautAngle = 0
    if np.random.randint(0, 2) == 1:  # you have a 50/50 chance of changing the angle of the arrow
        defautAngle = np.random.randint(-11, 11)
    angleImage = (angle + defautAngle) % 360
    fleche = cv2.imread(f"arrowSource/{numFleche}_{angleImage}.png", cv2.IMREAD_UNCHANGED)
    masque = cv2.imread(f"arrowMask/{numFleche}_{angleImage}.png", cv2.IMREAD_UNCHANGED)
    hOrigin, wOrigin = fleche.shape[:2]

    # leave a margin to prevent the arrow from sticking
    m = np.random.randint(10, 20)
    if 2 * m > e:
        m = 10

    # resize the arrow to find the coordinates of its upper left and lower right corners
    if directions[direction][0] == 1:
        yf1 = yi + hi + m
        yf2 = yi + hi + e - m
        fleche = cv2.resize(fleche, (fleche.shape[1], e - 2 * m))
        masque = cv2.resize(masque, (fleche.shape[1], e - 2 * m))
    if directions[direction][0] == -1:
        yf1 = yi - e + m
        yf2 = yi - m
        fleche = cv2.resize(fleche, (fleche.shape[1], e - 2 * m))
        masque = cv2.resize(masque, (fleche.shape[1], e - 2 * m))
    if directions[direction][1] == 1:
        xf1 = xi + wi + m
        xf2 = xi + wi + e - m
        fleche = cv2.resize(fleche, (e - 2 * m, fleche.shape[0]))
        masque = cv2.resize(masque, (e - 2 * m, fleche.shape[0]))
    if directions[direction][1] == -1:
        xf1 = xi - e + m
        xf2 = xi - m
        fleche = cv2.resize(fleche, (e - 2 * m, fleche.shape[0]))
        masque = cv2.resize(masque, (e - 2 * m, fleche.shape[0]))
    if directions[direction][0] == 0:
        fleche = cv2.resize(fleche, (fleche.shape[1], (hOrigin * fleche.shape[1]) // wOrigin))
        masque = cv2.resize(masque, (fleche.shape[1], (hOrigin * fleche.shape[1]) // wOrigin))
        yf1 = yi + (hi - fleche.shape[0]) // 2
        yf2 = yi + (hi + fleche.shape[0]) // 2
    if directions[direction][1] == 0:
        fleche = cv2.resize(fleche, ((wOrigin * fleche.shape[0]) // hOrigin, fleche.shape[0]))
        masque = cv2.resize(masque, ((wOrigin * fleche.shape[0]) // hOrigin, fleche.shape[0]))
        xf1 = xi + (wi - fleche.shape[1]) // 2
        xf2 = xi + (wi + fleche.shape[1]) // 2

    # draw the arrow on the collage considering the opacity of the image

    flecheBgr = fleche[:, :, 0:3]
    flecheAlpha = fleche[:, :, 3]

    for c in range(3):
        collage_image[yf1:yf2, xf1:xf2, c] = (1 - flecheAlpha / 255.0) * collage_image[yf1:yf2, xf1:xf2, c] + (
                flecheAlpha / 255.0) * flecheBgr[:, :, c]

    # same for the mask

    masqueBgr = masque[:, :, 0:3]
    masqueAlpha = masque[:, :, 3]

    for c in range(3):
        collage_masque[yf1:yf2, xf1:xf2, c] = (1 - masqueAlpha / 255.0) * collage_masque[yf1:yf2, xf1:xf2,
                                                                          c] + (
                                                      masqueAlpha / 255.0) * masqueBgr[:, :, c]


def draw_frames(collage_image, x, y, img_width, img_height):
    option = np.random.choice(["solid_box", "dotted_box", "parentheses"])  # all are same probability
    padding = 50
    color = (0, 0, 0)
    thickness = np.random.randint(1, 5)

    top_left = (x - padding, y - padding)
    bottom_right = (x + img_width + padding, y + img_height + padding)

    if option == "solid_box":
        # draw frame
        cv2.rectangle(collage_image, top_left, bottom_right, color, thickness)

    elif option == "dotted_box":
        # draw dotted frame
        dash_length = np.random.randint(5, 25)
        for i in range(top_left[0], bottom_right[0], dash_length * 2):
            cv2.line(collage_image, (i, top_left[1]), (i + dash_length, top_left[1]), color, thickness)
            cv2.line(collage_image, (i, bottom_right[1]), (i + dash_length, bottom_right[1]), color, thickness)
        for i in range(top_left[1], bottom_right[1], dash_length * 2):
            cv2.line(collage_image, (top_left[0], i), (top_left[0], i + dash_length), color, thickness)
            cv2.line(collage_image, (bottom_right[0], i), (bottom_right[0], i + dash_length), color, thickness)

    elif option == "parentheses":
        # draw parentheses
        length = np.random.randint(20, 45)  # Länge der Klammern

        # left side
        cv2.line(collage_image, top_left, (top_left[0] + length, top_left[1]), color, thickness)
        cv2.line(collage_image, top_left, (top_left[0], bottom_right[1]), color, thickness)
        cv2.line(collage_image, (top_left[0], bottom_right[1]), (top_left[0] + length, bottom_right[1]), color,
                 thickness)

        # right side
        cv2.line(collage_image, (bottom_right[0], top_left[1]), (bottom_right[0] - length, top_left[1]), color,
                 thickness)
        cv2.line(collage_image, (bottom_right[0], top_left[1]), (bottom_right[0], bottom_right[1]), color,
                 thickness)
        cv2.line(collage_image, bottom_right, (bottom_right[0] - length, bottom_right[1]), color, thickness)


def draw_plus(collage_image, yi, xi, hi, wi, direction, m):
    pass
    my, mx = round( yi + 0.5 * hi) , round(xi + 0.5 * wi)
    font = np.random.randint(1, 7)
    font_scale = np.random.randint(5, 15) / 10

    cv2.putText(collage_image, "+", (mx, my), font, font_scale, (0, 0, 0), 3, cv2.LINE_AA)

    pass


def create_random_box(image_paths, collage_width, collage_height):
    import numpy as np
    import cv2
    print(collage_height, collage_width)
    print(image_paths)
    print(os.path.exists(image_paths[0]))
    # Create a blank white canvas for the collage
    collage_image = np.zeros((collage_height, collage_width, 3), dtype=np.uint8)
    collage_image.fill(255)  # Remplir l'image avec du blanc

    # Create a mask with the same size
    collage_masque = np.zeros((collage_height, collage_width, 3), dtype=np.uint8)
    collage_masque.fill(255)

    # Select a random arrow
    numFleche = np.random.randint(0, 7)

    # Charger et placer chaque image aux positions spécifiées
    n = 0  # Counter for the number of images placed
    positions = []  # Stores image positions as [x, y, img_width, img_height] with x, y left lower corner position
    for image_path in image_paths:
        image = cv2.imread(image_path)

        # Crop the image to remove white borders
        for i in range(image.shape[0]):
            if np.any(image[i] != 255):
                image = image[i:]
                break
        for i in range(image.shape[0] - 1, 0, -1):
            if np.any(image[i] != 255):
                image = image[:i]
                break
        for i in range(image.shape[1]):
            if np.any(image[:, i] != 255):
                image = image[:, i:]
                break
        for i in range(image.shape[1] - 1, 0, -1):
            if np.any(image[:, i] != 255):
                image = image[:, :i]
                break

        # 50% chance to add a text label under the image
        if np.random.randint(0, 2) == 1:
            # enlarges the lower part of the image by 40 pixels
            image = np.concatenate((image, np.zeros((30, image.shape[1], 3), dtype=np.uint8) + 255), axis=0)

            # add text to the image
            font = np.random.randint(1, 7)
            font_scale = np.random.randint(5, 15) /10
            cv2.putText(image, formules[np.random.randint(0, len(formules))] + " (m/z = " + str(
                np.random.randint(2, 300)) + ")", (10, image.shape[0] - 5), font, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

        img_height, img_width, _ = image.shape

        angle = 0
        direction = 0
        e = 0
        xi, yi, wi, hi = 0, 0, 0, 0

        # Place the first image at a random position
        if n == 0:
            # x and y are the left lower corner
            x, y = np.random.randint(0, collage_width - img_width), np.random.randint(0, collage_height - img_height)
            positions.append([x, y, img_width, img_height])
        else:
            # Generate possible positions based on previous images
            positionsPossibles = [] # [x1, y1, j, espacement] this is what elements look like
            for i in range(n):
                positionsPossibles.append([])
                for j in range(len(directions)):
                    # look in all random directions to see if there is space
                    buffer_between_pictures = np.random.randint(100, 150)
                    # location of the area is tested
                    if directions[j][1] == 1:
                        x1 = positions[i][0] + positions[i][2] + buffer_between_pictures
                    elif directions[j][1] == -1:
                        x1 = positions[i][0] - img_width - buffer_between_pictures
                    else:
                        x1 = positions[i][0]
                    if directions[j][0] == 1:
                        y1 = positions[i][1] + positions[i][3] + buffer_between_pictures
                    elif directions[j][0] == -1:
                        y1 = positions[i][1] - img_height - buffer_between_pictures
                    else:
                        y1 = positions[i][1]

                    # if the zone is in the collage
                    if x1 >= 0 and x1 + img_width <= collage_width and y1 >= 0 and y1 + img_height <= collage_height:
                        # check if the zone is white
                        zone = collage_image[y1:y1 + img_height, x1:x1 + img_width]
                        if np.all(zone == 255):
                            positionsPossibles[i].append([x1, y1, j, buffer_between_pictures])

            # if no zone is possible, move on to the next image
            possible = False
            for k in range(len(positionsPossibles)):
                if len(positionsPossibles[k]) > 0:
                    possible = True

            if not possible:
                print(f"L'image {image_path} ne peut pas être placée dans le collage.")
                continue

            # select a random zone from the list of possible zones
            zone = np.random.randint(0, len(positionsPossibles))
            while len(positionsPossibles[zone]) == 0:
                zone = np.random.randint(0, len(positionsPossibles))
            x, y, direction, e = positionsPossibles[zone][np.random.randint(0, len(positionsPossibles[zone]))]
            xi, yi, wi, hi = positions[zone]
            positions.append([x, y, img_width, img_height])

            angle = transformations[directions[direction]]

        # We calculate the size of the image to be pasted. In theory, this shouldn't change anything, as we've made sure that the image can be pasted in its entirety.
        if x + img_width > collage_width:
            img_width = collage_width - x
        if y + img_height > collage_height:
            img_height = collage_height - y

        # in theory, these cases shouldn't happen
        if img_width <= 0 or img_height <= 0:
            print(
                f"L'image {image_path} ne peut pas être placée aux coordonnées ({x}, {y}) car elle dépasse les limites du collage.")
            continue

        # image = cv2.resize(image, (img_width, img_height))

        # crop image to fit collage size
        image = image[:img_height, :img_width]

        # Paste image into collage
        collage_image[y:y + img_height, x:x + img_width] = image


        # draw arrow or plus between the pictures
        if n > 0:
            # 10% chance of drawing a arrow
            if np.random.randint(0, 10) == 0:

                draw_plus(collage_image, yi, xi, hi, wi, direction, e)
            else :
                draw_arrow(collage_image, collage_masque, numFleche, angle, e, yi, xi, hi, wi, direction)

        # draw a mark arount the picture
        if np.random.randint(0, 10) == 0:
            draw_frames(collage_image, x, y, img_width, img_height)

        n += 1

    # Remove outer white borders
    for i in range(collage_image.shape[0]):
        if np.any(collage_image[i] != 255):
            collage_image = collage_image[i:]
            collage_masque = collage_masque[i:]
            break
    for i in range(collage_image.shape[0] - 1, 0, -1):
        if np.any(collage_image[i] != 255):
            collage_image = collage_image[:i]
            collage_masque = collage_masque[:i]
            break
    for i in range(collage_image.shape[1]):
        if np.any(collage_image[:, i] != 255):
            collage_image = collage_image[:, i:]
            collage_masque = collage_masque[:, i:]
            break
    for i in range(collage_image.shape[1] - 1, 0, -1):
        if np.any(collage_image[:, i] != 255):
            collage_image = collage_image[:, :i]
            collage_masque = collage_masque[:, :i]
            break

    # Add a 50px margin
    collage_image = np.concatenate((np.zeros((50, collage_image.shape[1], 3), dtype=np.uint8) + 255, collage_image,
                                    np.zeros((50, collage_image.shape[1], 3), dtype=np.uint8) + 255), axis=0)
    collage_image = np.concatenate((np.zeros((collage_image.shape[0], 50, 3), dtype=np.uint8) + 255, collage_image,
                                    np.zeros((collage_image.shape[0], 50, 3), dtype=np.uint8) + 255), axis=1)
    collage_masque = np.concatenate((np.zeros((50, collage_masque.shape[1], 3), dtype=np.uint8) + 255, collage_masque,
                                     np.zeros((50, collage_masque.shape[1], 3), dtype=np.uint8) + 255), axis=0)
    collage_masque = np.concatenate((np.zeros((collage_masque.shape[0], 50, 3), dtype=np.uint8) + 255, collage_masque,
                                     np.zeros((collage_masque.shape[0], 50, 3), dtype=np.uint8) + 255), axis=1)

    # Convert mask to binary (black/white)
    collage_masque = cv2.cvtColor(collage_masque, cv2.COLOR_BGR2GRAY)
    _, collage_masque = cv2.threshold(collage_masque, 127, 255, cv2.THRESH_BINARY)
    # Invert the mask
    collage_masque = 255 - collage_masque

    return collage_image, collage_masque


"""
#creates a dataset of 1000 images
for k in range(1000):

    # List of image paths
    image_paths = []
    nbMolecules = np.random.randint(5, 15)
    for j in range(nbMolecules):
        i=np.random.randint(0, 40000)
        image_paths.append(f"datasetGeneration/chemicalStructureSource/{i}.png")

    # Dimensions of the image 
    collage_width = 1500
    collage_height = 1500


    # make the image 
    # try to create a collage, if it returns an error, try again
    while True:
        try:
            collage, masque = create_random_box(image_paths, collage_width, collage_height)
            break
        except:
            print("Erreur lors de la création du collage, on recommence.")
            continue

    # Save collage
    cv2.imwrite(f'data/images/{k}.jpg', collage)
    cv2.imwrite(f'data/masks/{k}.jpg', masque)
"""
"""

image_paths = []
nbMolecules = np.random.randint(5, 15)
for j in range(nbMolecules):
    i = np.random.randint(0, 40000)
     image_paths.append(f"datasetGeneration/chemicalStructureSource/{i}.png")

collage_width, collage_height = 1500, 1500

collage, masque = create_random_box(image_paths, collage_width, collage_height)

# cv2.imwrite(f'image.jpg', collage)
# cv2.imwrite(f'masque.jpg', masque)

"""
