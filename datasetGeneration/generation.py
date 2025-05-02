import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image#
import os

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
# arrows coming together
# funnel thing

# DONE
# more complex boxes
# plus where arrows could be
#

def text_to_image(text, font_path, font_size=40, text_color=(0, 0, 0)):
    font = ImageFont.truetype(font_path, font_size)

    # Textgröße berechnen mit getbbox
    bbox = font.getbbox(text)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]

    # Neues Bild in passender Größe
    img = Image.new("RGB", (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)

    # Text zeichnen (ggf. y-offset wegen negativer bbox)
    draw.text((-bbox[0], -bbox[1]), text, font=font, fill=text_color)

    return np.array(img)


def center_image_on_canvas(image, canvas_width, canvas_height):
    h, w = image.shape[:2]


    if h/w > canvas_height/canvas_width:
        # image is taller than canvas
        new_width = int(canvas_height * w / h)
        new_height = canvas_height
        hight_offset = 0
        width_offset = int((canvas_width - new_width) / 2)
    else:
        new_width = canvas_width
        new_height = int(canvas_width * h / w)
        hight_offset = int((canvas_height - new_height) / 2)
        width_offset = 0

    image = cv2.resize(image, (new_width, new_height))
    canvas =  np.ones((canvas_height, canvas_width, 4), dtype=np.uint8)


    canvas[hight_offset:hight_offset + new_height, width_offset:width_offset + new_width] = image

    return canvas

def draw_arrow(collage_image, collage_masque, x_between, y_between, h_between, w_between, direction, numFleche):
    # load the random arrow in the right direction
    angle = transformations[directions[direction]]  # gives the number of degrees depending on the direction
    defautAngle = 0
    if np.random.randint(0, 2) == 1:  # you have a 50/50 chance of changing the angle of the arrow
        defautAngle = np.random.randint(-11, 11)
    angleImage = (angle + defautAngle) % 360
    fleche = cv2.imread(f"arrowSource/{numFleche}_{angleImage}.png", cv2.IMREAD_UNCHANGED)
    masque = cv2.imread(f"arrowMask/{numFleche}_{angleImage}.png", cv2.IMREAD_UNCHANGED)


    # leave a margin to prevent the arrow from sticking
    y_margin = round(np.random.randint(10, 20)/100 * h_between)
    x_margin = round(np.random.randint(10, 20) / 100 * w_between)

    # the max is just in case anything goes wrong, wich shouldnt be the case
    new_width = max(1, w_between - 2 * x_margin)
    new_height = max(1, h_between - 2 * y_margin)

    fleche = center_image_on_canvas(fleche, new_width, new_height)
    masque = center_image_on_canvas(masque, new_width, new_height)


    # draw the arrow on the collage considering the opacity of the image

    flecheBgr = fleche[:, :, 0:3]
    flecheAlpha = fleche[:, :, 3]

    y_start = y_between + y_margin
    y_end = y_between + h_between -y_margin
    x_start = x_between + x_margin
    x_end = x_between + w_between -x_margin


    for c in range(3):
        collage_image[y_start:y_end, x_start:x_end, c] = (1 - flecheAlpha / 255.0) * collage_image[y_start:y_end, x_start:x_end, c] + (
                flecheAlpha / 255.0) * flecheBgr[:, :, c]

    # same for the mask

    masqueBgr = masque[:, :, 0:3]
    masqueAlpha = masque[:, :, 3]

    for c in range(3):
        collage_masque[y_start:y_end, x_start:x_end, c] = (1 - masqueAlpha / 255.0) * collage_masque[y_start:y_end, x_start:x_end,
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

def draw_plus(collage_image, x_between, y_between, h_between, w_between):
    font_scale = np.random.randint(20, 50)
    h_text, w_text = -1, -1

    while h_buffer < 0 or w_buffer < 0 :
        font_scale = round(font_scale / 1,5)
        text = text_to_image("+", "RozhaOne-Regular.ttf", font_scale)
        h_text, w_text, _ = text.shape
        h_buffer = round((h_between - h_text) / 2)
        w_buffer = round((w_between - w_text) / 2)

    collage_image[y_between + h_buffer:y_between+h_text+ h_buffer, x_between + w_buffer : x_between+w_text+ w_buffer ] = text



def init(collage_width, collage_height):
    # Create a blank white canvas for the collage
    collage_image = np.zeros((collage_height, collage_width, 3), dtype=np.uint8)
    collage_image.fill(255)  # Remplir l'image avec du blanc

    # Create a mask with the same size
    collage_masque = np.zeros((collage_height, collage_width, 3), dtype=np.uint8)
    collage_masque.fill(255)
    return collage_image, collage_masque

def prepare_picture(image_path):
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
        font_scale = np.random.randint(7, 18) / 20
        cv2.putText(image, formules[np.random.randint(0, len(formules))] + " (m/z = " + str(
            np.random.randint(2, 300)) + ")", (10, image.shape[0] - 5), font, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

    return image

def calculate_possible_positions(collage_image, positions, img_width, img_height, n):
    """
    returns a number corresponding to a random starter picture with a list of possible positions the next picture could be placed
    returns -1 and [] if there is no possible position for any of the pictures
    [x1, y1, direction, buffer_between_pictures] this is what elements look like in possible_position
    """
    collage_height, collage_width, _ = collage_image.shape
    already_placed_pictures = list(range(n))
    np.random.shuffle(already_placed_pictures)
    for i in already_placed_pictures:
        positionsPossibles = []
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
                    positionsPossibles.append([x1, y1, j, buffer_between_pictures])
        if len(positionsPossibles) != 0:
            return i, positionsPossibles

    return -1, []

def end_changes(collage_image, collage_masque):
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

def calculate_space_between(collage_image, x_new, y_new, direction, buffer_between_pictures, x_starter, y_starter, w_starter, h_starter, img_height, img_width):
    x_between, y_between = 0, 0

    h_between = min(img_height, h_starter)
    w_between = min(img_width, w_starter)

    if directions[direction][1] == 1:
        x_between = x_starter + w_starter
        w_between = buffer_between_pictures
    if directions[direction][1] == -1:
        x_between = x_starter - buffer_between_pictures
        w_between = buffer_between_pictures
    if directions[direction][1] == 0:
        x_between = x_starter

    if directions[direction][0] == 1:
        y_between = y_starter + h_starter
        h_between = buffer_between_pictures
    if directions[direction][0] == -1:
        y_between = y_starter - buffer_between_pictures
        h_between = buffer_between_pictures
    if directions[direction][0] == 0:
        y_between = y_starter

    cv2.rectangle(collage_image, (x_starter, y_starter + h_starter), (x_starter + w_starter, y_starter),
                  (255, 0, 0), 3)
    cv2.rectangle(collage_image, (x_between, y_between + h_between), (x_between + w_between, y_between),
                  (0, 255, 0), 3)
    cv2.rectangle(collage_image, (x_new, y_new + img_height), (x_new + img_width, y_new),
                  (0, 0, 255), 3)

    return x_between, y_between, h_between, w_between

def create_random_box(image_paths, collage_width, collage_height):
    # get empty slate
    collage_image, collage_masque = init(collage_width, collage_height)

    # Select a random arrow
    numFleche = np.random.randint(0, 7)

    n = 0  # Counter for the number of images placed
    positions = []  # Stores image positions as [x, y, img_width, img_height] with x, y left lower corner position

    for image_path in image_paths:
        image = prepare_picture(image_path)
        img_height, img_width, _ = image.shape

        # Place the first image at a random position
        if n == 0:
            # x and y are the left lower corner
            x, y = np.random.randint(0, collage_width - img_width), np.random.randint(0, collage_height - img_height)
            positions.append([x, y, img_width, img_height])
        else:
            # try for every already placed picture if there is space for the next picture in any direction
            starter , positionsPossibles = calculate_possible_positions(collage_image, positions, img_width, img_height, n)

            if starter == -1:
                print(f"L'image {image_path} ne peut pas être placée dans le collage.")
                continue


            # select a random position from the list of possible positions
            x_new, y_new, direction, buffer_between_pictures = positionsPossibles[np.random.randint(0, len(positionsPossibles))]
            x_starter, y_starter, w_starter, h_starter = positions[starter]
            x, y = x_new, y_new
            positions.append([x, y, img_width, img_height])


            # calculate the space between the two pictures
            x_between, y_between, h_between, w_between = calculate_space_between(collage_image, x_new, y_new, direction, buffer_between_pictures, x_starter, y_starter, w_starter, h_starter, img_height, img_width)

            # draw the arrows or plus
            if np.random.randint(0, 10) == 0:
                draw_plus(collage_image, x_between, y_between,  h_between, w_between)
            else :
                draw_arrow(collage_image, collage_masque, x_between, y_between, h_between, w_between, direction, numFleche)


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

        # crop image to fit collage size
        image = image[:img_height, :img_width]

        # Paste image into collage
        collage_image[y:y + img_height, x:x + img_width] = image

        # draw a mark arount the picture with 10% chance
        if np.random.randint(0, 10) == 0:
            draw_frames(collage_image, x, y, img_width, img_height)

        n += 1

    collage_image, collage_masque = end_changes( collage_image, collage_masque)


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
