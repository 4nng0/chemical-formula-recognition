import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image#
import random
import string


basic_symbols = string.ascii_letters + string.digits + " !@#$%^&*()-_{}[];:,.<>?/|\\`~'\""


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
number_of_fonts = 12

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

def draw_random_shapes(image):
    color = (0, 0, 0)
    height, width = image.shape[:2]

    shape_type = random.choice([
        'rectangle', 'circle', 'line', 'polygon', 'ellipse',
        'points', 'filled_polygon', 'bezier', 'sinus_line'
    ])

    if shape_type == 'rectangle':
        pt1 = (random.randint(0, width), random.randint(0, height))
        pt2 = (random.randint(0, width), random.randint(0, height))
        cv2.rectangle(image, pt1, pt2, color, -1)

    elif shape_type == 'circle':
        center = (random.randint(0, width), random.randint(0, height))
        radius = random.randint(5, min(width, height) // 2)
        cv2.circle(image, center, radius, color, -1)

    elif shape_type == 'line':
        pt1 = (random.randint(0, width), random.randint(0, height))
        pt2 = (random.randint(0, width), random.randint(0, height))
        cv2.line(image, pt1, pt2, color, random.randint(1, 10))

    elif shape_type == 'polygon':
        num_points = random.randint(3, 8)
        points = [(random.randint(0, width), random.randint(0, height)) for _ in range(num_points)]
        cv2.polylines(image, [np.array(points)], isClosed=True, color=color, thickness=random.randint(1, 5))

    elif shape_type == 'filled_polygon':
        num_points = random.randint(3, 10)
        points = [(random.randint(0, width), random.randint(0, height)) for _ in range(num_points)]
        cv2.fillPoly(image, [np.array(points)], color=color)

    elif shape_type == 'ellipse':
        center = (random.randint(0, width), random.randint(0, height))
        axes = (random.randint(10, width // 4), random.randint(10, height // 4))
        angle = random.randint(0, 360)
        startAngle = 0
        endAngle = 360
        cv2.ellipse(image, center, axes, angle, startAngle, endAngle, color, -1)

    elif shape_type == 'points':
        for _ in range(random.randint(50, 200)):
            pt = (random.randint(0, width-1), random.randint(0, height-1))
            image[pt[1], pt[0]] = color

    elif shape_type == 'bezier':
        # Simuliere eine gebogene Linie durch viele kleine Punkte
        points = [(random.randint(0, width), random.randint(0, height)) for _ in range(4)]
        curve = cv2.approxPolyDP(np.array(points, dtype=np.int32), 3, False)
        cv2.polylines(image, [curve], isClosed=False, color=color, thickness=2)

    elif shape_type == 'sinus_line':
        freq = random.uniform(0.05, 0.15)
        amp = random.randint(10, 40)
        y_offset = random.randint(0, height)
        for x in range(width):
            y = int(amp * np.sin(2 * np.pi * freq * x) + y_offset)
            if 0 <= y < height:
                image[y, x] = color

def rotate(image, degrees):
    h, w = image.shape[:2]
    cX, cY = w // 2, h // 2
    M = cv2.getRotationMatrix2D((cX, cY), degrees, 1.0)


    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))


    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY


    return cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(255, 255, 255, 0))

def make_noise():

    option = np.random.choice(["Figure", "Text", "Letter"])#

    if option == "Letter":
        font_scale = np.random.randint(20, 60)
        font_path  = f"Fonts/{np.random.randint(1, number_of_fonts)}.ttf"
        noise = text_to_image(random.choice(basic_symbols),  font_path, font_scale)
        degrees = np.random.randint(0, 360)
        noise = rotate(noise, degrees)
        return noise

    if option == "Text":
        font_scale = np.random.randint(10, 30)
        font_path  = f"Fonts/{np.random.randint(11, 12)}.ttf"
        k = np.random.randint(3, 15)
        text = ''.join(random.choices(basic_symbols, k=k))
        noise = text_to_image(text,  font_path, font_scale)
        degrees = np.random.randint(0, 360)
        noise = rotate(noise, degrees)
        return noise

    if option == "Figure":
        image = np.full((np.random.randint(40, 100), np.random.randint(40, 100), 3), (255, 255, 255), dtype=np.uint8)
        draw_random_shapes(image)
        return image
    return None

def place_noise(collage_image, noise):
    x_orig, y_orig = np.random.randint(0, collage_image.shape[0] - noise.shape[0]), np.random.randint(0, collage_image.shape[1] - noise.shape[1])
    x, y = x_orig, y_orig
    y_distance, x_distance = 0, 0
    jumps = 50
    swich = 1
    max_val = round(np.max([collage_image.shape[0], collage_image.shape[1]]) / jumps)
    if np.mean(collage_image[x: x + noise.shape[0], y: y + noise.shape[1]]) == 255:
        collage_image[x: x + noise.shape[0], y: y + noise.shape[1]] = noise
        return
    for a in range(1, max_val):
        swich = swich * -1

        for b in range(1, a):
            x_distance += jumps * swich
            if np.array_equal(collage_image[x + x_distance: x + x_distance+ noise.shape[0], y+ y_distance: y+ y_distance + noise.shape[1]], noise) and np.mean(collage_image[x + x_distance: x + x_distance+ noise.shape[0], y+ y_distance: y+ y_distance + noise.shape[1]]) == 255:
                collage_image[x: x + noise.shape[0], y: y + noise.shape[1]] = noise
                return

        for b in range(1, a):
            y_distance += jumps * swich
            if np.array_equal(collage_image[x + x_distance: x + x_distance+ noise.shape[0], y+ y_distance: y+ y_distance + noise.shape[1]], noise) and np.mean(collage_image[x + x_distance: x + x_distance+ noise.shape[0], y+ y_distance: y+ y_distance + noise.shape[1]]) == 255:
                collage_image[x: x + noise.shape[0], y: y + noise.shape[1]] = noise
                return



def text_to_image(text, font_path, font_size=40, text_color=(0, 0, 0)):
    # Word of warning: in this methode it can come to a endless loop. it doesn't happen often but i could not fix the problem

    font = ImageFont.truetype(font_path, font_size)


    if not text.strip():
        raise ValueError("text is empty")


    # calculate necessary size with  getbbox
    bbox = font.getbbox(text)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]

    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Text is drawn
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

def draw_plus(collage_image, x_between, y_between, h_between, w_between):
    font_scale = np.random.randint(20, 50)
    h_buffer, w_buffer = -1, -1
    font = np.random.randint(1, number_of_fonts)

    while h_buffer < 0 or w_buffer < 0 :
        font_scale = round(font_scale / 1,5)
        text = text_to_image("+", f"Fonts/{font}.ttf", font_scale)
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
        font_path  = f"Fonts/{np.random.randint(1, number_of_fonts)}.ttf"
        font_scale = np.random.randint(10, 20)
        text = formules[np.random.randint(0, len(formules))] + " (m/z = " + str(np.random.randint(2, 300)) + ")"
        text_image = text_to_image(text, font_path, font_scale)
        height = min(text_image.shape[0],  25)
        width = min(text_image.shape[1], image.shape[1] - 5)  # maximal verfügbare Breite

        image[image.shape[0] - 25  : image.shape[0] - 25   + height,5 :5 + width] = text_image[:height, :width]

    # 20% chance to add a border
    if np.random.randint(0, 5) == 1:
        option = np.random.choice(["solid_box", "dotted_box", "parentheses", ])  # all are same probability
        padding = np.random.randint(30, 55)
        color = (0, 0, 0)
        thickness = np.random.randint(1, 5)

        if option == "solid_box":
            newimage = np.zeros((image.shape[0] + padding * 2, image.shape[1] + padding * 2, 3), dtype=np.uint8) + 255
            newimage[padding:padding + image.shape[0], padding:padding + image.shape[1]] = image
            image = newimage
            # draw frame
            cv2.rectangle(image, (5,5), (image.shape[1] -5, image.shape[0] -5), color, thickness)

        elif option == "dotted_box":
            # draw dotted frame

            newimage = np.zeros((image.shape[0] + padding * 2, image.shape[1] + padding * 2, 3), dtype=np.uint8) + 255
            newimage[padding:padding + image.shape[0], padding:padding + image.shape[1]] = image
            image = newimage
            top_left = (5, 5)
            bottom_right = (image.shape[1] -5, image.shape[0] -5)
            dash_length = np.random.randint(5, 25)

            for i in range(top_left[0], bottom_right[0], dash_length * 2):
                cv2.line(image, (i, top_left[1]), (i + dash_length, top_left[1]), color, thickness)
                cv2.line(image, (i, bottom_right[1]), (i + dash_length, bottom_right[1]), color, thickness)
            for i in range(top_left[1], bottom_right[1], dash_length * 2):
                cv2.line(image, (top_left[0], i), (top_left[0], i + dash_length), color, thickness)
                cv2.line(image, (bottom_right[0], i), (bottom_right[0], i + dash_length), color, thickness)

        elif option == "parentheses":
            # draw parentheses
            kind = np.random.choice(["basic", "font"])
            if kind == "basic":
                # basic just draws lines to get the wanted effect
                newimage = np.zeros((image.shape[0] + padding * 2, image.shape[1] + padding * 2, 3), dtype=np.uint8) + 255
                newimage[padding:padding + image.shape[0], padding:padding + image.shape[1]] = image
                image = newimage
                top_left = (5, 5)
                bottom_right = (image.shape[1] -5, image.shape[0] -5)
                dash_length = np.random.randint(5, 25)

                length = np.random.randint(20, 45)  # Länge der Klammern

                # left side
                cv2.line(image, top_left, (top_left[0] + length, top_left[1]), color, thickness)
                cv2.line(image, top_left, (top_left[0], bottom_right[1]), color, thickness)
                cv2.line(image, (top_left[0], bottom_right[1]), (top_left[0] + length, bottom_right[1]), color,
                         thickness)

                # right side
                cv2.line(image, (bottom_right[0], top_left[1]), (bottom_right[0] - length, top_left[1]), color,
                         thickness)
                cv2.line(image, (bottom_right[0], top_left[1]), (bottom_right[0], bottom_right[1]), color,
                         thickness)
                cv2.line(image, bottom_right, (bottom_right[0] - length, bottom_right[1]), color, thickness)
            elif kind == "font":
                # here the parentisis from the different fonts are used and added on the sides of the image
                if np.random.randint(0, 2) == 0:
                    l = "["
                    r = "]"
                else:
                    l = "{"
                    r = "}"

                font_path = f"Fonts/{np.random.randint(1, number_of_fonts)}.ttf"

                left = text_to_image(l, font_path, 50)
                h, w = left.shape[:2]
                scale = image.shape[0] / h
                new_width = int(w * scale)
                left = cv2.resize(left, (new_width, image.shape[0]))

                right = text_to_image(r, font_path, 50)
                right = cv2.resize(right, (new_width, image.shape[0]))

                image = cv2.hconcat([left, image, right])


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

    #cv2.rectangle(collage_image, (x_starter, y_starter + h_starter), (x_starter + w_starter, y_starter),
    #              (255, 0, 0), 3)
    #cv2.rectangle(collage_image, (x_between, y_between + h_between), (x_between + w_between, y_between),
    #              (0, 255, 0), 3)
    #cv2.rectangle(collage_image, (x_new, y_new + img_height), (x_new + img_width, y_new),
    #              (0, 0, 255), 3)

    return x_between, y_between, h_between, w_between

def create_random_box(image_paths, collage_width, collage_height):
    # get empty slate
    collage_image, collage_masque = init(collage_width, collage_height)

    # Select a random arrow
    numFleche = np.random.randint(0, 7)

    n = 0  # Counter for the number of images placed
    positions = []  # Stores image positions as [x, y, img_width, img_height] with x, y left lower corner position

    for image_path in image_paths:
        #loades the image and maybe adds text or a frame
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

        n += 1

    collage_image, collage_masque = end_changes( collage_image, collage_masque)


    if np.random.randint(0, 2) == 0:
        # add noise to make overfitting more unlikely
        while np.random.randint(0, 4) != 0 :
            noise = make_noise()
            place_noise(collage_image, noise)

    return collage_image, collage_masque


if __name__ == "__main__":

    for k in range(10):

        # List of image paths
        image_paths = []
        nbMolecules = np.random.randint(5, 15)
        for j in range(nbMolecules):
            i = np.random.randint(0, 40000)
            image_paths.append(f"chemicalStructureSource/{i}.png")

        # Dimensions of the image
        collage_width = 1500
        collage_height = 1500

        print(image_paths)

        # make the image
        # try to create a collage, if it returns an error, try again


        while 1 == 1:
            try:
                collage, masque = create_random_box(image_paths, collage_width, collage_height)
                break
            except Exception as e:
                print(f"Error making the picture: {e}")
                continue

        # Save collage

        cv2.imwrite(f'data/images/{k}.jpg', collage)
        cv2.imwrite(f'data/masks/{k}.jpg', masque)
