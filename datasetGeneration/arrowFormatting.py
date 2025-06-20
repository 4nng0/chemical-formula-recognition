import cv2
import numpy as np

def formattingArrow(arrowNumber, weight=-1, height=-1, arrowReference=None):
    if weight == -1 and height == -1:
        filePath = f"datasetGeneration/arrowSource/backup/{arrowNumber}.png"
    else:
        filePath = f"datasetGeneration/arrowMask/backup/{arrowNumber}.png"
    image = cv2.imread(filePath, cv2.IMREAD_UNCHANGED)
    

    #si l’image n’a pas de canal alpha, on retourne une erreur
    if image.shape[2] != 4:
        raise ValueError("L'image n'a pas de canal alpha")

    # Séparer les canaux de couleur et le canal alpha
    bgr = image[:, :, :3]
    alpha = image[:, :, 3]

    # Créer un fond blanc
    white_background = np.ones_like(bgr, dtype=np.uint8) * 255

    # Combiner les images en tenant compte de l'alpha
    alpha_factor = alpha[:, :, np.newaxis] / 255.0 # np.newaxis est utilisé pour ajouter une dimension à l'array
    image_with_white_background = bgr * alpha_factor + white_background * (1 - alpha_factor) # Formule de la fusion d'images
    image_with_white_background = image_with_white_background.astype(np.uint8) # Convertir l'image fusionnée en uint8

    
            

    #on trouve le rectangle englobant de l’image
    x1, y1, x2, y2 = 0, 0, 0, 0

    for i in range(image_with_white_background.shape[0]):
        if np.any(image_with_white_background[i] != 255):
            y1 = i
            break
    for i in range(image_with_white_background.shape[0]-1, 0, -1):
        if np.any(image_with_white_background[i] != 255):
            y2 = i
            break
    for i in range(image_with_white_background.shape[1]):
        if np.any(image_with_white_background[:, i] != 255):
            x1 = i
            break
    for i in range(image_with_white_background.shape[1]-1, 0, -1):
        if np.any(image_with_white_background[:, i] != 255):
            x2 = i
            break
    
    #on rogne l’image originale pour ne garder que la flèche
    image = image[y1:y2, x1:x2]

    #si c’est un masque
    if arrowReference is not None:
    
        #si les dimensions du masque sont plus grandes que weight x height, on redimensionne le masque
        if image.shape[0] > height:
            image = cv2.resize(image, (image.shape[1], height))
        if image.shape[1] > weight:
            image = cv2.resize(image, (weight, image.shape[0]))

        #on crée une nouvelle image de taille weight x height entièrement transparente
        newImage = np.zeros((height, weight, 4), dtype=np.uint8)

        #on teste tous les emplacements possibles pour la placer la plus à droite possible
        scoreMin = np.inf
        jMin = 0
        for j in range(height - image.shape[0]):
            score = 0
            for k in range(j, j + image.shape[0]):
                for l in range(weight - image.shape[1], weight):
                    score += np.sum(np.abs(image[k-j, l-(weight-image.shape[1])] - arrowReference[k, l]))
            if score < scoreMin:
                scoreMin = score
                jMin = j
        #on place l’image à l’emplacement trouvé
        newImage[jMin:jMin+image.shape[0], weight-image.shape[1]:weight] = image
            

        
        image = newImage
    imgHeight, imgWidth = image.shape[:2]

    return image, imgHeight, imgWidth



def rotateArrow(image, mask):
    #on charge l’image

    images = []
    masks = []

    h, w = image.shape[:2]
    cX, cY = w // 2, h // 2

    for i in range(360):
        M = cv2.getRotationMatrix2D((cX, cY), i, 1.0)

        # Calculer la taille de la nouvelle image pour éviter le découpage
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # Ajuster la matrice de rotation pour prendre en compte la nouvelle taille
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        # Appliquer la rotation
        rotated = cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        rotatedMask = cv2.warpAffine(mask, M, (nW, nH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

        images.append(rotated)
        masks.append(rotatedMask)

    return images, masks

for i in range(7):
    print(i)
    #on les formate pour enlever la transparence
    formattedImage, imgHeight, imgWidth = formattingArrow(i)
    formattedMask, _, _ = formattingArrow(i, imgWidth, imgHeight, formattedImage)
    #on applique les rotations
    images, masks = rotateArrow(formattedImage, formattedMask)
    for j in range(len(images)):
        cv2.imwrite(f"datasetGeneration/arrowSource/{i}_{j}.png", images[j])
        cv2.imwrite(f"datasetGeneration/arrowMask/{i}_{j}.png", masks[j])
