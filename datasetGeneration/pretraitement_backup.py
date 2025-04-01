import numpy as np
import cv2

def pretraitement(arrowPath):
    arrow = cv2.imread(arrowPath, cv2.IMREAD_UNCHANGED)

    #si l’image n’a pas de canal alpha, on en ajoute un en remplaçant le blanc par de la transparence
    if arrow.shape[2] < 4:
        b, g, r = cv2.split(arrow)
        a = np.ones(b.shape, dtype=b.dtype) * 255
        for i in range(b.shape[0]):
            for j in range(b.shape[1]):
                if b[i, j] == 255 and g[i, j] == 255 and r[i, j] == 255:
                    a[i, j] = 0
        arrow = cv2.merge((arrow, a))
    else:
        #on supprime les pixels trop peu visibles
        for i in range(arrow.shape[0]):
            for j in range(arrow.shape[1]):
                if arrow[i, j, 3] < 127 :
                    arrow[i, j] = [255, 255, 255, 0]
    
    #on enregistre l'image modifiée
    cv2.imwrite(arrowPath, arrow)


for k in range(7):
    print(k)
    filePath1 = f"datasetGeneration/arrowSource/backup/{k}.png"
    filePath2 = f"datasetGeneration/arrowMask/backup/{k}.png"
    pretraitement(filePath1)
    pretraitement(filePath2)