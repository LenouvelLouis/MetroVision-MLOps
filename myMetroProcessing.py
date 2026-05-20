# -*- coding: utf-8 -*-
"""
Created on Sun Jun 15 2025

@author: ESTEVES Gabriel & LENOUVEL Louis
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import skimage as ski
from skimage.feature import hog
import cv2
import tensorflow as tf
import joblib
import sklearn

# Chemins vers les modèles sauvegardés
cnn_model_path = "./model/model_binary_real_metro.h5"
knn_model_path = "./model/knn_line_model.joblib"
scaler_path = "./model/scaler_line.joblib"
IMG_SIZE = 64

# Paramètres pour la détection de cercles avec Hough
hough_params = {'dp': 1.2, 'minDist': 25, 'param1': 120, 'param2': 60, 'minRadius': 20, 'maxRadius': 100}

# Paramètres HOG : à passer à la fonction hog()
hog_params = dict(
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm='L2-Hys',
    transform_sqrt=False,
    feature_vector=True
)

# Seuil pour la classification binaire du CNN
cnn_thresh = 0.5  # 0.5 est un choix courant


def load_models():
    """
    Charge les modèles CNN, KNN et le scaler en mémoire.
    Appeler cette fonction une seule fois avant d'utiliser processOneMetroImage.
    """
    global model_bin, knn, scaler_line
    model_bin = tf.keras.models.load_model(cnn_model_path)
    knn = joblib.load(knn_model_path)
    scaler_line = joblib.load(scaler_path)


def processOneMetroImage(nom, im, n, resizeFactor):
    """
    Traite une image de métro pour détecter les lignes et les classer.
    @param nom: Nom de l'image
    @param im: Image à traiter (numpy array)
    @param n: Index de l'image dans la liste
    @param resizeFactor: Facteur de redimensionnement de l'image
    @return: Tuple (image redimensionnée, tableau des lignes détectées)
    """
    # Redimensionnement de l'image si nécessaire
    if resizeFactor != 1:
        im_resized = ski.transform.resize(im, (int(im.shape[0] * resizeFactor), int(im.shape[1] * resizeFactor)),
                                          anti_aliasing=True, preserve_range=True).astype(im.dtype)
    else:
        im_resized = im

        # Charger les modèles CNN, KNN et le scaler
    # model_bin = tf.keras.models.load_model(cnn_model_path)
    # knn = joblib.load(knn_model_path)
    # scaler_line = joblib.load(scaler_path)

    # Conversion de l'image en uint8 si besoin
    if im_resized.dtype != np.uint8:
        if im_resized.max() <= 1.0:
            im_resized = (im_resized * 255).astype(np.uint8)
        else:
            im_resized = im_resized.astype(np.uint8)

    # Conversion en niveaux de gris et flou médian
    gray = cv2.cvtColor(im_resized, cv2.COLOR_RGB2GRAY)
    blur = cv2.medianBlur(gray, 7)

    #Détection de cercles
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        **hough_params
    )
    circles = np.round(circles[0]).astype(int) if circles is not None else []

    # Pour chaque cercle détecté, appliquer CNN puis KNN si nécessaire
    rows = []
    for cx, cy, r in circles:
        # Coordonnées du ROI carré autour du cercle
        y1, y2 = max(cy - r, 0), min(cy + r, gray.shape[0] - 1)
        x1, x2 = max(cx - r, 0), min(cx + r, gray.shape[1] - 1)
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        # --- Prédiction binaire avec le CNN
        roi_cnn = cv2.resize(roi, (IMG_SIZE, IMG_SIZE)).astype('float32') / 255.0
        inp = roi_cnn[np.newaxis, ..., np.newaxis]
        p = float(model_bin.predict(inp, verbose=0)[0, 0])

        # Si la probabilité est inférieure au seuil, on considère que c'est un faux positif
        if p < cnn_thresh:
            classe = 0  # FP
        else:
            # --- Extraction des features HOG + classification KNN
            roi_line = cv2.resize(roi, (IMG_SIZE, IMG_SIZE), cv2.INTER_AREA)
            feat = hog(roi_line, **hog_params).reshape(1, -1)
            feat_s = scaler_line.transform(feat)
            classe = int(knn.predict(feat_s)[0])
            rows.append([n, y1, y2, x1, x2, classe])

    # Création du tableau de résultats
    bd = np.array(rows, dtype=int) if rows else np.zeros((0, 6), dtype=int)

    #display
    # plt.figure()
    # plt.imshow(im_resized)
    # for k in range(bd.shape[0]):
    #     draw_rectangle(bd[k,3], bd[k,4], bd[k,1], bd[k,2], 'g')
    # plt.title(f'{nom} - Lines {bd[:,5]}')
    # plt.show()

    return im_resized, bd


# Additional function =========================================================

def draw_rectangle(x1, x2, y1, y2, color):
    rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor='none')
    ax = plt.gca()
    ax.add_patch(rect)
