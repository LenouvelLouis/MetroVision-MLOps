import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------
# Paramètres & chemins
# -----------------------------------------------
apprentissage = "/content/drive/MyDrive/Projet_VPO_A2/Apprentissage_VF.xlsx"
test_path     = "/content/drive/MyDrive/Projet_VPO_A2/Test.xlsx"
folder_images = "/content/drive/MyDrive/Projet_VPO_A2/BD_METRO/"
model_path    = "/content/drive/MyDrive/Projet_VPO_A2/model_binary_real_metro.h5"
IMG_SIZE      = 64
batch_size    = 32
epochs        = 10

# Paramètres HoughCircles pour inférence
hough_params = {
    'dp': 1.3,
    'minDist': 50,
    'param1': 120,
    'param2': 50,
    'minRadius': 20,
    'maxRadius': 100
}
# Paramètres HOG pour classification de lignes
hog_params = dict(
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm='L2-Hys',
    transform_sqrt=False,
    feature_vector=True
)

# -----------------------------------------------
# Chargement des données d'apprentissage
# -----------------------------------------------
df_train = pd.read_excel(apprentissage)

# -----------------------------------------------
# A) Préparation des données pour CNN binaire
# -----------------------------------------------
# Utiliser directement les ROIs fournis (GT) avec HYP>0 ou HYP=0
X_train, y_train = [], []
for _, row in df_train.iterrows():
    img_path = os.path.join(folder_images, f"{row['NOM']}.JPG")
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    x1, x2 = int(row['x1']), int(row['x2'])
    y1, y2 = int(row['y1']), int(row['y2'])
    ymin, ymax = max(0, y1), min(img.shape[0], y2)
    xmin, xmax = max(0, x1), min(img.shape[1], x2)
    if ymin >= ymax or xmin >= xmax:
        continue
    roi = img[ymin:ymax, xmin:xmax]
    if roi.size == 0:
        continue
    roi_resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE), cv2.INTER_AREA)
    X_train.append(roi_resized)
    # label 1 si HYP>0, sinon 0
    y_train.append(1 if int(row['HYP'])>0 else 0)

# Conversion en numpy arrays
X_train = np.array(X_train, dtype='float32') / 255.0
X_train = X_train[..., np.newaxis]
y_train = np.array(y_train, dtype='float32')
print(f"CNN binaire train: {len(y_train)} exemples, +:{int(y_train.sum())}, -:{len(y_train)-int(y_train.sum())}")

# -----------------------------------------------
# B) Construction & entraînement du CNN binaire
# -----------------------------------------------
model_bin = models.Sequential([
    layers.Input((IMG_SIZE, IMG_SIZE, 1)),
    layers.Conv2D(32, 3, activation='relu', padding='same'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu', padding='same'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model_bin.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_bin.summary()
model_bin.fit(X_train, y_train, validation_split=0.2, epochs=epochs, batch_size=batch_size)
model_bin.save(model_path)
print(f"Modèle sauvegardé dans : {model_path}")