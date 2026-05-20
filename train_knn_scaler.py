from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib  # <- ajouté pour la sauvegarde
import os
import cv2
import numpy as np
from skimage.feature import hog
import pandas as pd


# -----------------------------------------------
# Paramètres & chemins
# -----------------------------------------------
apprentissage = "/content/drive/MyDrive/Projet_VPO_A2/Apprentissage_VF.xlsx"
test_path     = "/content/drive/MyDrive/Projet_VPO_A2/Test.xlsx"
folder_images = "/content/drive/MyDrive/Projet_VPO_A2/BD_METRO/"
IMG_SIZE       = 64


# Paramètres HOG : bien les passer à hog(), ne pas confondre avec hough_params
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

# --- 1) Extraire les ROIs GT et leurs labels HYP
df_classifier = df_train[df_train['HYP'] > 0]
line_rois, line_labels = [], []
for _, row in df_classifier.iterrows():
    img_path = os.path.join(folder_images, f"{row['NOM']}.JPG")
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    x1, x2 = int(row['x1']), int(row['x2'])
    y1, y2 = int(row['y1']), int(row['y2'])
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        continue
    roi_r = cv2.resize(roi, (IMG_SIZE, IMG_SIZE), cv2.INTER_AREA)
    line_rois.append(roi_r)
    line_labels.append(int(row['HYP']))

# --- 2) Calcul des vecteurs HOG
X_line = np.vstack([hog(r, **hog_params) for r in line_rois])
y_line = np.array(line_labels, dtype='int32')

# --- 3) Split train/test
X_tr_line, X_te_line, y_tr_line, y_te_line = train_test_split(
    X_line, y_line,
    test_size=0.2,
    stratify=y_line,
    random_state=42
)

# --- 4) Standardisation
scaler_line = StandardScaler().fit(X_tr_line)
X_tr_scaled = scaler_line.transform(X_tr_line)
X_te_scaled = scaler_line.transform(X_te_line)

# --- 5) Entraînement k-NN
knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn.fit(X_tr_scaled, y_tr_line)

# --- 6) Évaluation
y_pred_line = knn.predict(X_te_scaled)
print(f"Accuracy k-NN ligne : {accuracy_score(y_te_line, y_pred_line)*100:.2f}%\n")
print("Classification report :")
print(classification_report(y_te_line, y_pred_line, zero_division=0))
print("Confusion matrix :")
print(confusion_matrix(y_te_line, y_pred_line))

# --- 7) Sauvegarde du modèle et du scaler
model_knn_path   = "/content/drive/MyDrive/Projet_VPO_A2/knn_line_model.joblib"
scaler_path      = "/content/drive/MyDrive/Projet_VPO_A2/scaler_line.joblib"
joblib.dump(knn, model_knn_path)
joblib.dump(scaler_line, scaler_path)
print(f"Modèle k-NN sauvegardé dans : {model_knn_path}")
print(f"Scaler sauvegardé dans : {scaler_path}")