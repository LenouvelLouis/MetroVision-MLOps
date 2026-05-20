# Projet IG2405 – Détection et Classification de Signes Métro
French version below / English version above
## Automatic Detection and Classification of Paris Metro Signs

The aim of this project is to automatically detect and classify pictograms on Paris metro lines from images.  
It combines computer vision techniques (Hough Circle Transform), descriptors (HOG) and machine learning models (binary CNN, k-NN).

It also includes a Gradio interface for easy testing of the model locally or on Hugging Face Spaces.

---

## 📁 Project structure

```
evaluationV2.py
metroChallenge.py
myMetroProcessing.py
requirements.txt
teamsNN.mat
test.py
train_cnn.py
train_knn_scaler.py
BD_CHALLENGE/
model/
    knn_line_model.joblib
    model_binary_real_metro.h5
    scaler_line.joblib
```

- **myMetroProcessing.py**: Main image processing functions, circle detection, CNN and k-NN application.
- **train_cnn.py**: Training the CNN model for binary sign classification.
- **train_knn_scaler.py**: Training the k-NN model for line classification, calculation and saving of the scaler.
- **evaluationV2.py**: Script for evaluating system performance on test and reference datasets.
- **metroChallenge.py**: Main script for launching the challenge on a set of images.
- **model/**: Folder containing the trained models (`.h5`, `.joblib`).
- **BD_CHALLENGE/**: Folder for challenge data.
- **requirements.txt**: Python dependencies for the project.

## Installation

1. Clone the repository and navigate to the project folder.
2. Create and activate a virtual environment (optional but recommended):
```sh
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install the dependencies:
```sh
pip install -r requirements.txt
```
## 🚀 Usage

### 1. Processing an image

Use the [`processOneMetroImage`](myMetroProcessing.py) function to process an image and detect/classify signs.

### 2. Evaluation


### 🔹 1. Start processing an image (complete pipeline)

The main function is located in **`myMetroProcessing.py`**:

```python
processOneMetroImage(name, image, index, resize_factor)
```
### 🔹 2. Start the challenge evaluation

To evaluate performance on a test set:
```sh
python metroChallenge.py
```
Modify the paths in the script if necessary to point to your reference and test `.mat` files.

Then, to evaluate the results with the reference file:
```python
from evaluationV2 import evaluation

evaluation(‘ControlFile.mat’, ‘YourFile.mat’, resize_factor=1.0)
```

## 🧩  Gradio interface (local)

A Gradio interface is provided in app.py.

###  ▶️ Launch the Gradio app locally
```sh
python app.py
```
Then go to:
```
http://127.0.0.1:7860
```

## Examples of input images
![IM (1).jpg](IM_(1).jpg)
![IM (2).jpg](IM_(2).jpg)

## Main dependencies

- numpy 
- opencv-python 
- scikit-image 
- scikit-learn 
- tensorflow / keras 
- matplotlib 
- pandas 
- joblib 
- pillow 
- gradio

See [`requirements.txt`](requirements.txt) for the complete list.

## 👥 Authors

- [ESTEVES Gabriel](https://github.com/GabrielEstevesDev)
- [LENOUVEL Louis](https://github.com/LenouvelLouis)

---

## Détection et Classification Automatique de Signes du Métro Parisien

Ce projet a pour objectif de détecter et classifier automatiquement les pictogrammes des lignes du métro parisien à partir d’images.  
Il combine des techniques de vision par ordinateur (Hough Circle Transform), de descripteurs (HOG) et de modèles d’apprentissage automatique (CNN binaire, k-NN).

Il inclut également une interface Gradio permettant de tester facilement le modèle en local ou sur Hugging Face Spaces.

---

## 📁 Structure du projet

```
evaluationV2.py
metroChallenge.py
myMetroProcessing.py
requirements.txt
teamsNN.mat
test.py
train_cnn.py
train_knn_scaler.py
BD_CHALLENGE/
model/
    knn_line_model.joblib
    model_binary_real_metro.h5
    scaler_line.joblib
```

- **myMetroProcessing.py** : Fonctions principales de traitement d’image, détection de cercles, application du CNN et du k-NN.
- **train_cnn.py** : Entraînement du modèle CNN pour la classification binaire des signes.
- **train_knn_scaler.py** : Entraînement du modèle k-NN pour la classification des lignes, calcul et sauvegarde du scaler.
- **evaluationV2.py** : Script d’évaluation des performances du système sur des jeux de données de test et de référence.
- **metroChallenge.py** : Script principal pour lancer le challenge sur un ensemble d’images.
- **model/** : Dossier contenant les modèles entraînés (`.h5`, `.joblib`).
- **BD_CHALLENGE/** : Dossier pour les données du challenge.
- **requirements.txt** : Dépendances Python du projet.

## Installation

1. Clonez le dépôt et placez-vous dans le dossier du projet.
2. Créer et activer un environnement virtuel (optionnel mais recommandé) :
   ```sh
   python -m venv venv
   source venv/bin/activate  # Sur Windows : venv\Scripts\activate
   ```
3. Installez les dépendances :
   ```sh
   pip install -r requirements.txt
   ```
## 🚀 Utilisation

### 1. Traitement d’une image

Utilisez la fonction [`processOneMetroImage`](myMetroProcessing.py) pour traiter une image et détecter/classifier les signes.

### 2. Évaluation


### 🔹 1. Lancer le traitement d'une image (pipeline complet)

La fonction principale se trouve dans **`myMetroProcessing.py`** :

```python
processOneMetroImage(nom, image, index, resize_factor)
```
### 🔹 2. Lancer l’évaluation du challenge

Pour évaluer les performances sur un jeu de test :
```sh
python metroChallenge.py
```
Modifiez les chemins dans le script si besoin pour pointer vers vos fichiers `.mat` de référence et de test.

Puis pour évaluer les résultats avec le fichier de référence :
```python
from evaluationV2 import evaluation

evaluation("FichierContrôle.mat", "VotreFichier.mat", resize_factor=1.0)
```

## 🧩  Interface Gradio (local)

Une interface Gradio est fournie dans app.py.

###  ▶️ Lancer l’app Gradio en local
```sh
python app.py
```
Accéder ensuite à :
```
http://127.0.0.1:7860
```

## Exemples d’images d’entrée
![IM (1).jpg](IM_(1).jpg)
![IM (2).jpg](IM_(2).jpg)

## Dépendances principales

- numpy 
- opencv-python 
- scikit-image 
- scikit-learn 
- tensorflow / keras 
- matplotlib 
- pandas 
- joblib 
- pillow 
- gradio

Voir [`requirements.txt`](requirements.txt) pour la liste complète.

## 👥 Auteurs

- [ESTEVES Gabriel](https://github.com/GabrielEstevesDev)
- [LENOUVEL Louis](https://github.com/LenouvelLouis)

---

Projet IG2405 – Vision par ordinateur – ISEP 2025
