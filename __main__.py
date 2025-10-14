#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 18:28:57 2025

@authors: Arbi golemi, Jasper Veraart
"""

# Computational intelligence project

import os
import cv2
import numpy as np
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier

def kachel():
    path = kagglehub.dataset_download("farzadnekouei/trash-type-image-dataset")
    print(path)

# Path to dataset
DATASET_DIR = "dataset"

# Parameters
IMG_SIZE = 10  # resize all images to 10x10 pixels

# Lists for data and labels
data = []
labels = []

# Load all images
for category in os.listdir(DATASET_DIR):
    category_path = os.path.join(DATASET_DIR, category)
    if not os.path.isdir(category_path):
        continue
    for img_name in os.listdir(category_path):
        img_path = os.path.join(category_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.flatten()  # from 64x64x3 to a 1D vector
        data.append(img)
        labels.append(category)

# Convert to numpy arrays
X = np.array(data)
y = np.array(labels)

print("Images uploaded:", len(X))
print("Input dimensions:", X.shape)

# Encode the labels as numbers
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Split into train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(128, 64),
                    activation='relu',
                    solver='adam',
                    max_iter=1000,
                    random_state=42)

mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)

print(classification_report(y_test, y_pred, target_names=encoder.classes_))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
