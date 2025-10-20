#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 18:28:57 2025

@authors: Arbi Golemi, Jasper Veraart
"""

# Computational Intelligence Project - CNN version

import os
import cv2
import numpy as np
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_auc_score


def kachel():
    path = kagglehub.dataset_download("farzadnekouei/trash-type-image-dataset")
    print(path)


# Path to dataset
DATASET_DIR = "data/combined"

# Parameters
IMG_SIZE = 32  # CNNs work better with larger input (e.g. 64x64)
BATCH_SIZE = 32
EPOCHS = 40

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
        data.append(img)
        labels.append(category)

# Convert to numpy arrays
X = np.array(data, dtype="float32") / 255.0  # normalize pixel values
y = np.array(labels)

print("Images uploaded:", len(X))
print("Input dimensions:", X.shape)

# Encode the labels as numbers
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Convert to categorical (one-hot encoding)
num_classes = len(np.unique(y_encoded))
y_categorical = tf.keras.utils.to_categorical(y_encoded, num_classes)




# Split keeping labels for stratification
X_train, X_test, y_train, y_test, y_train_labels, y_test_labels = train_test_split(
    X, y_categorical, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Model builder function (recreate fresh model for each fold)
def build_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model



# Stratified K-Fold on the training set
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

fold_accuracies = []
best_val_acc = -1.0
best_model = None
best_history = None



# Data augmentation generator
datagen = ImageDataGenerator(
    rotation_range=15,         # rotazioni casuali fino a ±15°
    width_shift_range=0.1,     # spostamento orizzontale
    height_shift_range=0.1,    # spostamento verticale
    zoom_range=0.1,            # zoom casuale
    horizontal_flip=True       # flip orizzontale
)

datagen.fit(X_train)


print(f"Starting Stratified {n_splits}-Fold CV on training set ({len(X_train)} samples)")

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train_labels), 1):
    print(f"\n--- Fold {fold}/{n_splits} ---")
    model = build_model()

    callbacks = [
        EarlyStopping(patience=7, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=3, verbose=1)
    ]

    history = model.fit(
        datagen.flow(X_train[train_idx], y_train[train_idx], batch_size=BATCH_SIZE),
        validation_data=(X_train[val_idx], y_train[val_idx]),
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    # Evaluate on this fold's validation set
    val_loss, val_acc = model.evaluate(X_train[val_idx], y_train[val_idx], verbose=0)
    print(f"Fold {fold} val_acc: {val_acc:.4f}  val_loss: {val_loss:.4f}")

    fold_accuracies.append(val_acc)

    # Keep best model by validation accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model = model
        best_history = history

# CV summary
fold_accuracies = np.array(fold_accuracies)
print("\nCV accuracies per fold:", fold_accuracies)
print("CV mean accuracy: {:.4f} ± {:.4f}".format(fold_accuracies.mean(), fold_accuracies.std()))

# Save best model
if best_model is not None:
    best_model.save("best_model_fold.h5")
    print("Best model saved to best_model_fold.h5 (best val acc: {:.4f})".format(best_val_acc))

# Final evaluation on held-out test set using the best model
if best_model is not None:
    test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=0)
    print(f"\n✅ Test accuracy (best CV model): {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    # Predictions & reports
    y_pred = best_model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    print("\nClassification report:\n")
    print(classification_report(y_true, y_pred_classes, target_names=encoder.classes_))

    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=encoder.classes_, yticklabels=encoder.classes_, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix (best CV model)")
    plt.show()
    
    # Compute AUC (macro-average over all classes)
    y_pred_proba = best_model.predict(X_test)
    auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    print(f"AUC (macro-average): {auc:.4f}")

    # Compute specificity per class
    specificity = []
    for i in range(len(encoder.classes_)):
        tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        fp = np.sum(cm[:, i]) - cm[i, i]
        specificity.append(tn / (tn + fp))
    print("\nSpecificity per class:", np.round(specificity, 4))
    print("Mean specificity:", np.mean(specificity))

    # Plot training history of best fold
    if best_history is not None:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(best_history.history['accuracy'], label='Train acc')
        plt.plot(best_history.history['val_accuracy'], label='Val acc')
        plt.legend()
        plt.title('Accuracy (best fold)')

        plt.subplot(1, 2, 2)
        plt.plot(best_history.history['loss'], label='Train loss')
        plt.plot(best_history.history['val_loss'], label='Val loss')
        plt.legend()
        plt.title('Loss (best fold)')
        plt.show()

