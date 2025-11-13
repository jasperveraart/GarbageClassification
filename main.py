# @title
import os
import cv2
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

import sys, os
os.environ['PYTHONUNBUFFERED'] = '1'

# -----------------------------
# PARAMETERS
# -----------------------------
DATASET_DIR = "/content/drive/MyDrive/garbage_dataset"
IMG_SIZE = 64
BATCH_SIZE = 32
GSA_EPOCHS = 15
FINAL_EPOCHS = 40

# GSA PARAMETERS
num_agents = 4
num_iterations = 3
param_bounds = {
    'learning_rate': (1e-4, 5e-3),
    'dropout1': (0.2, 0.4),
    'dropout2': (0.4, 0.6)
}

# -----------------------------
# LOAD DATA
# -----------------------------
data, labels = [], []

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

X = np.array(data, dtype="float32") / 255.0
y = np.array(labels)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
num_classes = len(np.unique(y_encoded))
y_categorical = tf.keras.utils.to_categorical(y_encoded, num_classes)

# -----------------------------
# SPLIT DATA
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
)
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42, shuffle=True
)

# -----------------------------
# DATA AUGMENTATION
# -----------------------------
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

# -----------------------------
# MODEL BUILDER
# -----------------------------
def build_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=num_classes,
                dropout1=0.3, dropout2=0.5, learning_rate=1e-3):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.Dropout(dropout1),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(dropout2),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# -----------------------------
# FITNESS FUNCTION
# -----------------------------
def fitness(agent):
    model = build_model(
        dropout1=agent['dropout1'],
        dropout2=agent['dropout2'],
        learning_rate=agent['learning_rate']
    )
    history = model.fit(
        datagen.flow(X_train_final, y_train_final, batch_size=BATCH_SIZE),
        validation_data=(X_val, y_val),
        epochs=GSA_EPOCHS,
        verbose=2
    )
    val_acc = max(history.history['val_accuracy'])
    agent['fitness'] = 1 - val_acc
    return agent['fitness']

# -----------------------------
# INITIALIZE AGENTS
# -----------------------------
agents = []
for _ in range(num_agents):
    agents.append({
        'learning_rate': random.uniform(*param_bounds['learning_rate']),
        'dropout1': random.uniform(*param_bounds['dropout1']),
        'dropout2': random.uniform(*param_bounds['dropout2']),
        'fitness': np.inf
    })

# -----------------------------
# GSA MAIN LOOP
# -----------------------------
for it in range(num_iterations):
    print(f"\n--- GSA Iteration {it+1}/{num_iterations} ---", flush=True)
    for agent in agents:
        fitness(agent)
        print(f"lr={agent['learning_rate']:.5f}, d1={agent['dropout1']:.2f}, d2={agent['dropout2']:.2f}, fit={agent['fitness']:.4f}", flush=True)

    best = min(agents, key=lambda x: x['fitness'])
    for agent in agents:
        if agent == best:
            continue
        agent['learning_rate'] += 0.15 * (best['learning_rate'] - agent['learning_rate'])
        agent['dropout1'] += 0.15 * (best['dropout1'] - agent['dropout1'])
        agent['dropout2'] += 0.15 * (best['dropout2'] - agent['dropout2'])

# -----------------------------
# FINAL TRAINING ON BEST AGENT
# -----------------------------
best = min(agents, key=lambda x: x['fitness'])
print(f"\nBest Hyperparams: lr={best['learning_rate']:.5f}, d1={best['dropout1']:.2f}, d2={best['dropout2']:.2f}", flush=True)

final_model = build_model(
    dropout1=best['dropout1'],
    dropout2=best['dropout2'],
    learning_rate=best['learning_rate']
)

callbacks = [
    EarlyStopping(patience=7, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(factor=0.5, patience=3, verbose=1)
]

final_model.fit(
    datagen.flow(X_train_final, y_train_final, batch_size=BATCH_SIZE),
    validation_data=(X_val, y_val),
    epochs=FINAL_EPOCHS,
    callbacks=callbacks,
    verbose=2
)

# -----------------------------
# EVALUATION
# -----------------------------
test_loss, test_acc = final_model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Test accuracy: {test_acc:.4f}", flush=True)
print(f"Test loss: {test_loss:.4f}", flush=True)

y_pred = final_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nClassification report:\n", flush=True)
print(classification_report(y_true, y_pred_classes, target_names=encoder.classes_), flush=True)

cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=encoder.classes_, yticklabels=encoder.classes_, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()