# ♻️ Garbage Classification – Computational Intelligence Project

This project combines multiple public Kaggle waste classification datasets into one unified dataset for training and experimentation.
It automatically downloads, merges, cleans, and normalizes image data into **five waste categories**:

> `glass`, `metal`, `paper`, `plastic`, `residual`

---

## 📂 Project structure

```
GarbageClassification/
│
├── data/
│ ├── raw/ # Cached Kaggle datasets (downloaded automatically)
│ └── combined/ # Final merged dataset used for training
│
├── scripts/
│ ├── dataset_setup.py # Downloads + merges datasets into 5 classes
│ └── cleanup_datasets.py # Removes all dataset files safely
│
├── main.py # Model training and evaluation script
└── README.md
```

---

## ⚙️ Setup instructions

### 1. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

## Install dependencies
```bash
pip install kagglehub numpy scikit-learn opencv-python matplotlib seaborn
```


## 📦 Build the dataset
```bash
python3 scripts/dataset_setup.py
```

This will:

- Download all three Kaggle datasets using KaggleHub
- Merge and normalize all folder names
- Remove unwanted categories (e.g. clothes, textile)
- Deduplicate identical images (binary-level comparison)
- Rename all images to include their class label and a short hash

✅ After completion, your data/combined/ folder will contain roughly 10,000 unique images organized by class.

## 🧹 Clean up datasets

To remove all dataset files (both raw downloads and combined images):

```bash
python3 scripts/dataset_remove.py
```

This safely deletes:

- data/raw/*
- data/combined/*

while keeping the folder structure intact.

## Train model
Coming soon

## 🙌 Credits

Datasets used:

- https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification
- https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification
- https://www.kaggle.com/datasets/farzadnekouei/trash-type-image-dataset

Project authors:
Arbi Golemi & Jasper Veraart
Computational Intelligence – 2025