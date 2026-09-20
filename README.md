# 🔎 Forensic Analysis: Glass Identification

A Machine Learning framework for predicting **glass types** (building windows, vehicle windows, containers, tableware, headlamps) from **refractive index (RI)** and **elemental oxide concentrations** (Na, Mg, Al, Si, K, Ca, Ba, Fe). 

This repository models the exact analytical measurements taken by forensic scientists inspecting micro-fragmented glass recovered from crime scenes, hit-and-run incidents, or burglary investigations.

---

## 📌 Executive Summary & Key Results

| Strategy / Model Stage | Performance Metric | Details |
|---|---|---|
| **Baseline (Majority Class Guess)** | Macro-F1: `0.09` \| Accuracy: `36.0%` | Predicts type 2 indiscriminately |
| **Random Forest + SMOTE (Best Model)** | CV Macro-F1: `0.72 ± 0.05` \| Accuracy: `75.3%` | Repeated stratified 5-fold CV |
| **Held-Out Test Set (Single Split)** | Macro-F1: `0.83` \| Accuracy: `81.4%` | Evaluated once on 43 untouched samples |
| **Multi-Split Distribution (30 Trials)** | Macro-F1: `0.76 ± 0.07` | Range across seeds: 0.63 – 0.90 |

### Key Scientific Insights
* **Primary Discriminators**: Magnesium (`Mg`), Refractive Index (`RI`), and Barium (`Ba`) are the most informative features; Potassium (`K`) and Iron (`Fe`) contribute marginal separation signal.
* **Chemical Boundaries**: Elemental composition clearly separates major glass families (e.g. window glass vs. non-window glass achieves `>0.90` macro-F1), but window glass sub-types (float vs non-float, building vs vehicle) share overlapping chemical profiles.
* **Imbalance Management**: Utilizing `SMOTE(k_neighbors=2)` inside cross-validation folds boosts recall on rare classes (such as tableware and vehicle float glass) without causing data leakage.

---

## 📊 Dataset Overview

* **Source**: [UCI Glass Identification Dataset](https://archive.ics.uci.edu/dataset/42/glass+identification) (B. German, Home Office Forensic Science Service, UK).
* **Sample Count**: 214 total samples (213 unique after dropping 1 duplicate).
* **Features**: 9 numerical measurements (Refractive Index + 8 Oxide weight percentages).
* **Target Classes**:
  1. `Building windows (float)`
  2. `Building windows (non-float)`
  3. `Vehicle windows (float)`
  5. `Containers`
  6. `Tableware`
  7. `Headlamps`

*(Note: Class 4 `Vehicle windows (non-float)` exists in historical forensic taxonomy but has no samples present in the dataset.)*

---

## 📁 Project Directory Structure

```
glass-forensics/
├── data/
│   └── glass.csv                 # Raw dataset (auto-downloaded if missing)
├── notebooks/
│   ├── 01_eda.ipynb               # Exploratory Data Analysis & visualizations
│   └── 02_modeling.ipynb          # Model benchmarking & ablation experiments
├── src/
│   ├── __init__.py
│   ├── config.py                  # Paths, constants, feature definitions, seeds
│   ├── data_loader.py             # Dataset downloader, parser, and schema validator
│   ├── preprocess.py              # Log-scaling, ratio engineering & stratified splitting
│   ├── models.py                  # Model zoo (RandomForest, ExtraTrees, SVM, KNN, SMOTE)
│   ├── evaluate.py                # Cross-validation, metrics, confusion matrix & feature importances
│   ├── train.py                   # End-to-end training, tuning, and refitting CLI
│   └── predict.py                 # Single/batch prediction engine & range warning detector
├── models/
│   ├── glass_model.joblib         # Serialized production pipeline
│   └── metadata.json              # Model parameters, feature bounds, class labels & metrics
├── reports/
│   ├── cv_results.csv             # Model comparison cross-validation matrix
│   ├── model_comparison.png       # Macro-F1 comparison chart
│   ├── confusion_matrix.png       # Normalized confusion matrix plot
│   ├── permutation_importance.csv # Permutation importance rankings
│   └── permutation_importance.png # Permutation importance bar chart
├── tests/
│   └── test_pipeline.py           # Comprehensive pytest suite
├── app.py                         # Streamlit interactive web application
├── requirements.txt               # Project dependencies
└── README.md
```

---

## 🚀 Quick Start Guide

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/Forensic-Analysis-Glass-Identification.git
cd Forensic-Analysis-Glass-Identification

# Create & activate a Python virtual environment (optional)
python -m venv .venv
# On Windows PowerShell:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Run Automated Sanity Tests

```bash
python -m pytest
```

### 3. Launch Interactive Streamlit Web App

```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to test sample measurements and view model confidence distributions interactively.

---

## 🛠️ Model Training & CLI Usage

### Train a New Model

```bash
# Compare all models in the zoo, tune hyper-parameters, and refit on full data
python -m src.train --refit-all

# Fast training without grid search
python -m src.train --no-tune

# Train a specific classifier architecture
python -m src.train --model "SVM (RBF)"

# Include optional engineered feature ratios (e.g. Ca/Na, Mg/Al)
python -m src.train --engineer
```

### Perform Predictions via CLI

```bash
# Single sample prediction
python -m src.predict --RI 1.5165 --Na 14.38 --Mg 0 --Al 1.94 --Si 73.61 --K 0 --Ca 8.48 --Ba 1.57 --Fe 0

# Batch prediction from a CSV file
python -m src.predict --csv new_samples.csv
```

### Python API Usage

```python
from src.predict import load_model, predict

model = load_model()
sample = {
    "RI": 1.5165, "Na": 14.38, "Mg": 0.0, "Al": 1.94,
    "Si": 73.61, "K": 0.0, "Ca": 8.48, "Ba": 1.57, "Fe": 0.0
}
result = predict(model, sample)
print(result)
```

---

## 🔬 Methodology & Design Principles

1. **Zero Data Leakage**: Preprocessing transforms (StandardScaler, SMOTE) are encapsulated inside `imbalanced-learn` pipelines so that fitting occurs strictly on training folds during cross-validation.
2. **Duplicate Handling**: Identified and removed 1 duplicate row prior to train/test splitting to prevent identical samples spanning across training and test sets.
3. **Metric Selection**: Optimizes for **Macro-F1** due to heavy class imbalance (ranging from 76 building window samples down to 9 tableware samples).
4. **Extrapolation Warnings**: Inputs exceeding empirical min/max feature boundaries recorded during training automatically log extrapolation warnings.

---

## ⚠️ Scientific Limitations

* **Historical Dataset**: Based on forensic samples collected in the 1980s; chemical formulations of modern float glass and containers may differ.
* **Educational Purpose**: Designed for machine learning benchmarking and educational demonstration. Forensic legal testimony requires multi-analytical confirmation (e.g. LA-ICP-MS) and accredited forensic workflows.
