# Predicting Credit Card Fraud

> On Kaggle: [Playground Series Season 6 Episode 11](https://www.kaggle.com/competitions/playground-series-s6e11/overview)

Welcome to Episode 11 of the 2026 Kaggle Playground Series! In this challenge, we are tasked with detecting fraudulent credit card transactions from tabular transaction features.

Fraud detection is a classic machine learning problem characterized by:
- **Severe Class Imbalance**: Fraudulent transactions account for only ~1.67% of all observations (11,689 positive vs. 688,271 negative transactions in the training set).
- **Subtle Discriminative Signals**: Legitimate and fraudulent behaviors overlap significantly in feature space.
- **High Consequence of Errors**: False negatives mean financial loss for institutions, while false positives cause friction for genuine cardholders.

The dataset is synthetically generated from real-world credit card patterns inspired by the [*Credit Card Fraud Detection 2026*](https://www.kaggle.com/datasets/uditjain13/credit-card-fraud-detection-2026) dataset.

---

## 📊 Evaluation Metric

The competition is evaluated on **ROC-AUC (Area Under the Receiver Operating Characteristic Curve)**.

### Why ROC-AUC?
- Unlike standard accuracy—which is misleadingly high (over 98.3%) even if a model naively predicts all non-fraud—ROC-AUC evaluates how well the model **ranks** transactions by fraud risk across all possible probability thresholds.
- ROC-AUC measures the probability that a randomly chosen fraudulent transaction receives a higher predicted risk score than a randomly chosen genuine transaction.
- A score of `0.5` represents random guessing, while `1.0` represents perfect separation.

In addition to ROC-AUC, we monitor **PR-AUC (Precision-Recall AUC / Average Precision)** during validation because PR-AUC focuses specifically on the minority positive class without being inflated by true negatives.

---

## 🔬 Validation Strategy

To prevent data leakage and provide consistent out-of-fold (OOF) evaluation that mirrors the test set:
- **5-Fold Stratified Cross-Validation (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`)**: Preserves the ~1.67% fraud proportion evenly across all 5 folds.
- **Strict Pipeline Transformation**: Preprocessors (scalers, encoders) are fitted strictly within training folds, never on validation or full data.
- **OOF Ensembling**: Predictions on the test set are averaged across all 5 trained fold models.

---

## 🏆 Leaderboard & Experiments

**Benchmark Model:** LightGBM (v1) - `0.94755`

### LightGBM (Shallow Trees)

**Kernel:** [shivamgravity/pgs-s6e11-lightgbm](https://www.kaggle.com/code/shivamgravity/pgs-s6e11-lightgbm)

| Version | 5-Fold OOF ROC-AUC | PR-AUC | Public LB | Notes |
|:---:|:---:|:---:|:---:|:---|
| v1 | `0.94700` | `0.51074` | `0.94755` | 5-Fold Stratified CV, shallow trees (`num_leaves=4`, `max_depth=3`), lr=0.05 |

### XGBoost (GPU Hist)

**Kernel:** [shivamgravity/pgs-s6e11-xgboost](https://www.kaggle.com/code/shivamgravity/pgs-s6e11-xgboost)

| Version | 5-Fold OOF ROC-AUC | PR-AUC | Public LB | Notes |
|:---:|:---:|:---:|:---:|:---|
| v1 | *Running on Kaggle* | *Running* | *Pending* | 5-Fold Stratified CV, GPU accelerated (`tree_method='hist'`), shallow depth (`max_depth=3`), lr=0.05 |

### CatBoost (GPU Symmetric Trees)

**Kernel:** [shivamgravity/pgs-s6e11-catboost](https://www.kaggle.com/code/shivamgravity/pgs-s6e11-catboost)

| Version | 5-Fold OOF ROC-AUC | PR-AUC | Public LB | Notes |
|:---:|:---:|:---:|:---:|:---|
| v1 | *Running on Kaggle* | *Running* | *Pending* | 5-Fold Stratified CV, GPU accelerated (`task_type='GPU'`), `depth=4`, native `cat_features` |

### Baseline (Logistic Regression)

**Kernel:** [shivamgravity/pgs-s6e11-baseline](https://www.kaggle.com/code/shivamgravity/pgs-s6e11-baseline)

| Version | 5-Fold OOF ROC-AUC | PR-AUC | Public LB | Notes |
|:---:|:---:|:---:|:---:|:---|
| v1 | `0.94190` | `0.48966` | `0.94257` | 5-Fold Stratified CV (mean: 0.94193 ± 0.00343), StandardScaler + OneHotEncoder |

---

## 📁 Directory Layout

```text
pgs-s6e11/
├── dataset/
│   ├── train.csv              # 699,960 rows, 26 features
│   ├── test.csv               # 299,984 rows, 25 features
│   └── sample_submission.csv  # 299,984 rows
├── code/
│   ├── baseline/
│   │   ├── kernel-metadata.json
│   │   └── model.ipynb
│   ├── eda/
│   │   ├── kernel-metadata.json
│   │   └── eda.ipynb
│   └── lightgbm/
│       ├── kernel-metadata.json
│       └── model.ipynb
├── submission/
│   └── baseline_submission.csv
└── README.md
```
