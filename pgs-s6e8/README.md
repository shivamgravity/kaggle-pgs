# Predicting Smartphone Addiction

Here, we have another interseting competition dataset. You can visit the competition page from here: [Playground Series - August 2026](https://www.kaggle.com/competitions/playground-series-s6e8/overview).

This time, it is to find *smartphone addiction*. Many of us are already showing positive signal for this.

## Evaluation Metric

**AUC-ROC:** Area under the ROC curve.

**What is AUC-ROC?**

*Receiver Operating Characterstic Area Under Curve* tell us how correct is our binary classification model.

Want you understand this concept - learn it on [`numiquo-statistics-explanation`](https://youtu.be/QBVzZBsif20?si=GLkXHHz8i1aNv3gW).

## Leaderboard & Experiments

Here we log the Public Leaderboard (LB) scores of our experiments:

### Blending Ensemble
**Kernel:** [shivamgravity/pgs-s6e8-ensemble](https://www.kaggle.com/code/shivamgravity/pgs-s6e8-ensemble)

| Version | Public LB Score | Notes |
| :--- | :--- | :--- |
| v1 | pending | Weighted Average (XGB, LGBM, CatBoost, NN) |

### XGBoost
**Kernel:** [shivamgravity/pgs-s6e8-xgboost](https://www.kaggle.com/code/shivamgravity/pgs-s6e8-xgboost)

| Version | Public LB Score | Notes |
| :--- | :--- | :--- |
| v4 | `0.96657` | Optuna Tuned, Feature Engineered, 5-Fold CV |
| v5 | `0.96635` | Deep V5 Features, Optuna Tuned, 5-Fold CV |
| v2 | `0.96598` | Feature Engineered, 5-Fold CV |
| v1 | `0.96531` | 5-Fold Stratified CV, Early Stopping |

### CatBoost
**Kernel:** [shivamgravity/pgs-s6e8-catboost](https://www.kaggle.com/code/shivamgravity/pgs-s6e8-catboost)

| Version | Public LB Score | Notes |
| :--- | :--- | :--- |
| v2 | pending | Deep V5 Features, Optuna Tuned, 5-Fold CV |
| v1 | pending | Optuna Tuned, Feature Engineered, 5-Fold CV |

### LightGBM
**Kernel:** [shivamgravity/pgs-s6e8-lightgbm](https://www.kaggle.com/code/shivamgravity/pgs-s6e8-lightgbm)

| Version | Public LB Score | Notes |
| :--- | :--- | :--- |
| v5 | `0.96625` | Deep V5 Features, Optuna Tuned, 5-Fold CV |
| v4 | `0.96608` | Optuna Tuned, Feature Engineered, 5-Fold CV |
| v2 | `0.96526` | Feature Engineered, 5-Fold CV |
| v1 | `0.96464` | 5-Fold Stratified CV, Early Stopping |

### HistGradientBoosting
**Kernel:** [shivamgravity/pgs-s6e8-hgbc](https://www.kaggle.com/code/shivamgravity/pgs-s6e8-hgbc)

| Version | Public LB Score | Notes |
| :--- | :--- | :--- |
| v3 | `0.95856` | 5-Fold Stratified CV |

### Neural Network (MLP)
**Kernel:** [shivamgravity/pgs-s6e8-neural-network](https://www.kaggle.com/code/shivamgravity/pgs-s6e8-neural-network)

| Version | Public LB Score | Notes |
| :--- | :--- | :--- |
| v2 | `0.94160` | Keras MLP, V5 Features, GPU Accelerated |

### Logistic Regression
**Kernel:** [shivamgravity/pgs-s6e8-baseline-logistic-regression](https://www.kaggle.com/code/shivamgravity/pgs-s6e8-baseline-logistic-regression)

| Version | Public LB Score | Notes |
| :--- | :--- | :--- |
| v3 | `0.91385` | Baseline Model, 5-Fold Stratified CV |
