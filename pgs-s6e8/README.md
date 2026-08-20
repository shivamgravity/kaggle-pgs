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

| Kernel | Version | Model | Public LB Score | Notes |
| :--- | :--- | :--- | :--- | :--- |
| [pgs-s6e8-xgboost](https://www.kaggle.com/code/shivamgravity/pgs-s6e8-xgboost) | v1 | **XGBoost** | `0.96531` | 5-Fold Stratified CV, Early Stopping |
| [pgs-s6e8-lightgbm](https://www.kaggle.com/code/shivamgravity/pgs-s6e8-lightgbm) | v1 | **LightGBM** | `0.96464` | 5-Fold Stratified CV, Early Stopping |
| [pgs-s6e8-hgbc](https://www.kaggle.com/code/shivamgravity/pgs-s6e8-hgbc) | v3 | **HistGradientBoosting** | `0.95856` | 5-Fold Stratified CV |
| [pgs-s6e8-baseline-logistic-regression](https://www.kaggle.com/code/shivamgravity/pgs-s6e8-baseline-logistic-regression) | v3 | **Logistic Regression** | `0.91385` | Baseline Model, 5-Fold Stratified CV |
