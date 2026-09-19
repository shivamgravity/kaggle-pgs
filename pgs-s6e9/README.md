# Predicting Electric Vehicle Purchases

Hi,

This is `September, 2026`.

We have yet another interesting playground competition.

Electric vehicles does solve the problem of **carbon emissions** due to everyday vehicles running on roads.

Tesla and BYD are at the fore-front of the electric vechicle and autonomous vechile development.

What if the product is good and solves a real environmental problem, but the general public doesn't want to migrate towards new technology or product easily.

This competition addresses the very same issue and intends us to create a _prediction model_ around it.

## Evaluation Metric

The evaluation metric is **ROC-AUC**.

*Receiver Operating Characterstic Area Under Curve* tell us how correct is our binary classification model.

Want you understand this concept - learn it on [`numiquo-statistics-explanation`](https://youtu.be/QBVzZBsif20?si=GLkXHHz8i1aNv3gW).

## Leaderboard & Experiments

Here I log the Public Leaderboard (LB) scores of my experiments:

**Benchmark Model:** Light-GBM (v4)

### Baseline

**Kernel:** [shivamgravity/pgs-s6e9-baseline](https://www.kaggle.com/code/shivamgravity/pgs-s6e9-baseline)

**Benchmark version:** `v4`

| Version | Public LB Score | Notes |
|:---|:---|:---|
| v4 | `0.93738` | Logistic Regression - test_train_split |
| v5 | `0.93737` | Logistic Regression - 5 Fold CV |

### Light GBM

**Kernel:** [shivamgravity/pgs-s6e9-lightgbm](https://www.kaggle.com/code/shivamgravity/pgs-s6e9-lightgbm)

**Benchmark version:** `v4`

| Version | Public LB Score | Notes |
|:---:|:---:|:---|
| v1 | `0.94167` | 5 Fold CV |
| v2 | `0.94202` | 5 Fold CV + Optuna + Best Params Saved |
| v3 | `0.94191` | 5 Fold CV + Optuna + Best Params + All Optuna Trials Saved |
| v4 | `0.94205` | v3 + optimized params scope |
| v5 | `0.94194` | v4 + optimized params scope |
| v6 | `0.75494` | v5 - Standard Scaler |
| v10 | `0.94202` | v6 - Optuna - OHE + Native Categorical Handling + best_params_v4 |
| v13 | `0.88447` | v10 + StandardScaler |

### Catboost

**Kernel:** [shivamgravity/pgs-s6e9-catboost](https://www.kaggle.com/code/shivamgravity/pgs-s6e9-catboost)

**Benchmark version:** `v11`

| Version | Public LB Score | Notes |
|:---:|:---:|:---|
| v6 | `0.94151` | Optuna + GPU: NvidiaTeslaT4 |
| v7 | `0.94169` | v6 + narrowed params search space |
| v8 | `0.94183` | v7 + narrowed params search space |
| v9 | `0.94192` | v8 + optimized params search space |
| v10 | `0.94151` | v9 + adjusted params search space |
| v11 | `0.94195` | around v9 search space |
| v12 | `0.94152` | final optuna with focused v9 parasms |

### XG Boost

**Kernel:** [shivamgravity/pgs-s6e9-xgboost](https://www.kaggle.com/code/shivamgravity/pgs-s6e9-xgboost)

**Benchmark version:** `v5`

| Version | Public LB Score | Notes |
|:---:|:---:|:---|
| v4 | `0.94191` | Optuna + enable_categorical=True |
| v5 | `0.94194` | v4 + narrowed params search space |

### Ensemble

**Kernel:** [shivamgravity/pgs-s6e9-ensemble](https://www.kaggle.com/code/shivamgravity/pgs-s6e9-ensemble)

**Benchmark version:** `v2`, `v3`

| Version | Public LB Score | Notes |
|:---:|:---:|:---|
| v2 | `0.94205` | Mean aggregation, best_params: xgb, catboost, lgbm. |
| v3 | `0.94205` | Optimize ensemble weights via grid search |