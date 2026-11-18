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

**Benchmark Model:** Grand Apex Mega-Ensemble (WideLM + GLM-Boost MLP + Deep ResNet + WideTree + LM7) - **`0.952597`** 5-Fold OOF CV (PR-AUC: **`0.543316`**, Public LB: **`0.95292`** — Rank #398 / Top 18%)

### Grand Apex Mega-Ensemble (WideLM + GLM-Boost MLP + Deep Tabular ResNet + WideTree + LM7 GPT-2)

| Version | 5-Fold OOF ROC-AUC | PR-AUC | Public LB | Notes |
|:---:|:---:|:---:|:---:|:---|
| v1 | **`0.952597`** | **`0.543316`** | **`0.95292`** | 👑 **ALL-TIME PEAK RECORD (Rank #398 / Top 18%)!** Greedy forward selection & Nelder-Mead simplex combining WideLM (88.48%) + GLM-boost MLP w128 (4.85%) + PyTorch Tabular ResNet (3.69%) + WideTree (2.50%) + Compact LM7 GPT-2 (0.49%). Total neural diversity weight reaches 8.54%, breaking all previous ceilings and propelling our team directly into the top 18%! |

### Master Mega-Ensemble (WideLM + Deep Tabular ResNet + Triple Joint TE + GPT-2)

| Version | 5-Fold OOF ROC-AUC | PR-AUC | Public LB | Notes |
|:---:|:---:|:---:|:---:|:---|
| v1 | **`0.952592`** | **`0.543419`** | **`0.95290`** | 🏆 **Historic 5-Fold Performance.** Optimal simplex blend combining 95.64% WideLM Stack (201 members) + 3.88% PyTorch Tabular ResNet (entity embeddings for digits/cents) + 0.24% Triple GLM Joint TE + 0.20% Compact LM7 GPT-2. Proves that continuous neural embeddings inject genuine orthogonal signal over 201 tree and language models! |

### 10-Fold Stratified Master GLM Residual Booster (74 Digit Keys + Velocity + Joint Pairs)

**Kernel:** [shivamgravity/pgs-s6e11-glm-10fold-boost](https://www.kaggle.com/code/shivamgravity/pgs-s6e11-glm-10fold-boost)

| Version | 10-Fold OOF ROC-AUC | PR-AUC | Public LB | Notes |
|:---:|:---:|:---:|:---:|:---|
| v1 | **`0.950647`** | **`0.531152`** | **`0.95034`** | ⚡ **10-Fold Leak-Free Residual Booster.** 100 Laplace-smoothed target-encoded multi-scale digit & 2-way joint keys (`mc_x_cents`, `amt10_x_bal10`, `mc_x_foreign`, etc.) + GLM base linear model (`0.949757` CV) + LightGBM 4-leaf booster (`0.950647` CV) + XGBoost depth-2 booster (`0.950545` CV). Executed remotely on Kaggle cloud (30 GB RAM). Adds critical 3.73% direct weight in the master mega-ensemble! |

### Multi-Seed Dual-Neural Bagged Architecture (ResNet + FT-Transformer)

**Kernel:** [shivamgravity/pgs-s6e11-dual-neural-multiseed](https://www.kaggle.com/code/shivamgravity/pgs-s6e11-dual-neural-multiseed)

| Version | 5-Fold OOF ROC-AUC | PR-AUC | Public LB | Notes |
|:---:|:---:|:---:|:---:|:---|
| v1 | **`0.943441`** | **`0.482984`** | **`0.94353`** | 🧠 **Multi-Seed Neural Ensemble.** 6 distinct neural models: 3-seed Tabular ResNet (`0.941584` CV) + 3-seed FT-Transformer with self-attention (`0.943094` CV) trained across seeds 42, 100, 2026 with FP16 AMP. Generalizes well on the public test set (`0.94353` LB) and injects crucial non-tree representations. |

### Feature Tokenizer Transformer (FT-Transformer - PyTorch)

**Kernel:** [shivamgravity/pgs-s6e11-tabular-transformer](https://www.kaggle.com/code/shivamgravity/pgs-s6e11-tabular-transformer)

| Version | 5-Fold OOF ROC-AUC | PR-AUC | Public LB | Notes |
|:---:|:---:|:---:|:---:|:---|
| v1 | **`0.942509`** | **`0.484291`** | *Queued Priority #4* | Feature Tokenizer Transformer with 15 continuous feature projections + 21 learned discrete token embeddings into $d=64$ token space + 3 Pre-LN Transformer blocks + multi-head self-attention + `[CLS]` token head. Outperformed standalone Tabular ResNet by +0.00136 AUC and +0.0155 PR-AUC! Blending 60% FT-Transformer + 40% Tabular ResNet achieves **0.943585** Dual-Neural CV! |

### Grand Champion Meta-Stacker (52 Diverse Models: GPT-2 LMs + EBMs + GBDT Boosters + Deep ResNet)

| Version | 5-Fold OOF ROC-AUC | PR-AUC | Public LB | Notes |
|:---:|:---:|:---:|:---:|:---|
| v1 | **`0.951902`** | **`0.539986`** | **`0.95240`** | 🥈 **Rank #481 / Top 23%!** Regularized Ridge Logistic Regression ($C=0.005$) on logit-ranks of 52 diverse models on identical 5-folds. Top positive contributors: `megalm_gpt2` (+0.383), `lm7_gpt2` (+0.222), `bank_gb_ohk` (+0.096), `bank_gb_bag5` (+0.086), `ebm_i25_bag4` (+0.038). |

### GPT-2 Language Model + GBDT Boosters + Tabular ResNet Hybrid

| Version | 5-Fold OOF ROC-AUC | PR-AUC | Public LB | Notes |
|:---:|:---:|:---:|:---:|:---|
| v1 | **`0.951631`** | **`0.539003`** | `0.95218` | 49.5% GPT-2 Language Models (megalm + lm7) + 47.2% GLM-offset GBDT Boosters (Joint TE + Dual Margin + 1D) + 3.3% Tabular Deep ResNet. Broke past 0.9520 barrier! |

### Velocity & Burst Interaction Augmented Triple GLM Residual Booster

**Kernel:** [shivamgravity/pgs-s6e11-velocity-burst-glm-boost](https://www.kaggle.com/code/shivamgravity/pgs-s6e11-velocity-burst-glm-boost)

| Version | 5-Fold OOF ROC-AUC | PR-AUC | Public LB | Notes |
|:---:|:---:|:---:|:---:|:---|
| v1 | `0.950652` | `0.531390` | `0.95090` | 10 new velocity & burst features (`spend_velocity`, `travel_speed_kmh`, `balance_drain_ratio`, `cvv_x_risk`, `fraud_risk_index`, `amount_to_cat_median`) + 1D/2D Joint TE + GLM offset for LGB, XGB, CatBoost. |

### Tabular Deep ResNet with Learned Entity Embeddings (PyTorch)

**Kernel:** [shivamgravity/pgs-s6e11-tabular-nn-boost](https://www.kaggle.com/code/shivamgravity/pgs-s6e11-tabular-nn-boost)

| Version | 5-Fold OOF ROC-AUC | PR-AUC | Public LB | Notes |
|:---:|:---:|:---:|:---:|:---|
| v1 | `0.941145` | `0.468793` | - | Pre-activation ResNet blocks + learned entity embeddings for `cents` (16D), `first_decimal`, `last_digit`, and categoricals. Provides orthogonal non-tree inductive bias (Spearman rank correlation 0.898 to trees). |

### Master Tri-Paradigm Blend

| Version | 5-Fold OOF ROC-AUC | PR-AUC | Public LB | Notes |
|:---:|:---:|:---:|:---:|:---|
| v1 | **`0.95073`** | **`0.53247`** | **`0.95152`** | 88% Tri-Gen TE Ensemble (1D + DualMargin + TripleJoint) + 12% Raw Continuous Booster. All-time highest CV & PR-AUC. |

### Cross-Paradigm Optimal Rank Blend

| Version | 5-Fold OOF ROC-AUC | PR-AUC | Public LB | Notes |
|:---:|:---:|:---:|:---:|:---|
| v1 | - | - | **`0.95152`** | 85% Digit-Key TE Booster Blend + 15% Dual GLM Raw Feature Booster. Spearman corr 0.9657. Advanced to **Rank #638**! |

### Triple GLM Residual Booster (LightGBM + XGBoost + CatBoost + 2-Way Joint TE)

**Kernel:** [shivamgravity/pgs-s6e11-triple-glm-joint-te-boost](https://www.kaggle.com/code/shivamgravity/pgs-s6e11-triple-glm-joint-te-boost)

| Version | 5-Fold OOF ROC-AUC | PR-AUC | Public LB | Notes |
|:---:|:---:|:---:|:---:|:---|
| v1 | **`0.95064`** | **`0.53219`** | `0.95138` | 2-Way Joint Interaction Keys + Multi-Scale Smoothing + Triple GBDT (29.5% XGB + 57.6% LGB + 13.0% CatBoost). All-time record 0.53219 OOF PR-AUC. |

### True Dual-Margin GLM Residual Booster (LightGBM + XGBoost)

**Kernel:** [shivamgravity/pgs-s6e11-dual-glm-digit-te-boost](https://www.kaggle.com/code/shivamgravity/pgs-s6e11-dual-glm-digit-te-boost)

| Version | 5-Fold OOF ROC-AUC | PR-AUC | Public LB | Notes |
|:---:|:---:|:---:|:---:|:---|
| v1 | **`0.95064`** | **`0.53180`** | `0.95130` | Dual margin offsets (init_score for LGBM + base_margin for XGBoost) + COUNT_COLS in trees. Record 0.53180 OOF PR-AUC. |

### Digit-Key Target Encoding & GLM Residual Booster Blend

**Kernel:** [shivamgravity/pgs-s6e11-glm-digit-te-boost](https://www.kaggle.com/code/shivamgravity/pgs-s6e11-glm-digit-te-boost)

| Version | 5-Fold OOF ROC-AUC | PR-AUC | Public LB | Notes |
|:---:|:---:|:---:|:---:|:---|
| v1 | **`0.95062`** | **`0.53154`** | **`0.95149`** | In-fold smoothed digit target encoding + GLM margin offset + shallow trees (65% XGB-GLM + 35% LGBM 3-leaves). Jumped to **Rank #645**! |

### Dual GLM Residual Booster Blend

**Kernel:** [shivamgravity/pgs-s6e11-dual-glm-blend](https://www.kaggle.com/code/shivamgravity/pgs-s6e11-dual-glm-blend)

| Version | 5-Fold OOF ROC-AUC | PR-AUC | Public LB | Notes |
|:---:|:---:|:---:|:---:|:---|
| v1 | **`0.94745`** | **`0.51534`** | **`0.94792`** | Optimal Rank Blend (29% LightGBM GLM + 71% XGBoost GLM), all-time record PR-AUC |

### GLM Residual Boosting (LightGBM)

**Kernel:** [shivamgravity/pgs-s6e11-glm-boost](https://www.kaggle.com/code/shivamgravity/pgs-s6e11-glm-boost)

| Version | 5-Fold OOF ROC-AUC | PR-AUC | Public LB | Notes |
|:---:|:---:|:---:|:---:|:---|
| v1 | `0.94708` | `0.51306` | **`0.94782`** | Logistic Regression log-odds margin ($z_i$) as LightGBM `init_score`, `num_leaves=4`, `max_depth=3`, lr=0.03 |

### XGBoost GLM Residual Boosting

**Kernel:** [shivamgravity/pgs-s6e11-xgb-glm-boost](https://www.kaggle.com/code/shivamgravity/pgs-s6e11-xgb-glm-boost)

| Version | 5-Fold OOF ROC-AUC | PR-AUC | Public LB | Notes |
|:---:|:---:|:---:|:---:|:---|
| v1 | `0.94739` | `0.51504` | `0.94780` | Logistic Regression log-odds margin ($z_i$) as XGBoost `base_margin`, GPU hist, `max_depth=3`, lr=0.03 |

### Multi-Model Ensemble & Stacking

**Kernel:** [shivamgravity/pgs-s6e11-ensemble](https://www.kaggle.com/code/shivamgravity/pgs-s6e11-ensemble)

| Version | 5-Fold OOF ROC-AUC | PR-AUC | Public LB | Notes |
|:---:|:---:|:---:|:---:|:---|
| v1 | `0.94735` | `0.51095` | `0.94768` | 4-Model 5-Fold CV (LGBM + XGB + CatBoost + LR), Optimal Rank Blend (LGBM 44.3% + XGB 55.7%) |

### Feature Engineering Ladder

**Kernel:** [shivamgravity/pgs-s6e11-feature-engineering](https://www.kaggle.com/code/shivamgravity/pgs-s6e11-feature-engineering)

| Version | 5-Fold OOF ROC-AUC | PR-AUC | Public LB | Notes |
|:---:|:---:|:---:|:---:|:---|
| v2 | `0.94567` | `0.50321` | `0.94673` | Block A-D (Cents, Benford, Risk flags, In-Fold TE). Confirmed feature dilution effect; CV and LB track closely |

### LightGBM (Shallow Trees)

**Kernel:** [shivamgravity/pgs-s6e11-lightgbm](https://www.kaggle.com/code/shivamgravity/pgs-s6e11-lightgbm)

| Version | 5-Fold OOF ROC-AUC | PR-AUC | Public LB | Notes |
|:---:|:---:|:---:|:---:|:---|
| v1 | `0.94700` | `0.51074` | `0.94755` | 5-Fold Stratified CV, shallow trees (`num_leaves=4`, `max_depth=3`), lr=0.05 |

### XGBoost (GPU Hist)

**Kernel:** [shivamgravity/pgs-s6e11-xgboost](https://www.kaggle.com/code/shivamgravity/pgs-s6e11-xgboost)

| Version | 5-Fold OOF ROC-AUC | PR-AUC | Public LB | Notes |
|:---:|:---:|:---:|:---:|:---|
| v1 | `0.94715` | `0.51105` | `0.94753` | 5-Fold Stratified CV, GPU accelerated (`tree_method='hist'`), shallow depth (`max_depth=3`), lr=0.05 |

### CatBoost (GPU Symmetric Trees)

**Kernel:** [shivamgravity/pgs-s6e11-catboost](https://www.kaggle.com/code/shivamgravity/pgs-s6e11-catboost)

| Version | 5-Fold OOF ROC-AUC | PR-AUC | Public LB | Notes |
|:---:|:---:|:---:|:---:|:---|
| v1 | `0.94467` | `0.50388` | `0.94565` | 5-Fold Stratified CV, GPU accelerated (`task_type='GPU'`), `depth=4`, native `cat_features` |

### Baseline (Logistic Regression)

**Kernel:** [shivamgravity/pgs-s6e11-baseline](https://www.kaggle.com/code/shivamgravity/pgs-s6e11-baseline)

| Version | 5-Fold OOF ROC-AUC | PR-AUC | Public LB | Notes |
|:---:|:---:|:---:|:---:|:---|
| v1 | `0.94190` | `0.48966` | `0.94257` | 5-Fold Stratified CV (mean: 0.94193 ± 0.00343), StandardScaler + OneHotEncoder |


