import nbformat as nbf
from pathlib import Path
import json

nb = nbf.v4.new_notebook()
cells = []

# Cell 0: Header & Strategy
cells.append(nbf.v4.new_markdown_cell('''# ⚡ PGS-S6E11: Velocity & Burst Interaction Augmented Triple GLM Residual Booster
### Kaggle Playground Series Season 6 Episode 11: Credit Card Fraud Detection

**Key Innovations in this Architecture:**
1. **Transaction Velocity & Burst Dynamics**:
   - Spending velocity: `amount_usd / (hours_since_last_txn + 0.05)`
   - Impossible travel velocity: `distance_from_home_km / (hours_since_last_txn + 0.05)`
   - Balance drain ratio: `amount_usd / (account_balance_usd + 1.0)`
   - 24-Hour spending volume: `amount_usd * txn_count_last_24h`
   - Velocity-weighted exposure: `velocity_score * amount_usd`
   - CVV retry risk amplification: `cvv_retry_count * merchant_risk_score`
   - Total risk flags & amount interaction: `total_flags * amount_usd`
   - Merchant category relative ratio: `amount_usd / (cat_median + 1.0)`
   - Category z-score: `(amount_usd - cat_mean) / cat_std`
   - Fraud risk index: `total_flags * (velocity_score + 1.0) * (cvv_retry_count + 1)`
2. **1D Digit & 2-Way Joint Key Target Encoding**:
   - Exact digit keys (`cents`, `first_decimal`, `last_digit`, `int_div10`, `lead1`, `lead2`)
   - 12 Cross-feature joint interaction keys (`merchant_category x cents`, `amount_div10 x balance_div10`, etc.)
   - Strict in-fold Bayesian smoothed encoding ($m=10$ and $m=25$) to prevent target leakage.
3. **True GLM Margin Offset ($z_i = \mathbf{w}^T \mathbf{x}_i + b$)**:
   - Scaled Quantile-Transformed continuous + velocity features + target encodings.
4. **Triple Residual Boosters**:
   - **LightGBM** (`init_score = z`)
   - **XGBoost GPU Hist** (`base_margin = z`)
   - **CatBoost GPU** (`baseline = z`)
5. **Optimal Simplex Blending via Nelder-Mead Optimization**.
'''))

# Cell 1: Imports & Setup
cells.append(nbf.v4.new_code_cell('''import os
import gc
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from scipy.optimize import minimize
from scipy.special import expit, logit

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, QuantileTransformer

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool

warnings.filterwarnings("ignore")
USE_GPU = True
print(f"CUDA Available: {xgb.__version__}, LightGBM: {lgb.__version__}")
'''))

# Cell 2: Config
cells.append(nbf.v4.new_code_cell('''class Config:
    SEED = 42
    N_FOLDS = 5
    N_SPLITS = 5
    SMOOTH_SHARP = 10.0
    SMOOTH_BROAD = 25.0
    ES_ROUNDS = 250
    
    # Path resolution
    if os.path.exists("/kaggle/input/playground-series-s6e11"):
        DATA_DIR = Path("/kaggle/input/playground-series-s6e11")
        OUTPUT_DIR = Path(".")
    elif os.path.exists("/kaggle/input/competitions/playground-series-s6e11"):
        DATA_DIR = Path("/kaggle/input/competitions/playground-series-s6e11")
        OUTPUT_DIR = Path(".")
    else:
        DATA_DIR = Path("../../dataset")
        if not (DATA_DIR / "train.csv").exists():
            DATA_DIR = Path("pgs-s6e11/dataset")
        OUTPUT_DIR = Path(".")

print(f"Config initialized. Folds: {Config.N_FOLDS}, Data: {Config.DATA_DIR}")
'''))

# Cell 3: Data Loading & Velocity Feature Engineering
cells.append(nbf.v4.new_code_cell('''train_df = pd.read_csv(Config.DATA_DIR / "train.csv")
test_df = pd.read_csv(Config.DATA_DIR / "test.csv")
sample_sub = pd.read_csv(Config.DATA_DIR / "sample_submission.csv")

TARGET, ID_COL = "is_fraud", "id"
FLAG_COLS = list(train_df.select_dtypes(include="bool").columns)

for df in (train_df, test_df):
    df[FLAG_COLS] = df[FLAG_COLS].astype("int8")

y = train_df[TARGET].astype(int).values

# --- FEATURE ENGINEERING: VELOCITY & BURST DYNAMICS ---
print("Engineering Velocity & Burst features...")
for df in (train_df, test_df):
    # 1. Velocity and impossible travel
    df["spend_velocity"] = df["amount_usd"] / (df["hours_since_last_txn"] + 0.05)
    df["travel_speed_kmh"] = df["distance_from_home_km"] / (df["hours_since_last_txn"] + 0.05)
    
    # 2. Financial strain and volume
    df["balance_drain_ratio"] = df["amount_usd"] / (df["account_balance_usd"] + 1.0)
    df["amount_24h_volume"] = df["amount_usd"] * df["txn_count_last_24h"]
    df["velocity_x_amount"] = df["velocity_score"] * df["amount_usd"]
    
    # 3. Risk flag interactions
    df["cvv_x_risk"] = df["cvv_retry_count"] * df["merchant_risk_score"]
    df["total_flags"] = df[FLAG_COLS].sum(axis=1)
    df["flag_x_amount"] = df["total_flags"] * df["amount_usd"]
    df["fraud_risk_index"] = df["total_flags"] * (df["velocity_score"] + 1.0) * (df["cvv_retry_count"] + 1)

# 4. Group Aggregation: Category relative amounts
cat_medians = train_df.groupby("merchant_category")["amount_usd"].median()
cat_means = train_df.groupby("merchant_category")["amount_usd"].mean()
cat_stds = train_df.groupby("merchant_category")["amount_usd"].std()

for df in (train_df, test_df):
    df["amount_to_cat_median"] = df["amount_usd"] / (df["merchant_category"].map(cat_medians) + 1.0)
    df["amount_cat_zscore"] = (df["amount_usd"] - df["merchant_category"].map(cat_means)) / (df["merchant_category"].map(cat_stds) + 1e-5)

VELOCITY_COLS = [
    "spend_velocity", "travel_speed_kmh", "balance_drain_ratio",
    "amount_24h_volume", "velocity_x_amount", "cvv_x_risk",
    "total_flags", "flag_x_amount", "fraud_risk_index",
    "amount_to_cat_median", "amount_cat_zscore"
]

BASE_FEATURES = [c for c in test_df.columns if c not in [ID_COL] + VELOCITY_COLS]
CAT_COLS = [c for c in BASE_FEATURES if not pd.api.types.is_numeric_dtype(train_df[c])]
NUM_COLS = [c for c in BASE_FEATURES if c not in CAT_COLS + FLAG_COLS]

for c in CAT_COLS:
    train_df[c] = train_df[c].astype(str)
    test_df[c] = test_df[c].astype(str)

print(f"Loaded: {len(train_df):,} Train rows, {len(test_df):,} Test rows.")
print(f"Base Num: {len(NUM_COLS)}, Velocity/Burst: {len(VELOCITY_COLS)}, Flags: {len(FLAG_COLS)}, Cats: {len(CAT_COLS)}")
'''))

# Cell 4: Digit Keys, Frequency Counts & 2-Way Joint Keys
cells.append(nbf.v4.new_code_cell('''DECIMALS = {
    "amount_usd": 2, "hours_since_last_txn": 2, "txn_count_last_24h": 0,
    "distance_from_home_km": 2, "card_age_months": 0, "customer_age": 0,
    "account_balance_usd": 2, "cvv_retry_count": 0, "velocity_score": 1,
    "time_of_day_hour": 0, "day_of_week": 0, "merchant_risk_score": 1,
    "prior_disputes": 0,
}

# Add precision counts for high-impact numeric features
train_df["cents"] = np.round((train_df["amount_usd"] % 1) * 100).astype(int)
test_df["cents"] = np.round((test_df["amount_usd"] % 1) * 100).astype(int)

train_df["amount_div10"] = (train_df["amount_usd"] // 10).astype(int)
test_df["amount_div10"] = (test_df["amount_usd"] // 10).astype(int)

train_df["balance_div10"] = (train_df["account_balance_usd"] // 10).astype(int)
test_df["balance_div10"] = (test_df["account_balance_usd"] // 10).astype(int)

CENT_COLS = ["cents", "amount_div10", "balance_div10"]

# Add Frequency Count features (Synthetic Artifact Multipliers)
both_raw = pd.concat([train_df[CENT_COLS], test_df[CENT_COLS]], ignore_index=True)
for c in CENT_COLS:
    counts = both_raw[c].value_counts()
    train_df[f"{c}_cnt"] = train_df[c].map(counts).astype(np.int32)
    test_df[f"{c}_cnt"] = test_df[c].map(counts).astype(np.int32)

COUNT_COLS = [f"{c}_cnt" for c in CENT_COLS]
del both_raw; gc.collect()
'''))

# Cell 5: Construction of 1D and 2-Way Joint Keys
cells.append(nbf.v4.new_code_cell('''def digit_keys(x, d):
    n = np.rint(x * 10 ** d).astype(np.int64)
    a = np.abs(n)
    ip = a // 10 ** d
    nd = np.where(ip > 0, np.floor(np.log10(np.maximum(ip, 1))).astype(np.int64) + 1, 0)
    keys = {"value": n}
    if d >= 1:
        keys["last_digit"] = a % 10
        keys["first_decimal"] = (a // 10 ** (d - 1)) % 10
    if d >= 2:
        keys["cents"] = a % 100
        keys["int_10x"] = n // 10 ** (d - 1)
    keys["int"] = n // 10 ** d
    keys["int_div10"] = n // 10 ** (d + 1)
    keys["lead1"] = nd * 10 + ip // 10 ** np.maximum(nd - 1, 0)
    keys["lead2"] = nd * 100 + ip // 10 ** np.maximum(nd - 2, 0)
    return keys

both = pd.concat([train_df, test_df], ignore_index=True)
KEY_CODES, KEY_K = {}, {}

# 1. 1D Digit & Value Keys
for c in NUM_COLS:
    x = np.nan_to_num(both[c].values.astype(float), nan=-1e9)
    kept = []
    for kname, vals in digit_keys(x, DECIMALS[c]).items():
        codes, uniques = pd.factorize(vals)
        if len(uniques) < 2 or any(np.array_equal(codes, KEY_CODES[k]) for k in kept):
            continue
        name = f"{c}__{kname}"
        KEY_CODES[name], KEY_K[name] = codes.astype(np.int32), len(uniques)
        kept.append(name)

for c in CAT_COLS:
    codes, uniques = pd.factorize(both[c].astype(str))
    name = f"{c}__value"
    KEY_CODES[name], KEY_K[name] = codes.astype(np.int32), len(uniques)

# 2. 2-Way Joint Keys (Crucial Interaction Pairs)
cents_str = (np.abs(np.rint(both["amount_usd"].values * 100)).astype(np.int64) % 100).astype(str)
amt_div10_str = (np.abs(np.rint(both["amount_usd"].values)) // 10).astype(np.int64).astype(str)
bal_div10_str = (np.abs(np.rint(both["account_balance_usd"].values)) // 10).astype(np.int64).astype(str)

joint_pairs = {
    "mc_x_cents": both["merchant_category"] + "_" + cents_str,
    "mc_x_foreign": both["merchant_category"] + "_" + both["is_foreign_transaction"].astype(str),
    "mc_x_scam": both["merchant_category"] + "_" + both["is_ai_generated_scam_attempt"].astype(str),
    "amt10_x_bal10": amt_div10_str + "_" + bal_div10_str,
    "auth_x_dev": both["auth_method"] + "_" + both["device_type"],
    "chan_x_dev": both["channel"] + "_" + both["device_type"],
    "ip_x_foreign": both["ip_country_mismatch"].astype(str) + "_" + both["is_foreign_transaction"].astype(str),
    "vpn_x_ip": both["used_vpn"].astype(str) + "_" + both["ip_country_mismatch"].astype(str),
    "hour_x_dow": both["time_of_day_hour"].astype(str) + "_" + both["day_of_week"].astype(str),
    "card_x_chan": both["card_type"] + "_" + both["channel"],
    "mc_x_hour": both["merchant_category"] + "_" + both["time_of_day_hour"].astype(str),
    "cents_x_scam": cents_str + "_" + both["is_ai_generated_scam_attempt"].astype(str),
}

for kname, s in joint_pairs.items():
    codes, uniques = pd.factorize(s)
    KEY_CODES[f"joint__{kname}"], KEY_K[f"joint__{kname}"] = codes.astype(np.int32), len(uniques)

del both; gc.collect()

KEY_NAMES = list(KEY_CODES)
TE_COLS = [f"te_{k}" for k in KEY_NAMES]

# Shared category encoding for GBDTs
for c in CAT_COLS:
    cats = sorted(set(train_df[c]) | set(test_df[c]))
    train_df[c] = pd.Categorical(train_df[c], categories=cats)
    test_df[c] = pd.Categorical(test_df[c], categories=cats)

TREE_FEATS = BASE_FEATURES + VELOCITY_COLS + CENT_COLS + COUNT_COLS + TE_COLS
RAW_ALL = BASE_FEATURES + VELOCITY_COLS + CENT_COLS + COUNT_COLS

print(f"Total encoded keys: {len(KEY_NAMES)} (including {len(joint_pairs)} 2-way joint keys)")
print(f"Total features for tree models: {len(TREE_FEATS)}")
'''))

# Cell 6: In-Fold Bayesian Target Encoding Pipeline
cells.append(nbf.v4.new_code_cell('''N_TR = len(train_df)
skf = StratifiedKFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)
folds = np.zeros(N_TR, dtype=int)
for f, (_, va) in enumerate(skf.split(train_df, y)):
    folds[va] = f

GROUPS = np.concatenate([folds, np.full(len(test_df), -1)])

def target_encode(stat_rows, apply_rows, yv, smooth=Config.SMOOTH_SHARP, exclude_own_fold=True):
    ys = yv[stat_rows].astype(float)
    prior = ys.mean()
    G = GROUPS
    g_stat, g_app = G[stat_rows], G[apply_rows]
    fold_ids = np.unique(g_stat) if exclude_own_fold else []
    out = np.empty((len(apply_rows), len(KEY_NAMES)), dtype=np.float32)
    
    for j, k in enumerate(KEY_NAMES):
        codes, K = KEY_CODES[k], KEY_K[k]
        cs, ca = codes[stat_rows], codes[apply_rows]
        s = np.bincount(cs, weights=ys, minlength=K)
        n = np.bincount(cs, minlength=K).astype(float)
        out[:, j] = (s[ca] + prior * smooth) / (n[ca] + smooth)
        
        for g in fold_ids:
            m_app = g_app == g
            if not m_app.any():
                continue
            m_st = g_stat == g
            sg = np.bincount(cs[m_st], weights=ys[m_st], minlength=K)
            ng = np.bincount(cs[m_st], minlength=K)
            cg = ca[m_app]
            out[m_app, j] = (s[cg] - sg[cg] + prior * smooth) / (n[cg] - ng[cg] + smooth)
            
    return out

def enc_frame(df, rows, block):
    part = df[RAW_ALL] if rows is None else df[RAW_ALL].iloc[rows]
    return pd.concat([part.reset_index(drop=True), pd.DataFrame(block, columns=TE_COLS)], axis=1)

def fold_frames(f):
    tr_idx, va_idx = np.where(folds != f)[0], np.where(folds == f)[0]
    te = target_encode(tr_idx, np.concatenate([tr_idx, va_idx, N_TR + np.arange(len(test_df))]), y)
    a, b = len(tr_idx), len(tr_idx) + len(va_idx)
    return (tr_idx, va_idx,
            enc_frame(train_df, tr_idx, te[:a]),
            enc_frame(train_df, va_idx, te[a:b]),
            enc_frame(test_df, None, te[b:]))

print("Target encoding pipeline with Velocity features ready.")
'''))

# Cell 7: GLM Hyperplane & GBDT Learners
cells.append(nbf.v4.new_code_cell('''def glm_matrix(X, qt=None, sc=None):
    p = np.clip(X[TE_COLS].values, 1e-4, 1 - 1e-4)
    num = X[NUM_COLS + VELOCITY_COLS + COUNT_COLS].values.astype(float)
    if qt is None:
        qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal", subsample=200_000, random_state=Config.SEED).fit(num)
    M = np.hstack([np.log(p / (1 - p)), qt.transform(num), X[FLAG_COLS].values.astype(float)])
    if sc is None:
        sc = StandardScaler().fit(M)
    return np.nan_to_num(sc.transform(M)), qt, sc

def glm_fit(X, yv):
    M, qt, sc = glm_matrix(X)
    m = LogisticRegression(C=0.5, max_iter=500, random_state=Config.SEED).fit(M, yv)
    return lambda Z: m.decision_function(glm_matrix(Z, qt, sc)[0])

# 1. LightGBM Residual Booster
TE_LGB_PARAMS = dict(
    objective="binary", metric="auc", boosting_type="gbdt", learning_rate=0.03,
    num_leaves=4, max_depth=3, min_child_samples=100,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0,
    random_state=Config.SEED, n_jobs=-1, verbose=-1
)

def lgb_fit(X, yv, X_es=None, y_es=None, margin=None, margin_es=None, n=None):
    trn_data = lgb.Dataset(X[TREE_FEATS], label=yv, init_score=margin)
    if X_es is None:
        bst = lgb.train(TE_LGB_PARAMS, trn_data, num_boost_round=n)
        return bst, n
    val_data = lgb.Dataset(X_es[TREE_FEATS], label=y_es, init_score=margin_es, reference=trn_data)
    bst = lgb.train(TE_LGB_PARAMS, trn_data, num_boost_round=10000, valid_sets=[val_data],
                    callbacks=[lgb.early_stopping(Config.ES_ROUNDS, verbose=False)])
    return bst, bst.best_iteration

def lgb_pred(bst, X, n=None, margin=None):
    raw = bst.predict(X[TREE_FEATS], num_iteration=n, raw_score=True)
    return expit(margin + raw) if margin is not None else expit(raw)

# 2. XGBoost Residual Booster
TE_XGB_PARAMS = dict(
    objective="binary:logistic", eval_metric="auc", tree_method="hist",
    device="cuda" if USE_GPU else "cpu", grow_policy="lossguide", max_leaves=4,
    learning_rate=0.03, subsample=0.8, colsample_bytree=0.5, reg_lambda=1.0,
    random_state=Config.SEED
)

def xgb_fit(X, yv, X_es=None, y_es=None, margin=None, margin_es=None, n=None):
    dtrain = xgb.DMatrix(X[TREE_FEATS], label=yv, enable_categorical=True) if margin is None else xgb.DMatrix(X[TREE_FEATS], label=yv, base_margin=margin, enable_categorical=True)
    if X_es is None:
        bst = xgb.train(TE_XGB_PARAMS, dtrain, num_boost_round=n)
        return bst, n
    dval = xgb.DMatrix(X_es[TREE_FEATS], label=y_es, enable_categorical=True) if margin_es is None else xgb.DMatrix(X_es[TREE_FEATS], label=y_es, base_margin=margin_es, enable_categorical=True)
    bst = xgb.train(TE_XGB_PARAMS, dtrain, num_boost_round=10000, evals=[(dval, "val")],
                    early_stopping_rounds=Config.ES_ROUNDS, verbose_eval=False)
    return bst, bst.best_iteration + 1

def xgb_pred(bst, X, n=None, margin=None):
    dx = xgb.DMatrix(X[TREE_FEATS], enable_categorical=True) if margin is None else xgb.DMatrix(X[TREE_FEATS], base_margin=margin, enable_categorical=True)
    return bst.predict(dx, iteration_range=(0, n)) if n is not None else bst.predict(dx)

# 3. CatBoost Residual Booster
def cb_fit(X, yv, X_es=None, y_es=None, margin=None, margin_es=None, n=None):
    cb_params = dict(
        iterations=n or 10000, learning_rate=0.03, depth=4, eval_metric="AUC",
        random_seed=Config.SEED, task_type="GPU" if USE_GPU else "CPU", verbose=False
    )
    if X_es is not None:
        cb_params["early_stopping_rounds"] = Config.ES_ROUNDS
    m = CatBoostClassifier(**cb_params)
    p_tr = Pool(X[TREE_FEATS], yv, cat_features=CAT_COLS, baseline=margin)
    if X_es is None:
        m.fit(p_tr)
        return m, n
    p_es = Pool(X_es[TREE_FEATS], y_es, cat_features=CAT_COLS, baseline=margin_es)
    m.fit(p_tr, eval_set=p_es)
    return m, m.get_best_iteration() + 1

def cb_pred(m, X, margin=None):
    p = Pool(X[TREE_FEATS], cat_features=CAT_COLS, baseline=margin)
    return m.predict_proba(p)[:, 1]

print("Triple Residual Booster training functions configured.")
'''))

# Cell 8: 5-Fold Training Loop
cells.append(nbf.v4.new_code_cell('''MODELS = ["glm", "lgb_glm", "xgb_glm", "cb_glm"]
res = {k: {"oof": np.zeros(N_TR), "test": np.zeros(len(test_df)), "folds": []} for k in MODELS}
iters = {k: [] for k in ["lgb_glm", "xgb_glm", "cb_glm"]}

print("=" * 75)
print("Starting 5-Fold Velocity-Augmented Triple GLM Residual Boosting...")
print("=" * 75)

start_time = time.time()

for fold in range(Config.N_FOLDS):
    fold_t0 = time.time()
    tr_idx, va_idx, X_tr, X_va, X_te = fold_frames(fold)
    y_tr, y_va = y[tr_idx], y[va_idx]
    
    fit_i, es_i = train_test_split(np.arange(len(tr_idx)), test_size=0.10, stratify=y_tr, random_state=Config.SEED + fold)
    X_fit, X_es, y_fit, y_es = X_tr.iloc[fit_i], X_tr.iloc[es_i], y_tr[fit_i], y_tr[es_i]
    
    preds = {}
    
    # 1. GLM Convex Hyperplane
    glm = glm_fit(X_fit, y_fit)
    z_fit, z_es, z_va, z_te = glm(X_fit), glm(X_es), glm(X_va), glm(X_te)
    z_tr = glm(X_tr)
    preds["glm"] = expit(z_va)
    res["glm"]["oof"][va_idx] = preds["glm"]
    res["glm"]["test"] += expit(z_te) / Config.N_FOLDS
    
    # 2. LightGBM Residual Booster
    lgb_m, lgb_it = lgb_fit(X_fit, y_fit, X_es, y_es, margin=z_fit, margin_es=z_es)
    iters["lgb_glm"].append(lgb_it)
    preds["lgb_glm"] = lgb_pred(lgb_m, X_va, n=lgb_it, margin=z_va)
    res["lgb_glm"]["oof"][va_idx] = preds["lgb_glm"]
    
    # 3. XGBoost Residual Booster
    xgb_m, xgb_it = xgb_fit(X_fit, y_fit, X_es, y_es, margin=z_fit, margin_es=z_es)
    iters["xgb_glm"].append(xgb_it)
    preds["xgb_glm"] = xgb_pred(xgb_m, X_va, n=xgb_it, margin=z_va)
    res["xgb_glm"]["oof"][va_idx] = preds["xgb_glm"]
    
    # 4. CatBoost Residual Booster
    cb_m, cb_it = cb_fit(X_fit, y_fit, X_es, y_es, margin=z_fit, margin_es=z_es)
    iters["cb_glm"].append(cb_it)
    preds["cb_glm"] = cb_pred(cb_m, X_va, margin=z_va)
    res["cb_glm"]["oof"][va_idx] = preds["cb_glm"]
    
    fold_aucs = {k: roc_auc_score(y_va, preds[k]) for k in MODELS}
    fold_prs = {k: average_precision_score(y_va, preds[k]) for k in MODELS}
    
    for k in MODELS:
        res[k]["folds"].append(fold_aucs[k])
        
    print(f"Fold {fold+1}/{Config.N_FOLDS} ({time.time()-fold_t0:.1f}s) | "
          f"GLM: {fold_aucs['glm']:.5f} | "
          f"LGB: {fold_aucs['lgb_glm']:.5f} ({lgb_it} it) | "
          f"XGB: {fold_aucs['xgb_glm']:.5f} ({xgb_it} it) | "
          f"CB: {fold_aucs['cb_glm']:.5f} ({cb_it} it)")
    
    del X_fit, X_es, X_tr, X_va, X_te; gc.collect()

print("=" * 75)
print(f"5-Fold CV Complete in {(time.time()-start_time)/60:.2f} mins.")
'''))

# Cell 9: Full Refit on Entire Folds for Test Predictions
cells.append(nbf.v4.new_code_cell('''print("=" * 75)
print("Retraining Boosters on full fold data with optimal early-stopped rounds...")
print("=" * 75)

for fold in range(Config.N_FOLDS):
    t0 = time.time()
    tr_idx, va_idx, X_tr, X_va, X_te = fold_frames(fold)
    y_tr = y[tr_idx]
    
    glm = glm_fit(X_tr, y_tr)
    z_tr, z_te = glm(X_tr), glm(X_te)
    
    # 1. LightGBM full refit
    bst_lgb, _ = lgb_fit(X_tr, y_tr, n=iters["lgb_glm"][fold], margin=z_tr)
    res["lgb_glm"]["test"] += lgb_pred(bst_lgb, X_te, margin=z_te) / Config.N_FOLDS
    
    # 2. XGBoost full refit
    bst_xgb, _ = xgb_fit(X_tr, y_tr, n=iters["xgb_glm"][fold], margin=z_tr)
    res["xgb_glm"]["test"] += xgb_pred(bst_xgb, X_te, margin=z_te) / Config.N_FOLDS
    
    # 3. CatBoost full refit
    m_cb, _ = cb_fit(X_tr, y_tr, n=iters["cb_glm"][fold], margin=z_tr)
    res["cb_glm"]["test"] += cb_pred(m_cb, X_te, margin=z_te) / Config.N_FOLDS
    
    print(f"Refit Fold {fold+1}/{Config.N_FOLDS} in {time.time()-t0:.1f}s")
    del X_tr, X_va, X_te; gc.collect()

print("All fold test predictions generated.")
'''))

# Cell 10: Nelder-Mead Simplex Optimization
cells.append(nbf.v4.new_code_cell('''BOOSTERS = ["lgb_glm", "xgb_glm", "cb_glm"]
oof_ranks = {k: rankdata(res[k]["oof"]) / N_TR for k in BOOSTERS}
test_ranks = {k: rankdata(res[k]["test"]) / len(test_df) for k in BOOSTERS}

def simplex_loss(weights):
    w = np.maximum(weights, 0)
    w_sum = w.sum()
    if w_sum == 0:
        return 0.0
    w = w / w_sum
    blend = sum(w[i] * oof_ranks[m] for i, m in enumerate(BOOSTERS))
    return -roc_auc_score(y, blend)

init_weights = np.ones(len(BOOSTERS)) / len(BOOSTERS)
opt = minimize(simplex_loss, init_weights, method="Nelder-Mead", options={"maxiter": 500})
best_w = np.maximum(opt.x, 0)
best_w /= best_w.sum()

print("=" * 75)
print("OPTIMAL TRIPLE BLEND WEIGHTS (Nelder-Mead):")
for m, w in zip(BOOSTERS, best_w):
    print(f"  {m:12s}: {w*100:6.2f}%")

oof_triple = sum(best_w[i] * oof_ranks[m] for i, m in enumerate(BOOSTERS))
test_triple = sum(best_w[i] * test_ranks[m] for i, m in enumerate(BOOSTERS))

res["blend"] = {
    "oof": oof_triple,
    "refit": test_triple,
    "cv_auc": roc_auc_score(y, oof_triple),
    "pr_auc": average_precision_score(y, oof_triple)
}

for m in MODELS:
    res[m]["cv_auc"] = roc_auc_score(y, res[m]["oof"])
    res[m]["pr_auc"] = average_precision_score(y, res[m]["oof"])

print("-" * 75)
print(f"Triple Simplex Blend OOF ROC-AUC: {res['blend']['cv_auc']:.6f} | PR-AUC: {res['blend']['pr_auc']:.6f}")
print("=" * 75)
'''))

# Cell 11: Save Artifacts and Generate Submissions
cells.append(nbf.v4.new_code_cell('''# Save Out-Of-Fold predictions for downstream master stacking
np.save(Config.OUTPUT_DIR / "oof_velocity_glm.npy", res["glm"]["oof"])
np.save(Config.OUTPUT_DIR / "oof_velocity_lgb.npy", res["lgb_glm"]["oof"])
np.save(Config.OUTPUT_DIR / "oof_velocity_xgb.npy", res["xgb_glm"]["oof"])
np.save(Config.OUTPUT_DIR / "oof_velocity_cb.npy", res["cb_glm"]["oof"])
np.save(Config.OUTPUT_DIR / "oof_velocity_blend.npy", res["blend"]["oof"])

# Generate Submissions
sub_blend = sample_sub.copy()
sub_blend["is_fraud"] = res["blend"]["refit"]
sub_blend.to_csv(Config.OUTPUT_DIR / "velocity_burst_glm_blend_submission.csv", index=False)

sub_xgb = sample_sub.copy()
sub_xgb["is_fraud"] = res["xgb_glm"]["test"]
sub_xgb.to_csv(Config.OUTPUT_DIR / "velocity_burst_glm_xgb_submission.csv", index=False)

sub_lgb = sample_sub.copy()
sub_lgb["is_fraud"] = res["lgb_glm"]["test"]
sub_lgb.to_csv(Config.OUTPUT_DIR / "velocity_burst_glm_lgb_submission.csv", index=False)

sub_cb = sample_sub.copy()
sub_cb["is_fraud"] = res["cb_glm"]["test"]
sub_cb.to_csv(Config.OUTPUT_DIR / "velocity_burst_glm_cb_submission.csv", index=False)

print(f"Generated submissions in {Config.OUTPUT_DIR}:")
print("  - velocity_burst_glm_blend_submission.csv")
print("  - velocity_burst_glm_xgb_submission.csv")
print("  - velocity_burst_glm_lgb_submission.csv")
print("  - velocity_burst_glm_cb_submission.csv")
'''))

# Cell 12: Summary Diagnostics Table & Evaluation Curves
cells.append(nbf.v4.new_code_cell('''summary_df = pd.DataFrame({
    "Model": ["GLM (Logistic Regression)", "LightGBM GLM-Boost", "XGBoost GLM-Boost", "CatBoost GLM-Boost", "Optimal Triple Simplex Blend"],
    "5-Fold OOF ROC-AUC": [res["glm"]["cv_auc"], res["lgb_glm"]["cv_auc"], res["xgb_glm"]["cv_auc"], res["cb_glm"]["cv_auc"], res["blend"]["cv_auc"]],
    "5-Fold OOF PR-AUC": [res["glm"]["pr_auc"], res["lgb_glm"]["pr_auc"], res["xgb_glm"]["pr_auc"], res["cb_glm"]["pr_auc"], res["blend"]["pr_auc"]]
})

print("-" * 75)
print("VELOCITY-AUGMENTED TRIPLE GLM RESIDUAL BOOSTING BENCHMARK RESULTS")
print("-" * 75)
print(summary_df.to_string(index=False))

# Plot ROC & PR Curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

for m in ["glm", "lgb_glm", "xgb_glm", "cb_glm", "blend"]:
    pred = res[m]["oof"]
    label = f"{m.upper()} (AUC = {res[m]['cv_auc']:.5f})"
    from sklearn.metrics import roc_curve, precision_recall_curve
    fpr, tpr, _ = roc_curve(y, pred)
    ax1.plot(fpr, tpr, label=label, lw=1.5)
    
    p_curve, r_curve, _ = precision_recall_curve(y, pred)
    ax2.plot(r_curve, p_curve, label=f"{m.upper()} (PR = {res[m]['pr_auc']:.5f})", lw=1.5)

ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax1.set_title("ROC Curves Comparison", fontsize=12, fontweight='bold')
ax1.set_xlabel("False Positive Rate")
ax1.set_ylabel("True Positive Rate")
ax1.legend(loc="lower right")
ax1.grid(True, alpha=0.3)

ax2.set_title("Precision-Recall Curves Comparison", fontsize=12, fontweight='bold')
ax2.set_xlabel("Recall")
ax2.set_ylabel("Precision")
ax2.legend(loc="upper right")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(Config.OUTPUT_DIR / "velocity_burst_evaluation_curves.png", dpi=200)
plt.show()
print("Saved evaluation curves to velocity_burst_evaluation_curves.png")
'''))

nb['cells'] = cells
nb['metadata'] = {
    'kernelspec': {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'python3'
    },
    'language_info': {
        'name': 'python',
        'version': '3.10.12'
    }
}

# Strip outputs/execution_count from markdown cells and validate
for cell in nb['cells']:
    if cell['cell_type'] == 'markdown':
        cell.pop('outputs', None)
        cell.pop('execution_count', None)

nbf.validate(nb)

target_path = Path('pgs-s6e11/code/velocity_burst_glm_boost/model.ipynb')
with open(target_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f'Successfully generated and validated {target_path}')

# Write kernel-metadata.json
meta = {
    'id': 'shivamgravity/pgs-s6e11-velocity-burst-glm-boost',
    'title': 'PGS S6E11 Velocity Burst GLM Boost',
    'code_file': 'model.ipynb',
    'language': 'python',
    'kernel_type': 'notebook',
    'is_private': True,
    'enable_gpu': True,
    'enable_tpu': False,
    'enable_internet': False,
    'dataset_sources': [],
    'competition_sources': ['playground-series-s6e11'],
    'kernel_sources': []
}

meta_path = Path('pgs-s6e11/code/velocity_burst_glm_boost/kernel-metadata.json')
with open(meta_path, 'w', encoding='utf-8') as f:
    json.dump(meta, f, indent=2)

print(f'Successfully wrote metadata: {meta_path}')
