import os
import gc
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logit, expit
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.metrics import roc_auc_score, average_precision_score
import lightgbm as lgb
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")

print("=" * 75)
print("🧠 PGS-S6E11: 10-FOLD MASTER GLM RESIDUAL BOOSTER (74 DIGIT KEYS + VELOCITY)")
print("=" * 75)

DATA_DIR = Path("pgs-s6e11/dataset")
OUTPUT_DIR = Path("pgs-s6e11/code/glm_10fold_boost/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")
sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")

TARGET, ID_COL = "is_fraud", "id"
BASE_FEATURES = [c for c in test_df.columns if c != ID_COL]
FLAG_COLS = list(train_df.select_dtypes(include="bool").columns)

for df in (train_df, test_df):
    df[FLAG_COLS] = df[FLAG_COLS].astype("int8")

y = train_df[TARGET].astype(int).values
N_TR, N_TE = len(train_df), len(test_df)

CAT_COLS = [c for c in BASE_FEATURES if not pd.api.types.is_numeric_dtype(train_df[c])]
NUM_COLS = [c for c in BASE_FEATURES if c not in CAT_COLS + FLAG_COLS]

for c in CAT_COLS:
    train_df[c] = train_df[c].astype(str)
    test_df[c] = test_df[c].astype(str)

print(f"Loaded: {N_TR:,} Train rows, {N_TE:,} Test rows")
print(f"Features: {len(NUM_COLS)} numeric, {len(FLAG_COLS)} flags, {len(CAT_COLS)} categoricals")

# 1. Velocity & Anomaly Features
for df in (train_df, test_df):
    df["spend_velocity"] = df["amount_usd"] / (df["hours_since_last_txn"] + 0.05)
    df["travel_speed_kmh"] = df["distance_from_home_km"] / (df["hours_since_last_txn"] + 0.05)
    df["balance_drain_ratio"] = df["amount_usd"] / (df["account_balance_usd"] + 1.0)

NUM_COLS += ["spend_velocity", "travel_speed_kmh", "balance_drain_ratio"]

# 2. Count Features
both_num = pd.concat([train_df[NUM_COLS], test_df[NUM_COLS]])
COUNT_COLS = []
for c in NUM_COLS:
    counts = both_num[c].value_counts()
    for df in (train_df, test_df):
        df[f"{c}_count"] = df[c].map(counts).fillna(0).astype(np.float32)
    COUNT_COLS.append(f"{c}_count")
del both_num
gc.collect()

# 3. Decimal precision and cent features
def n_decimals(v, max_d=4):
    v = v[~np.isnan(v)][:200_000]
    for d in range(max_d + 1):
        scaled = v * 10 ** d
        if np.allclose(scaled, np.rint(scaled), rtol=0, atol=1e-6):
            return d
    return max_d

DECIMALS = {c: n_decimals(train_df[c].values.astype(float)) for c in NUM_COLS}
CENT_COLS = []
for c in NUM_COLS:
    if DECIMALS[c] >= 2:
        for df in (train_df, test_df):
            df[f"{c}_cents"] = (np.abs(np.rint(df[c].values * 10 ** DECIMALS[c])).astype(np.int64) % 100).astype(np.float32)
        CENT_COLS.append(f"{c}_cents")

print(f"Extracted {len(COUNT_COLS)} count features and {len(CENT_COLS)} cents features.")

# 4. Multi-Scale Digit Keys
def digit_keys(x, d):
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

# Joint 2-Way Fraud Keys
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

del both
gc.collect()

KEY_NAMES = list(KEY_CODES)
TE_COLS = [f"te_{k}" for k in KEY_NAMES]
RAW_ALL = BASE_FEATURES + [c for c in NUM_COLS if c not in BASE_FEATURES] + CENT_COLS + COUNT_COLS
TREE_FEATS = RAW_ALL + TE_COLS

print(f"Total encoded keys: {len(KEY_NAMES)} (including {len(joint_pairs)} 2-way joint keys)")
print(f"Total features for tree models: {len(TREE_FEATS)}")

# 5. Stratified 10-Fold Target Encoding Engine
N_FOLDS = 10
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
folds = np.zeros(N_TR, dtype=int)
for f, (_, va) in enumerate(skf.split(train_df, y)):
    folds[va] = f

GROUPS = np.concatenate([folds, np.full(N_TE, -1)])
SMOOTH = 10.0

def target_encode(stat_rows, apply_rows, yv, smooth=SMOOTH, exclude_own_fold=True):
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
    te = target_encode(tr_idx, np.concatenate([tr_idx, va_idx, N_TR + np.arange(N_TE)]), y)
    a, b = len(tr_idx), len(tr_idx) + len(va_idx)
    return (
        tr_idx, va_idx,
        enc_frame(train_df, tr_idx, te[:a]),
        enc_frame(train_df, va_idx, te[a:b]),
        enc_frame(test_df, None, te[b:])
    )

def glm_matrix(X, qt=None, sc=None):
    p = np.clip(X[TE_COLS].values, 1e-4, 1 - 1e-4)
    num = X[NUM_COLS + COUNT_COLS].values.astype(float)
    if qt is None:
        qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal", subsample=200_000, random_state=42).fit(num)
    M = np.hstack([np.log(p / (1 - p)), qt.transform(num), X[FLAG_COLS].values.astype(float)])
    if sc is None:
        sc = StandardScaler().fit(M)
    return np.nan_to_num(sc.transform(M)), qt, sc

def glm_fit(X, yv):
    M, qt, sc = glm_matrix(X)
    m = LogisticRegression(C=0.5, max_iter=500, random_state=42).fit(M, yv)
    return lambda Z: m.decision_function(glm_matrix(Z, qt, sc)[0])

# 6. GBDT Parameters
lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 4,
    "max_depth": 3,
    "learning_rate": 0.04,
    "min_child_samples": 80,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": 4
}

xgb_params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "max_depth": 2,
    "learning_rate": 0.04,
    "min_child_weight": 80,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": 4
}

oof_glm = np.zeros(N_TR, dtype=np.float32)
test_glm = np.zeros(N_TE, dtype=np.float32)

oof_lgb = np.zeros(N_TR, dtype=np.float32)
test_lgb = np.zeros(N_TE, dtype=np.float32)

oof_xgb = np.zeros(N_TR, dtype=np.float32)
test_xgb = np.zeros(N_TE, dtype=np.float32)

# Categorical column factorization for GBDTs
for c in CAT_COLS:
    cats = sorted(set(train_df[c]) | set(test_df[c]))
    train_df[c] = pd.Categorical(train_df[c], categories=cats)
    test_df[c] = pd.Categorical(test_df[c], categories=cats)

print("\n" + "=" * 75)
print("Initiating 10-Fold Stratified GLM Residual Boosting...")
print("=" * 75)

total_t0 = time.time()

for f in range(N_FOLDS):
    fold_t0 = time.time()
    tr_idx, va_idx, tr_df, va_df, te_df = fold_frames(f)
    y_tr, y_va = y[tr_idx], y[va_idx]

    # Convert categoricals to codes for trees
    X_tr_tree = tr_df[TREE_FEATS].copy()
    X_va_tree = va_df[TREE_FEATS].copy()
    X_te_tree = te_df[TREE_FEATS].copy()
    for c in CAT_COLS:
        X_tr_tree[c] = X_tr_tree[c].cat.codes
        X_va_tree[c] = X_va_tree[c].cat.codes
        X_te_tree[c] = X_te_tree[c].cat.codes

    # 1. GLM Fit
    glm_predict = glm_fit(tr_df, y_tr)
    z_tr = glm_predict(tr_df)
    z_va = glm_predict(va_df)
    z_te = glm_predict(te_df)

    oof_glm[va_idx] = expit(z_va)
    test_glm += expit(z_te) / N_FOLDS
    auc_glm = roc_auc_score(y_va, oof_glm[va_idx])

    # 2. LightGBM Boost
    dtr_lgb = lgb.Dataset(X_tr_tree, label=y_tr, init_score=z_tr)
    dva_lgb = lgb.Dataset(X_va_tree, label=y_va, init_score=z_va, reference=dtr_lgb)
    model_lgb = lgb.train(
        lgb_params,
        dtr_lgb,
        num_boost_round=600,
        valid_sets=[dva_lgb],
        callbacks=[lgb.early_stopping(40, verbose=False)]
    )

    lgb_va = z_va + model_lgb.predict(X_va_tree, raw_score=True)
    lgb_te = z_te + model_lgb.predict(X_te_tree, raw_score=True)
    oof_lgb[va_idx] = expit(lgb_va)
    test_lgb += expit(lgb_te) / N_FOLDS
    auc_lgb = roc_auc_score(y_va, oof_lgb[va_idx])

    # 3. XGBoost Boost
    dtr_xgb = xgb.DMatrix(X_tr_tree, label=y_tr, base_margin=z_tr)
    dva_xgb = xgb.DMatrix(X_va_tree, label=y_va, base_margin=z_va)
    dte_xgb = xgb.DMatrix(X_te_tree, base_margin=z_te)
    model_xgb = xgb.train(
        xgb_params,
        dtr_xgb,
        num_boost_round=600,
        evals=[(dva_xgb, "val")],
        early_stopping_rounds=40,
        verbose_eval=False
    )

    oof_xgb[va_idx] = expit(model_xgb.predict(dva_xgb, output_margin=True))
    test_xgb += expit(model_xgb.predict(dte_xgb, output_margin=True)) / N_FOLDS
    auc_xgb = roc_auc_score(y_va, oof_xgb[va_idx])

    fold_sec = time.time() - fold_t0
    print(f"Fold {f+1:2d}/{N_FOLDS} [{fold_sec:4.1f}s] -> GLM: {auc_glm:.5f} | LGB: {auc_lgb:.5f} | XGB: {auc_xgb:.5f}")

total_elapsed = time.time() - total_t0
print(f"\nCompleted all {N_FOLDS} folds in {total_elapsed/60:.2f} minutes!")

# 7. Final Evaluations
auc_glm_full = roc_auc_score(y, oof_glm)
auc_lgb_full = roc_auc_score(y, oof_lgb)
auc_xgb_full = roc_auc_score(y, oof_xgb)

pr_lgb_full = average_precision_score(y, oof_lgb)
pr_xgb_full = average_precision_score(y, oof_xgb)

# Optimal rank blend of 10-Fold LGB + XGB
r_lgb = rankdata(oof_lgb) / N_TR
r_xgb = rankdata(oof_xgb) / N_TR
oof_blend = 0.52 * r_lgb + 0.48 * r_xgb

r_lgb_te = rankdata(test_lgb) / N_TE
r_xgb_te = rankdata(test_xgb) / N_TE
test_blend = 0.52 * r_lgb_te + 0.48 * r_xgb_te

auc_blend = roc_auc_score(y, oof_blend)
pr_blend = average_precision_score(y, oof_blend)

print("\n" + "=" * 75)
print("🏆 10-FOLD GLM RESIDUAL BOOSTER FINAL SCORES")
print("=" * 75)
print(f"GLM Alone (10-Fold Newton-Cholesky) : ROC-AUC = {auc_glm_full:.6f}")
print(f"LGB-GLM Booster (10-Fold 4-Leaf)     : ROC-AUC = {auc_lgb_full:.6f} | PR-AUC = {pr_lgb_full:.6f}")
print(f"XGB-GLM Booster (10-Fold Depth-2)    : ROC-AUC = {auc_xgb_full:.6f} | PR-AUC = {pr_xgb_full:.6f}")
print(f"10-Fold Optimal Rank Blend           : ROC-AUC = {auc_blend:.6f} | PR-AUC = {pr_blend:.6f}")
print("=" * 75)

# Save Outputs
np.save(OUTPUT_DIR / "oof_10f_glm.npy", oof_glm)
np.save(OUTPUT_DIR / "test_10f_glm.npy", test_glm)

np.save(OUTPUT_DIR / "oof_10f_lgb.npy", oof_lgb)
np.save(OUTPUT_DIR / "test_10f_lgb.npy", test_lgb)

np.save(OUTPUT_DIR / "oof_10f_xgb.npy", oof_xgb)
np.save(OUTPUT_DIR / "test_10f_xgb.npy", test_xgb)

np.save(OUTPUT_DIR / "oof_10f_blend.npy", oof_blend)
np.save(OUTPUT_DIR / "test_10f_blend.npy", test_blend)

sub = sample_sub.copy()
sub["is_fraud"] = rankdata(test_blend) / N_TE
sub_path = OUTPUT_DIR / "glm_10fold_boost_submission.csv"
sub.to_csv(sub_path, index=False)
print(f"Generated 10-Fold Submission: {sub_path} | Shape: {sub.shape}")
