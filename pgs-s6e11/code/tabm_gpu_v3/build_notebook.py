#!/usr/bin/env python3
"""
Build Kaggle GPU Notebook for Multi-Seed Deep TabM-D Piecewise-Linear Ensemble (Phase 1, SOTA V3).
Trains a 32-slice dual-seed (k=32, Seed 42 & Seed 2026) TabM-D ensemble on Tesla T4 GPU.
"""

from pathlib import Path
import nbformat as nbf

def build_tabm_v3_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12"
        }
    }

    cells = []

    # Cell 0: Install PyTabKit
    cells.append(nbf.v4.new_code_cell("""# Step 0: Install PyTabKit
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pytabkit"])
print("PyTabKit installed successfully.")
"""))

    # Cell 1: Environment & Setup
    cells.append(nbf.v4.new_code_cell("""# Step 1: Device Setup & Imports
import os
import gc
import glob
import time
import random
import warnings
import traceback
from pathlib import Path
from importlib.metadata import version

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder

import torch
from pytabkit import TabM_D_Classifier

warnings.filterwarnings('ignore')

print("PyTorch version: ", torch.__version__)
print("CUDA available:  ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU device:      ", torch.cuda.get_device_name(0))
    print("Device count:    ", torch.cuda.device_count())
print("PyTabKit version:", version("pytabkit"))

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(42)
"""))

    # Cell 2: Dynamic Data Discovery
    cells.append(nbf.v4.new_code_cell("""# Step 2: Load Competition Data
train_candidates = (
    glob.glob('/kaggle/input/**/train.csv', recursive=True)
    + glob.glob('**/train.csv', recursive=True)
    + glob.glob('../../dataset/train.csv')
    + glob.glob('pgs-s6e11/dataset/train.csv')
)

if not train_candidates:
    raise FileNotFoundError("Could not find train.csv in candidate paths.")

TRAIN_PATH = Path(train_candidates[0])
DATA_DIR = TRAIN_PATH.parent
TEST_PATH = DATA_DIR / "test.csv"
SUB_PATH = DATA_DIR / "sample_submission.csv"

print(f"Loading data from: {DATA_DIR}")
t0 = time.time()
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
sample_sub = pd.read_csv(SUB_PATH)

print(f"Loaded train: {train_df.shape}, test: {test_df.shape} in {time.time() - t0:.2f}s")

ID = 'id'
TARGET = 'is_fraud'

train_id = train_df[ID].values
test_id = test_df[ID].values
y = train_df[TARGET].values.astype(int)

# Extract raw DataFrames for feature engineering
X = train_df.drop(columns=[ID, TARGET]).copy()
X_test = test_df.drop(columns=[ID]).copy()
del train_df, test_df
gc.collect()

print(f"Global fraud rate: {y.mean():.4%} ({y.sum():,} frauds out of {len(y):,} rows)")
"""))

    # Cell 3: Feature Engineering with Decimal Residues
    cells.append(nbf.v4.new_code_cell("""# Step 3: Domain & Decimal Residue Feature Engineering
t0 = time.time()

cat_cols = X.select_dtypes(include=['object', 'bool']).columns.tolist()
num_cols = X.select_dtypes(exclude=['object', 'bool']).columns.tolist()
print(f"Base columns: {len(cat_cols)} categorical, {len(num_cols)} numerical")

important_combos = [
    ('merchant_category', 'card_type', 'txn_count_last_24h'),
    ('merchant_category', 'customer_age'),
]

category_map = {}

def feature_engineering(df, fit=False):
    # 1. Decimal / Fractional parts
    for col in ['amount_usd', 'hours_since_last_txn', 'velocity_score', 'distance_from_home_km']:
        if col in df.columns:
            df[f'_{col}_decimal'] = (df[col] % 1).round(2).astype('float32')

    # 2. Modulo Cents & Digit Quantization Signatures
    if 'amount_usd' in df.columns:
        amt = df['amount_usd']
        df['_amt_cents'] = np.floor((amt - np.floor(amt)) * 100).astype('int32')
        df['_amt_tenths'] = ((np.floor(amt * 10).astype('int32') % 10)).astype(str).astype('category')
        df['_is_exact_dollar'] = ((amt % 1.0 == 0).astype('int32')).astype(str).astype('category')

    if 'hours_since_last_txn' in df.columns:
        hrs = df['hours_since_last_txn']
        df['_hrs_tenths'] = ((np.floor(hrs * 10).astype('int32') % 10)).astype(str).astype('category')

    if 'account_balance_usd' in df.columns:
        bal = df['account_balance_usd']
        df['_bal_tenths'] = ((np.floor(bal * 10).astype('int32') % 10)).astype(str).astype('category')

    # 3. Categorize Numerical Floor Values
    for col in num_cols + ['_amount_usd_decimal', '_hours_since_last_txn_decimal', '_velocity_score_decimal', '_distance_from_home_km_decimal']:
        if col in df.columns:
            cat_name = f'{col}_cat_' if col in num_cols else f'{col[1:]}_cat_'
            values = np.floor(df[col]) if col in num_cols else df[col]
            if fit:
                codes, uniques = values.factorize()
                category_map[col] = uniques
            else:
                uniques = category_map[col]
                code_map = {cat: i for i, cat in enumerate(uniques)}
                codes = values.map(code_map).fillna(-1).astype('int32')
            df[cat_name] = codes
            df[cat_name] = df[cat_name].astype('category')

    # 4. Quantile Bins
    bin_config = {'merchant_risk_score': [7], 'hours_since_last_txn': [50, 100]}
    for col, bins_list in bin_config.items():
        if col in df.columns:
            for n_bins in bins_list:
                bin_name = f'{col}_{n_bins}_quantile_bin_'
                if fit:
                    kb = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile', subsample=None)
                    binned = kb.fit_transform(df[[col]]).ravel().astype('int32')
                    category_map[bin_name] = kb
                else:
                    kb = category_map[bin_name]
                    binned = kb.transform(df[[col]]).ravel().astype('int32')
                df[bin_name] = binned
                df[bin_name] = df[bin_name].astype('category')

    # 5. Domain Interaction Combos
    for combo in important_combos:
        cname = f"_{'_x_'.join(combo)}"
        df[cname] = df[combo[0]].astype(str)
        for c in combo[1:]:
            df[cname] = df[cname] + '_' + df[c].astype(str)
        df[cname] = df[cname].astype('category')

    return df

print("Transforming training features...")
X = feature_engineering(X, fit=True)
print("Transforming test features...")
X_test = feature_engineering(X_test, fit=False)

combo_names = [f"_{'_x_'.join(c)}" for c in important_combos]
print(f"Engineered features ready in {time.time() - t0:.2f}s. Total columns: {X.shape[1]}")
"""))

    # Cell 4: Multi-Seed TabM-D Training (k=32, Seeds 42 and 2026)
    cells.append(nbf.v4.new_code_cell("""# Step 4: Multi-Seed TabM-D (k=32 Slices, Seeds 42 & 2026) 5-Fold Stratified Training
FOLDS = 5
SEEDS = [42, 2026]
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof_preds = np.zeros(len(y), dtype=np.float32)
test_preds = np.zeros(len(X_test), dtype=np.float32)

tabm_base_params = {
    'verbosity': 1,
    'val_metric_name': '1-auc_ovr',
    'arch_type': 'tabm',
    'tabm_k': 32,             # Scaled to 32 parallel ensemble slices
    'num_emb_type': 'pwl',     # Piecewise linear numerical embeddings
    'num_emb_n_bins': 12,      # Quantile bins per numerical feature
    'd_embedding': 16,
    'n_blocks': 3,
    'd_block': 512,
    'dropout': 0.02,
    'n_epochs': 2,
    'patience': 1,
    'batch_size': 'auto',
    'lr': 0.0012,
    'weight_decay': 0.001,
}

te_cols = combo_names

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
    fold_t0 = time.time()
    print("=" * 70)
    print(f">>> Fold {fold}/{FOLDS} Multi-Seed TabM-D (k=32, PWL, Dual-Seed) Training")
    print("=" * 70)

    X_tr = X.iloc[tr_idx].copy()
    X_val = X.iloc[val_idx].copy()
    y_tr = y[tr_idx]
    y_val = y[val_idx]
    X_tst = X_test.copy()

    # Out-of-fold Target Encoding on key interaction combos
    TE = TargetEncoder(cv=FOLDS, smooth='auto', shuffle=True, random_state=42 + fold)
    tr_enc = TE.fit_transform(X_tr[te_cols], y_tr)
    val_enc = TE.transform(X_val[te_cols])
    tst_enc = TE.transform(X_tst[te_cols])

    te_names = [f'_{col}_TE' for col in te_cols]
    X_tr[te_names] = tr_enc.astype('float32')
    X_val[te_names] = val_enc.astype('float32')
    X_tst[te_names] = tst_enc.astype('float32')

    fold_val_preds = np.zeros(len(val_idx), dtype=np.float32)
    fold_test_preds = np.zeros(len(X_tst), dtype=np.float32)

    for seed in SEEDS:
        print(f"  --> Training Seed {seed} (k=32 slices)...")
        seed_everything(seed)
        fold_params = dict(tabm_base_params)
        fold_params['random_state'] = seed + fold

        model = TabM_D_Classifier(**fold_params)
        model.fit(X_tr, y_tr, X_val, y_val)

        v_p = model.predict_proba(X_val)[:, 1]
        t_p = model.predict_proba(X_tst)[:, 1]

        fold_val_preds += v_p / len(SEEDS)
        fold_test_preds += t_p / len(SEEDS)

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    oof_preds[val_idx] = fold_val_preds
    test_preds += fold_test_preds / FOLDS

    fold_auc = roc_auc_score(y_val, fold_val_preds)
    fold_prauc = average_precision_score(y_val, fold_val_preds)
    elapsed = time.time() - fold_t0
    print(f"Fold {fold} finished in {elapsed:.1f}s | ROC-AUC: {fold_auc:.6f} | PR-AUC: {fold_prauc:.6f}")

    del X_tr, X_val, X_tst, y_tr, y_val
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
"""))

    # Cell 5: Evaluation & Export with 4-Zone Manifold
    cells.append(nbf.v4.new_code_cell("""# Step 5: Overall Evaluation & Export with 4-Zone Manifold
overall_auc = roc_auc_score(y, oof_preds)
overall_prauc = average_precision_score(y, oof_preds)

print("=" * 70)
print(f">>> Overall Multi-Seed TabM-D PWL 5-Fold OOF ROC-AUC: {overall_auc:.6f}")
print(f">>> Overall Multi-Seed TabM-D PWL 5-Fold OOF PR-AUC:  {overall_prauc:.6f}")
print("=" * 70)

# 4-Zone Manifold Calibration
def compute_zall(df):
    clean = ((df['cvv_retry_count'] == 0) & (df['prior_disputes'] == 0) & 
             (~df['is_new_merchant'].astype(bool)) & (~df['is_foreign_transaction'].astype(bool)) & 
             (~df['used_vpn'].astype(bool)) & (~df['ip_country_mismatch'].astype(bool)) & 
             (~df['billing_shipping_mismatch'].astype(bool)) & (~df['is_ai_generated_scam_attempt'].astype(bool))).to_numpy()
    zone1 = df['merchant_risk_score'].to_numpy() < 13.1
    zone2 = (df['velocity_score'].to_numpy() <= 4.1) & (df['merchant_risk_score'].to_numpy() <= 20.0)
    zone34 = clean & (((df['velocity_score'].to_numpy() <= 4.1) & (df['merchant_risk_score'].to_numpy() <= 33.8)) |
                      ((df['velocity_score'].to_numpy() <= 6.0) & (df['merchant_risk_score'].to_numpy() <= 25.0)))
    return zone1 | zone2 | zone34

zall_te = compute_zall(X_test)
print(f"4-Zone Zero-Fraud accounts in test set: {zall_te.sum():,} ({zall_te.mean():.2%})")

# Rank preserving manifold calibration
def apply_rank_preserving_manifold(preds, mask_zero):
    n_total = len(preds)
    n_zero = mask_zero.sum()
    ranks = np.zeros(n_total, dtype=np.float64)
    ranks[mask_zero] = (rankdata(preds[mask_zero]) - 0.5) / n_total
    ranks[~mask_zero] = (n_zero + rankdata(preds[~mask_zero]) - 0.5) / n_total
    return ranks

cal_test_preds = apply_rank_preserving_manifold(test_preds, zall_te)

# Save arrays
np.save('oof_tabm_pwl_v3.npy', oof_preds.astype(np.float32))
np.save('test_tabm_pwl_v3.npy', test_preds.astype(np.float32))
np.save('test_tabm_pwl_v3_calibrated.npy', cal_test_preds.astype(np.float32))

# Export Submissions
sub = pd.DataFrame({'id': test_id, 'is_fraud': cal_test_preds})
sub.to_csv('submission.csv', index=False)
sub.to_csv('tabm_pwl_v3_submission.csv', index=False)

print(f"Successfully generated submission.csv ({len(sub):,} rows)")
"""))

    nb.cells = cells
    out_path = Path("pgs-s6e11/code/tabm_gpu_v3/model.ipynb")
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Wrote notebook to: {out_path}")

if __name__ == "__main__":
    build_tabm_v3_notebook()
