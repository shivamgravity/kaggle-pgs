#!/usr/bin/env python3
"""
Build Kaggle GPU Notebook for Deep TabM-D Piecewise-Linear Ensemble (Phase 1, SOTA V2).
Trains a 16-member TabM ensemble with PWL numerical embeddings and decimal residue features on Tesla T4.
"""

from pathlib import Path
import nbformat as nbf

def build_tabm_v2_notebook():
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
    cells.append(nbf.v4.new_code_cell("""# Step 1: Imports and Device Setup
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

    # Cell 2: Data Loading
    cells.append(nbf.v4.new_code_cell("""# Step 2: Dynamic Dataset Discovery & Loading
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

X = train_df.drop(columns=[ID, TARGET]).copy()
X_test = test_df.drop(columns=[ID]).copy()
del train_df, test_df
gc.collect()

print(f"Global fraud rate: {y.mean():.4%} ({y.sum():,} frauds out of {len(y):,} rows)")
"""))

    # Cell 3: Feature Engineering with Decimal Residues (Tested & Bug-Free)
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

    # 2. Modulo Cents & Digit Quantization Signatures (Pandas-native)
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

    # 5. Interaction Categories
    combo_names = []
    for cols in important_combos:
        combo_name = '_'.join(cols) + '_'
        combo_names.append(combo_name)
        combo_series = df[cols[0]].astype(str)
        for col in cols[1:]: 
            combo_series = combo_series + '_' + df[col].astype(str)
        if fit:
            codes, uniques = pd.factorize(combo_series, sort=False)
            category_map[combo_name] = uniques
        else:
            uniques = category_map[combo_name]
            code_map = {cat: i for i, cat in enumerate(uniques)}
            codes = combo_series.map(code_map).fillna(-1).astype('int32')
        df[combo_name] = codes
        df[combo_name] = df[combo_name].astype('category')

    new_cat_cols = [col for col in df.columns if col.endswith('_')]
    new_num_cols = [col for col in df.columns if col.startswith('_')]
    return df, new_cat_cols, new_num_cols, combo_names

X, new_cat_cols, new_num_cols, combo_names = feature_engineering(X, fit=True)
X_test, _, _, _ = feature_engineering(X_test, fit=False)
cat_cols += new_cat_cols
num_cols += new_num_cols
print(f"Feature engineering completed in {time.time() - t0:.2f}s: X shape={X.shape}, X_test={X_test.shape}")
"""))

    # Cell 4: 5-Fold Stratified TabM Training Loop with PWL
    cells.append(nbf.v4.new_code_cell("""# Step 4: 5-Fold Stratified TabM-D Piecewise-Linear Training Loop
FOLDS = 5
SEED = 42

skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
oof_preds = np.zeros(len(X), dtype=np.float64)
test_preds = np.zeros(len(X_test), dtype=np.float64)

# SOTA Piecewise-Linear TabM-D configuration
tabm_params = {
    'random_state': SEED,
    'verbosity': 1,
    'val_metric_name': '1-auc_ovr',

    'arch_type': 'tabm',
    'tabm_k': 16,             # 16 ensemble members in one forward pass
    'num_emb_type': 'pwl',     # Piecewise-linear embeddings
    'num_emb_n_bins': 10,      # Quantile bins per numerical feature
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
    print("=" * 60)
    print(f">>> Fold {fold}/{FOLDS} TabM-D (k=16, PWL) Training")
    print("=" * 60)

    X_tr = X.iloc[tr_idx].copy()
    X_val = X.iloc[val_idx].copy()
    y_tr = y[tr_idx]
    y_val = y[val_idx]
    X_tst = X_test.copy()

    # Out-of-fold Target Encoding on key interaction combos
    TE = TargetEncoder(cv=FOLDS, smooth='auto', shuffle=True, random_state=SEED + fold)
    tr_enc = TE.fit_transform(X_tr[te_cols], y_tr)
    val_enc = TE.transform(X_val[te_cols])
    tst_enc = TE.transform(X_tst[te_cols])

    te_names = [f'_{col}_TE' for col in te_cols]
    X_tr[te_names] = tr_enc.astype('float32')
    X_val[te_names] = val_enc.astype('float32')
    X_tst[te_names] = tst_enc.astype('float32')

    print(f"Features for training: {X_tr.shape[1]} columns")

    fold_params = dict(tabm_params)
    fold_params['random_state'] = SEED + fold

    model = TabM_D_Classifier(**fold_params)
    model.fit(X_tr, y_tr, X_val, y_val)

    val_preds = model.predict_proba(X_val)[:, 1]
    fold_test_preds = model.predict_proba(X_tst)[:, 1]

    oof_preds[val_idx] = val_preds
    test_preds += fold_test_preds / FOLDS

    fold_auc = roc_auc_score(y_val, val_preds)
    fold_prauc = average_precision_score(y_val, val_preds)
    elapsed = time.time() - fold_t0
    print(f"Fold {fold} finished in {elapsed:.1f}s | ROC-AUC: {fold_auc:.6f} | PR-AUC: {fold_prauc:.6f}")

    del X_tr, X_val, X_tst, y_tr, y_val, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
"""))

    # Cell 5: Evaluation & Export
    cells.append(nbf.v4.new_code_cell("""# Step 5: Overall Evaluation & Export
overall_auc = roc_auc_score(y, oof_preds)
overall_prauc = average_precision_score(y, oof_preds)

print("=" * 60)
print(f">>> Overall TabM-D PWL 5-Fold OOF ROC-AUC: {overall_auc:.6f}")
print(f">>> Overall TabM-D PWL 5-Fold OOF PR-AUC:  {overall_prauc:.6f}")
print("=" * 60)

np.save('oof_tabm_pwl_v2.npy', oof_preds.astype(np.float32))
np.save('test_tabm_pwl_v2.npy', test_preds.astype(np.float32))

r_test = (rankdata(test_preds) - 0.5) / len(test_preds)
sub = pd.DataFrame({'id': test_id, 'is_fraud': r_test})
sub.to_csv('tabm_pwl_v2_submission.csv', index=False)

# Also export as standard oof_preds.csv / submission.csv
pd.DataFrame({'id': train_id, 'is_fraud': oof_preds}).to_csv('oof_preds.csv', index=False)
pd.DataFrame({'id': test_id, 'is_fraud': test_preds}).to_csv('submission.csv', index=False)

print("Exported files:")
print(f"  oof_tabm_pwl_v2.npy: {os.path.getsize('oof_tabm_pwl_v2.npy') / (1024*1024):.2f} MB")
print(f"  test_tabm_pwl_v2.npy: {os.path.getsize('test_tabm_pwl_v2.npy') / (1024*1024):.2f} MB")
print(f"  submission.csv: {os.path.getsize('submission.csv') / (1024*1024):.2f} MB")
print("All tasks completed successfully.")
"""))

    nb.cells = cells
    out_file = Path(__file__).resolve().parent / "model.ipynb"
    with open(out_file, "w") as f:
        nbf.write(nb, f)
    print(f"Successfully generated TabM PWL V2 notebook at: {out_file}")

if __name__ == "__main__":
    build_tabm_v2_notebook()
