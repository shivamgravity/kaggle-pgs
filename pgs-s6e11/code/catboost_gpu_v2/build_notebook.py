#!/usr/bin/env python3
"""
Build Kaggle GPU Notebook for Multi-Seed CatBoost GPU 5-Fold Ensemble (Phase 2 SOTA).
Features:
  - Dual-Seed Multi-Fold CatBoostClassifier on Tesla T4 GPU
  - CTGAN Decimal Residues & Multi-Scale Interaction Keys
  - GPU Target Statistics on high-cardinality features
  - Calibrated with the 5-Zone Deterministic Zero-Fraud Manifold (ZALL_5)
  - Exports: oof_catboost_gpu_v2.npy, test_catboost_gpu_v2.npy, catboost_gpu_v2_submission.csv
"""

from pathlib import Path
import json
import nbformat as nbf

def build_catboost_v2_notebook():
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

    # Cell 1: Environment & Setup
    cells.append(nbf.v4.new_code_cell("""# Step 1: Environment & Imports
import os
import gc
import glob
import time
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
import catboost as cb
from catboost import CatBoostClassifier, Pool

warnings.filterwarnings('ignore')

print("CatBoost version:", cb.__version__)

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything(42)
"""))

    # Cell 2: Load Data
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

y = train_df['is_fraud'].values.astype(int)
n_train = len(train_df)
n_test = len(test_df)
print(f"Global fraud rate: {y.mean():.4%} ({y.sum():,} frauds out of {len(y):,} rows)")
"""))

    # Cell 3: 5-Zone Deterministic Zero-Fraud Manifold
    cells.append(nbf.v4.new_code_cell("""# Step 3: Compute 5-Zone Zero-Fraud Manifold (ZALL_5)
def compute_zall5(df):
    clean = ((df['cvv_retry_count'] == 0) & (df['prior_disputes'] == 0) & 
             (~df['is_new_merchant'].astype(bool)) & (~df['is_foreign_transaction'].astype(bool)) & 
             (~df['used_vpn'].astype(bool)) & (~df['ip_country_mismatch'].astype(bool)) & 
             (~df['billing_shipping_mismatch'].astype(bool)) & (~df['is_ai_generated_scam_attempt'].astype(bool))).to_numpy()

    zone1 = df['merchant_risk_score'].to_numpy() < 13.1
    zone2 = (df['velocity_score'].to_numpy() <= 4.1) & (df['merchant_risk_score'].to_numpy() <= 20.0)
    zone34 = clean & (((df['velocity_score'].to_numpy() <= 4.1) & (df['merchant_risk_score'].to_numpy() <= 33.8)) |
                      ((df['velocity_score'].to_numpy() <= 6.0) & (df['merchant_risk_score'].to_numpy() <= 25.0)))
    zone5 = clean & (df['velocity_score'].to_numpy() <= 5.0) & (df['merchant_risk_score'].to_numpy() <= 35.0) & (df['amount_usd'].to_numpy() <= 30.0)
    return zone1 | zone2 | zone34 | zone5

def apply_rank_preserving_manifold(preds, mask_zero):
    n_total = len(preds)
    n_zero = mask_zero.sum()
    ranks = np.zeros(n_total, dtype=np.float64)
    ranks[mask_zero] = (rankdata(preds[mask_zero]) - 0.5) / n_total
    ranks[~mask_zero] = (n_zero + rankdata(preds[~mask_zero]) - 0.5) / n_total
    return ranks

zall_5_tr = compute_zall5(train_df)
zall_5_te = compute_zall5(test_df)
print(f"5-Zone Manifold: {zall_5_tr.sum():,} train rows (frauds: {y[zall_5_tr].sum()}), {zall_5_te.sum():,} test accounts ({zall_5_te.mean():.2%})")
"""))

    # Cell 4: Domain & CTGAN Decimal Residue Feature Engineering
    cells.append(nbf.v4.new_code_cell("""# Step 4: CTGAN Decimal Residues & High-Cardinality Interaction Features
t0 = time.time()

dfs = [train_df, test_df]
for df in dfs:
    # 1. Decimal & Quantization Residues
    amt = df['amount_usd']
    df['amt_cents'] = np.round((amt % 1) * 100).astype(int).clip(0, 99).astype(str)
    df['amt_tenths'] = np.floor((amt % 1) * 10).astype(int).clip(0, 9).astype(str)
    df['amt_last_digit'] = (np.floor(amt).astype(int) % 10).clip(0, 9).astype(str)
    df['amt_mod10'] = (amt % 10.0).astype(np.float32)
    df['amt_mod50'] = (amt % 50.0).astype(np.float32)
    df['is_round_dollar'] = (amt % 1.0 == 0).astype(str)
    df['is_99_cents'] = (df['amt_cents'] == '99').astype(str)

    # Velocity residues
    vel = df['velocity_score']
    df['vel_residue_0p25'] = (vel % 0.25).astype(np.float32)
    df['vel_residue_0p1'] = (vel % 0.1).astype(np.float32)
    df['vel_is_integer'] = (vel % 1.0 == 0).astype(str)

    # Merchant risk residues
    mrs = df['merchant_risk_score']
    df['mrs_decile'] = (mrs // 10).astype(int).clip(0, 10).astype(str)
    df['mrs_residue_1p0'] = (mrs % 1.0).astype(np.float32)

    # Distance and Time residues
    df['dist_mod10'] = (df['distance_from_home_km'] % 10.0).astype(np.float32)
    df['hours_mod24'] = (df['hours_since_last_txn'] % 24.0).astype(np.float32)
    df['hours_mod6'] = (df['hours_since_last_txn'] % 6.0).astype(np.float32)

    # High-Order Discrete Interactions
    df['cat_x_auth'] = df['merchant_category'].astype(str) + '_' + df['auth_method'].astype(str)
    df['cat_x_device'] = df['merchant_category'].astype(str) + '_' + df['device_type'].astype(str)
    df['cat_x_card'] = df['merchant_category'].astype(str) + '_' + df['card_type'].astype(str)
    df['channel_x_device'] = df['channel'].astype(str) + '_' + df['device_type'].astype(str)
    df['sec_triplet'] = (df['is_foreign_transaction'].astype(str) + '_' + 
                         df['used_vpn'].astype(str) + '_' + 
                         df['ip_country_mismatch'].astype(str))
    df['cents_x_cat'] = df['amt_cents'].astype(str) + '_' + df['merchant_category'].astype(str)

cat_features = [
    'merchant_category', 'card_type', 'auth_method', 'channel', 'device_type',
    'time_of_day_hour', 'day_of_week', 'is_new_merchant',
    'is_foreign_transaction', 'used_vpn', 'ip_country_mismatch',
    'billing_shipping_mismatch', 'is_ai_generated_scam_attempt',
    'amt_cents', 'amt_tenths', 'amt_last_digit', 'is_round_dollar', 'is_99_cents',
    'vel_is_integer', 'mrs_decile',
    'cat_x_auth', 'cat_x_device', 'cat_x_card', 'channel_x_device', 'sec_triplet', 'cents_x_cat'
]

num_features = [
    'amount_usd', 'hours_since_last_txn', 'txn_count_last_24h',
    'distance_from_home_km', 'card_age_months', 'customer_age',
    'account_balance_usd', 'cvv_retry_count', 'velocity_score',
    'merchant_risk_score', 'prior_disputes',
    'amt_mod10', 'amt_mod50', 'vel_residue_0p25', 'vel_residue_0p1',
    'mrs_residue_1p0', 'dist_mod10', 'hours_mod24', 'hours_mod6'
]

features = cat_features + num_features

for col in cat_features:
    train_df[col] = train_df[col].astype(str).fillna('missing')
    test_df[col] = test_df[col].astype(str).fillna('missing')

for col in num_features:
    train_df[col] = train_df[col].astype(np.float32).fillna(0.0)
    test_df[col] = test_df[col].astype(np.float32).fillna(0.0)

print(f"Total features: {len(features)} ({len(cat_features)} categoricals, {len(num_features)} numericals) engineered in {time.time() - t0:.2f}s")
"""))

    # Cell 5: Dual-Seed Multi-Fold GPU Training
    cells.append(nbf.v4.new_code_cell("""# Step 5: Multi-Seed 5-Fold Stratified CatBoost Training on GPU
seeds = [42, 2026]
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds_raw = np.zeros(n_train, dtype=np.float64)
test_preds_raw = np.zeros(n_test, dtype=np.float64)

X_test_pool = Pool(test_df[features], cat_features=cat_features)

total_runs = len(seeds) * n_splits
run_count = 0
overall_start = time.time()

for seed_idx, seed in enumerate(seeds):
    print("=" * 70)
    print(f"🔥 Training CatBoost GPU Stream with Seed {seed} ({seed_idx+1}/{len(seeds)})")
    print("=" * 70)
    
    seed_oof = np.zeros(n_train, dtype=np.float64)
    seed_test = np.zeros(n_test, dtype=np.float64)
    
    cb_params = {
        'iterations': 3000,
        'learning_rate': 0.04,
        'depth': 7,
        'l2_leaf_reg': 5.0,
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'task_type': 'GPU',
        'random_seed': seed,
        'early_stopping_rounds': 150,
        'verbose': 300
    }
    
    for fold, (trn_idx, val_idx) in enumerate(skf.split(train_df, y)):
        run_count += 1
        t_fold = time.time()
        
        X_tr = train_df.iloc[trn_idx][features]
        y_tr = y[trn_idx]
        X_va = train_df.iloc[val_idx][features]
        y_va = y[val_idx]
        
        trn_pool = Pool(X_tr, y_tr, cat_features=cat_features)
        val_pool = Pool(X_va, y_va, cat_features=cat_features)
        
        model = CatBoostClassifier(**cb_params)
        model.fit(trn_pool, eval_set=val_pool, use_best_model=True)
        
        val_pred = model.predict_proba(val_pool)[:, 1]
        seed_oof[val_idx] = val_pred
        
        fold_auc = roc_auc_score(y_va, val_pred)
        fold_pr = average_precision_score(y_va, val_pred)
        print(f"Fold {fold+1} [Seed {seed}] in {time.time() - t_fold:.1f}s | ROC-AUC: {fold_auc:.6f} | PR-AUC: {fold_pr:.6f}")
        
        test_pred_fold = model.predict_proba(X_test_pool)[:, 1]
        seed_test += test_pred_fold / n_splits
        
        del model, trn_pool, val_pool
        gc.collect()
        
    seed_auc = roc_auc_score(y, seed_oof)
    seed_pr = average_precision_score(y, seed_oof)
    print(f">> Seed {seed} Overall OOF ROC-AUC: {seed_auc:.6f} | PR-AUC: {seed_pr:.6f}")
    
    oof_preds_raw += seed_oof / len(seeds)
    test_preds_raw += seed_test / len(seeds)

print("\\n" + "=" * 70)
print(f"🏁 Multi-Seed CatBoost Training Finished in {time.time() - overall_start:.1f}s")
print("=" * 70)
"""))

    # Cell 6: Evaluation & Manifold Calibration
    cells.append(nbf.v4.new_code_cell("""# Step 6: 5-Zone Manifold Calibration & Metrics
raw_auc = roc_auc_score(y, oof_preds_raw)
raw_pr = average_precision_score(y, oof_preds_raw)

oof_calibrated = apply_rank_preserving_manifold(oof_preds_raw, zall_5_tr)
cal_auc = roc_auc_score(y, oof_calibrated)
cal_pr = average_precision_score(y, oof_calibrated)

print(f"=== CatBoost GPU Multi-Seed Overall Results ===")
print(f"Raw OOF ROC-AUC:        {raw_auc:.6f} | PR-AUC: {raw_pr:.6f}")
print(f"Calibrated OOF ROC-AUC: {cal_auc:.6f} | PR-AUC: {cal_pr:.6f} (+{(cal_auc - raw_auc)*1e5:.1f} pts from 5-Zone)")

test_calibrated = apply_rank_preserving_manifold(test_preds_raw, zall_5_te)
"""))

    # Cell 7: Save Outputs
    cells.append(nbf.v4.new_code_cell("""# Step 7: Save Output Arrays and Submission
OUT_DIR = Path('output')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Save arrays
np.save(OUT_DIR / 'oof_catboost_gpu_v2.npy', oof_calibrated)
np.save(OUT_DIR / 'test_catboost_gpu_v2.npy', test_calibrated)
np.save(OUT_DIR / 'oof_catboost_gpu_v2_raw.npy', oof_preds_raw)
np.save(OUT_DIR / 'test_catboost_gpu_v2_raw.npy', test_preds_raw)

# Generate submission CSV
sub_df = sample_sub[['id']].copy()
sub_df['is_fraud'] = test_calibrated
sub_df.to_csv(OUT_DIR / 'catboost_gpu_v2_submission.csv', index=False)
sub_df.to_csv('submission.csv', index=False)

print(f"Saved oof and test arrays to {OUT_DIR}/")
print(f"Saved submission: {sub_df.shape} rows, nulls: {sub_df.isnull().sum().to_dict()}")
print(sub_df.head(5))
"""))

    nb.cells = cells

    out_dir = Path('pgs-s6e11/code/catboost_gpu_v2')
    out_dir.mkdir(parents=True, exist_ok=True)
    nb_path = out_dir / 'model.ipynb'

    with open(nb_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

    meta = {
        "id": "shivamgravity/pgs-s6e11-catboost-v2",
        "title": "pgs-s6e11-catboost-v2",
        "code_file": "model.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "false",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [],
        "competition_sources": [
            "playground-series-s6e11"
        ],
        "kernel_sources": [],
        "model_sources": []
    }

    with open(out_dir / 'kernel-metadata.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print(f"✅ Successfully built CatBoost GPU V2 notebook at: {nb_path}")

if __name__ == '__main__':
    build_catboost_v2_notebook()
