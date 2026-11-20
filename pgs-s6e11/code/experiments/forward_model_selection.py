"""
Forward Model Selection on 5-Fold Nested Cross-Validation.
Starts with our core pillars (Tree Bag 10, GPT-2 s2026, TabM V3, RealMLP V3, RealMLP BPE)
and iteratively tests adding each of the remaining 79 models from oof_bank, dual_nn, and weni
to see which model strictly improves 5-fold nested CV ROC-AUC and PR-AUC.
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LinearRegression
from scipy.stats import rankdata
from scipy.special import logit

DATA_DIR = Path('pgs-s6e11/dataset')

U = lambda v: (rankdata(v) - 0.5) / len(v)
L = lambda v: logit(np.clip(U(v), 1e-6, 1 - 1e-6))

def apply_rank_preserving_manifold(preds, mask_zero):
    n_total = len(preds)
    n_zero = mask_zero.sum()
    ranks = np.zeros(n_total, dtype=np.float64)
    ranks[mask_zero] = (rankdata(preds[mask_zero]) - 0.5) / n_total
    ranks[~mask_zero] = (n_zero + rankdata(preds[~mask_zero]) - 0.5) / n_total
    return ranks

print("=" * 75)
print("🔍 Forward Model Selection on 5-Fold Nested CV (5-Zone Manifold)")
print("=" * 75)

train_df = pd.read_csv(DATA_DIR / 'train.csv')
y = train_df['is_fraud'].values
n_train = len(y)

# 5-Zone Manifold
clean_tr = ((train_df['cvv_retry_count'] == 0) & (train_df['prior_disputes'] == 0) & 
            (~train_df['is_new_merchant'].astype(bool)) & (~train_df['is_foreign_transaction'].astype(bool)) & 
            (~train_df['used_vpn'].astype(bool)) & (~train_df['ip_country_mismatch'].astype(bool)) & 
            (~train_df['billing_shipping_mismatch'].astype(bool)) & (~train_df['is_ai_generated_scam_attempt'].astype(bool))).to_numpy()

zone1_tr = train_df['merchant_risk_score'].to_numpy() < 13.1
zone2_tr = (train_df['velocity_score'].to_numpy() <= 4.1) & (train_df['merchant_risk_score'].to_numpy() <= 20.0)
zone34_tr = clean_tr & (((train_df['velocity_score'].to_numpy() <= 4.1) & (train_df['merchant_risk_score'].to_numpy() <= 33.8)) |
                        ((train_df['velocity_score'].to_numpy() <= 6.0) & (train_df['merchant_risk_score'].to_numpy() <= 25.0)))
zone5_tr = clean_tr & (train_df['velocity_score'].to_numpy() <= 5.0) & (train_df['merchant_risk_score'].to_numpy() <= 35.0) & (train_df['amount_usd'].to_numpy() <= 30.0)
zall_5 = zone1_tr | zone2_tr | zone34_tr | zone5_tr

# Base 4 pillars
base_models = {
    'tree_bag10': L(np.load('pgs-s6e11/external_models/oof_bank/oof_gb_bag10.npy')),
    'gpt2_s2026': L(np.load('pgs-s6e11/external_models/gpt2_seed2026/gpt2_s2026_oof_5fold.npy')),
    'tabm_v3': L(np.load('pgs-s6e11/code/tabm_gpu_v3/output/oof_tabm_pwl_v3.npy')),
    'rmlp_bpe': L(np.load('pgs-s6e11/external_models/weni_bpe_pack/oof_weni_realmlp_bpe_v2.npy')),
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_cols(col_dict):
    X = np.column_stack(list(col_dict.values()))
    K = X.shape[1]
    nested_preds = np.zeros(n_train, dtype=np.float64)
    
    for f, (trn_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X[trn_idx], y[trn_idx]
        X_va, y_va = X[val_idx], y[val_idx]
        
        nnls = LinearRegression(positive=True, fit_intercept=False)
        nnls.fit(X_tr, y_tr)
        w = nnls.coef_
        if w.sum() > 0:
            w = w / w.sum()
        else:
            w = np.ones(K) / K
        va_raw = X_va @ w
        nested_preds[val_idx] = apply_rank_preserving_manifold(va_raw, zall_5[val_idx])
        
    return roc_auc_score(y, nested_preds), average_precision_score(y, nested_preds)

baseline_auc, baseline_pr = evaluate_cols(base_models)
print(f"Baseline 4-Pillar 5-Fold Nested CV ROC-AUC: {baseline_auc:.6f} | PR-AUC: {baseline_pr:.6f}")

# Candidate pool from oof_bank, dual_nn, weni, realmlp_v3
pool = {}

# 1. RealMLP V3
pool['realmlp_v3'] = L(np.load('pgs-s6e11/code/realmlp_gpu_v3/output/oof_realmlp_v3.npy'))

# 2. Dual NN
dn = Path('pgs-s6e11/code/dual_neural_multiseed/output')
for p in dn.glob('oof_*.npy'):
    pool[f'dual_nn:{p.stem[4:]}'] = L(np.load(p))

# 3. Weni BPE
weni = Path('pgs-s6e11/external_models/weni_bpe_pack')
for p in weni.glob('oof_*.npy'):
    name = f'weni:{p.stem[4:]}'
    if 'realmlp_bpe_v2' not in name:
        pool[name] = L(np.load(p))

# 4. Top from oof_bank
bank = Path('pgs-s6e11/external_models/oof_bank')
for p in sorted(bank.glob('oof_*.npy')):
    name = f'bank:{p.stem[4:]}'
    if 'gb_bag10' not in name:
        pool[name] = L(np.load(p))

print(f"\nEvaluating {len(pool)} candidate models for additive CV gain...")

improvements = []
for name, arr in pool.items():
    test_dict = dict(base_models)
    test_dict[name] = arr
    auc, pr = evaluate_cols(test_dict)
    delta_auc = (auc - baseline_auc) * 1e5
    if delta_auc > 0:
        improvements.append((name, auc, pr, delta_auc))
        print(f"  GAIN: {name:32s} -> CV: {auc:.6f} (+{delta_auc:5.2f} pts) | PR-AUC: {pr:.6f}")

improvements.sort(key=lambda x: x[3], reverse=True)
print("\n" + "=" * 75)
print(f"Top CV-Boosting Additive Models (Baseline: {baseline_auc:.6f}):")
for name, auc, pr, delta in improvements[:15]:
    print(f"  +{delta:5.2f} pts | {name:32s} | New CV: {auc:.6f} | PR: {pr:.6f}")
