"""
Evaluate V3 Guardrail Ensembles with RealMLP V3, TabM V3, GPT-2, and Tree Bags.
Computes 5-fold nested cross-validation ROC-AUC and PR-AUC, with 4-zone zero-fraud manifold.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LinearRegression
from scipy.stats import rankdata
from scipy.special import logit

U = lambda v: (rankdata(v) - 0.5) / len(v)
L = lambda v: logit(np.clip(U(v), 1e-6, 1 - 1e-6))

def apply_rank_preserving_manifold(preds, mask_zero):
    n_total = len(preds)
    n_zero = mask_zero.sum()
    ranks = np.zeros(n_total, dtype=np.float64)
    ranks[mask_zero] = (rankdata(preds[mask_zero]) - 0.5) / n_total
    ranks[~mask_zero] = (n_zero + rankdata(preds[~mask_zero]) - 0.5) / n_total
    return ranks

print("=" * 70)
print("🛡️ Evaluating V3 Multi-Family Convex Ensembles")
print("=" * 70)

# 1. Load train labels and compute 4-zone manifold
train_df = pd.read_csv('pgs-s6e11/dataset/train.csv', usecols=[
    'is_fraud', 'velocity_score', 'merchant_risk_score',
    'cvv_retry_count', 'prior_disputes', 'is_new_merchant',
    'is_foreign_transaction', 'used_vpn', 'ip_country_mismatch',
    'billing_shipping_mismatch', 'is_ai_generated_scam_attempt'
])
y = train_df['is_fraud'].values
n_train = len(y)

clean = ((train_df['cvv_retry_count'] == 0) & (train_df['prior_disputes'] == 0) & 
         (~train_df['is_new_merchant'].astype(bool)) & (~train_df['is_foreign_transaction'].astype(bool)) & 
         (~train_df['used_vpn'].astype(bool)) & (~train_df['ip_country_mismatch'].astype(bool)) & 
         (~train_df['billing_shipping_mismatch'].astype(bool)) & (~train_df['is_ai_generated_scam_attempt'].astype(bool))).to_numpy()
zone1 = train_df['merchant_risk_score'].to_numpy() < 13.1
zone2 = (train_df['velocity_score'].to_numpy() <= 4.1) & (train_df['merchant_risk_score'].to_numpy() <= 20.0)
zone34 = clean & (((train_df['velocity_score'].to_numpy() <= 4.1) & (train_df['merchant_risk_score'].to_numpy() <= 33.8)) |
                  ((train_df['velocity_score'].to_numpy() <= 6.0) & (train_df['merchant_risk_score'].to_numpy() <= 25.0)))
zall_tr = zone1 | zone2 | zone34

# Load verified models
models = {
    'tree_bag10': np.load('pgs-s6e11/external_models/oof_bank/oof_gb_bag10.npy'),
    'tabm_v3': np.load('pgs-s6e11/code/tabm_gpu_v3/output/oof_tabm_pwl_v3.npy'),
    'gpt2_s2026': np.load('pgs-s6e11/external_models/gpt2_seed2026/gpt2_s2026_oof_5fold.npy'),
    'rmlp_bpe': np.load('pgs-s6e11/external_models/weni_bpe_pack/oof_weni_realmlp_bpe_v2.npy'),
    'rmlp_v3': np.load('pgs-s6e11/code/realmlp_gpu_v3/output/oof_realmlp_v3.npy'),
    'bpe_cat_d5': np.load('pgs-s6e11/external_models/weni_bpe_pack/oof_weni_bpe_cat_d5.npy'),
    'ebm_bag': np.load('pgs-s6e11/external_models/oof_bank/oof_ebm_all_bag.npy'),
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_ensemble(model_dict, name):
    cols = list(model_dict.keys())
    X = np.column_stack([L(model_dict[c]) for c in cols])
    
    oof_preds = np.zeros(n_train, dtype=np.float64)
    fold_weights = []
    
    for f, (trn_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X[trn_idx], y[trn_idx]
        X_va, y_va = X[val_idx], y[val_idx]
        
        nnls = LinearRegression(positive=True, fit_intercept=False)
        nnls.fit(X_tr, y_tr)
        w = nnls.coef_
        if w.sum() > 0:
            w = w / w.sum()
        else:
            w = np.ones(len(cols)) / len(cols)
        fold_weights.append(w)
        
        va_raw = X_va @ w
        va_cal = apply_rank_preserving_manifold(va_raw, zall_tr[val_idx])
        oof_preds[val_idx] = va_cal
        
    mean_w = np.mean(fold_weights, axis=0)
    score_roc = roc_auc_score(y, oof_preds)
    score_pr = average_precision_score(y, oof_preds)
    
    print(f"\n--- {name} ---")
    print(f"5-Fold Nested CV ROC-AUC: {score_roc:.6f}")
    print(f"5-Fold Nested CV PR-AUC:  {score_pr:.6f}")
    print("Mean Weights:")
    for col, w in zip(cols, mean_w):
        print(f"  {col:15s}: {w*100:5.2f}%")
        
    return score_roc, score_pr, mean_w

# Config 1: Previous 4-Pillar (Tree Bag 10 + TabM V3 + GPT2 + RealMLP BPE)
evaluate_ensemble({
    'tree_bag10': models['tree_bag10'],
    'tabm_v3': models['tabm_v3'],
    'gpt2_s2026': models['gpt2_s2026'],
    'rmlp_bpe': models['rmlp_bpe'],
}, "Config 1: 4-Pillar with RealMLP BPE")

# Config 2: 4-Pillar replacing RealMLP BPE with RealMLP V3
evaluate_ensemble({
    'tree_bag10': models['tree_bag10'],
    'tabm_v3': models['tabm_v3'],
    'gpt2_s2026': models['gpt2_s2026'],
    'rmlp_v3': models['rmlp_v3'],
}, "Config 2: 4-Pillar with RealMLP V3")

# Config 3: 5-Pillar Pentad (Tree Bag 10 + TabM V3 + GPT2 + RealMLP BPE + RealMLP V3)
evaluate_ensemble({
    'tree_bag10': models['tree_bag10'],
    'tabm_v3': models['tabm_v3'],
    'gpt2_s2026': models['gpt2_s2026'],
    'rmlp_bpe': models['rmlp_bpe'],
    'rmlp_v3': models['rmlp_v3'],
}, "Config 3: 5-Pillar Pentad (Dual Neural: TabM V3 + RealMLP V3 + BPE)")

# Config 4: 7-Pillar Multi-Family Sovereign Ensemble
evaluate_ensemble({
    'tree_bag10': models['tree_bag10'],
    'tabm_v3': models['tabm_v3'],
    'gpt2_s2026': models['gpt2_s2026'],
    'rmlp_bpe': models['rmlp_bpe'],
    'rmlp_v3': models['rmlp_v3'],
    'bpe_cat_d5': models['bpe_cat_d5'],
    'ebm_bag': models['ebm_bag'],
}, "Config 4: 7-Pillar Multi-Family Sovereign Ensemble")
