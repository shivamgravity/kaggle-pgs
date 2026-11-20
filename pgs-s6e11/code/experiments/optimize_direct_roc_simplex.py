"""
Direct ROC-AUC Simplex Optimization on Multi-Family OOF Predictions.
Optimizes weights directly to maximize 5-Fold Stratified ROC-AUC using Nelder-Mead / SLSQP
with 5-Zone Deterministic Zero-Fraud Manifold calibration.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import rankdata
from scipy.optimize import minimize

DATA_DIR = Path('pgs-s6e11/dataset')

U = lambda v: (rankdata(v) - 0.5) / len(v)

def apply_rank_preserving_manifold(preds, mask_zero):
    n_total = len(preds)
    n_zero = mask_zero.sum()
    ranks = np.zeros(n_total, dtype=np.float64)
    ranks[mask_zero] = (rankdata(preds[mask_zero]) - 0.5) / n_total
    ranks[~mask_zero] = (n_zero + rankdata(preds[~mask_zero]) - 0.5) / n_total
    return ranks

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
zall_5_tr = zone1_tr | zone2_tr | zone34_tr | zone5_tr

# Distinct Family Pillars
pillars = {
    'Tree Bag 10': U(np.load('pgs-s6e11/external_models/oof_bank/oof_gb_bag10.npy')),
    'GPT-2 s2026': U(np.load('pgs-s6e11/external_models/gpt2_seed2026/gpt2_s2026_oof_5fold.npy')),
    'RealMLP V3': U(np.load('pgs-s6e11/code/realmlp_gpu_v3/output/oof_realmlp_v3.npy')),
    'RealMLP BPE': U(np.load('pgs-s6e11/external_models/weni_bpe_pack/oof_weni_realmlp_bpe_v2.npy')),
    'TabM V3': U(np.load('pgs-s6e11/code/tabm_gpu_v3/output/oof_tabm_pwl_v3.npy')),
    'Dual NN': U(np.load('pgs-s6e11/code/dual_neural_multiseed/output/oof_dual_neural_bag3.npy')),
    'EBM Bag': U(np.load('pgs-s6e11/external_models/oof_bank/oof_ebm_all_bag.npy')),
}

names = list(pillars.keys())
P_mat = np.column_stack([pillars[k] for k in names])
M = len(names)

print("=" * 70)
print(f"🎯 Direct ROC-AUC Simplex Optimization on {M} Diverse Model Families")
print("=" * 70)

# Nested 5-Fold Cross Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
nested_oof = np.zeros(n_train, dtype=np.float64)
fold_weights = []

for f, (trn_idx, val_idx) in enumerate(skf.split(P_mat, y)):
    P_tr, y_tr = P_mat[trn_idx], y[trn_idx]
    P_va, y_va = P_mat[val_idx], y[val_idx]
    z_va = zall_5_tr[val_idx]

    # Objective: Minimize negative ROC-AUC
    # Use softmax parametrization w = softmax(theta) for unconstrained optimization
    def obj(theta):
        w = np.exp(theta) / np.sum(np.exp(theta))
        pred = P_tr @ w
        return -roc_auc_score(y_tr, pred)

    init_theta = np.zeros(M)
    res = minimize(obj, init_theta, method='Nelder-Mead', options={'maxiter': 250, 'xatol': 1e-3, 'fatol': 1e-5})
    w_opt = np.exp(res.x) / np.sum(np.exp(res.x))
    fold_weights.append(w_opt)

    va_pred = P_va @ w_opt
    nested_oof[val_idx] = apply_rank_preserving_manifold(va_pred, z_va)
    fold_auc = roc_auc_score(y_va, nested_oof[val_idx])
    fold_pr = average_precision_score(y_va, nested_oof[val_idx])
    print(f"  Fold {f+1} ROC-AUC: {fold_auc:.6f} | PR-AUC: {fold_pr:.6f}")

mean_w = np.mean(fold_weights, axis=0)
overall_auc = roc_auc_score(y, nested_oof)
overall_pr = average_precision_score(y, nested_oof)

print("\n" + "=" * 70)
print(f"🏆 Overall 5-Fold Nested CV ROC-AUC: {overall_auc:.6f}")
print(f"🏆 Overall 5-Fold Nested CV PR-AUC:  {overall_pr:.6f}")
print("=" * 70)

print("\nMean Optimal Weights Across Folds:")
for name, w in zip(names, mean_w):
    print(f"  {name:15s}: {w*100:5.2f}%")
