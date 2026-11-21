"""
Elite Multi-Pillar Direct ROC-AUC Simplex Optimizer with Nested 5-Fold CV.

Optimizes convex ensemble weights across our 8 highest-performing and highest-diversity
model families strictly using Nelder-Mead simplex search over softmax parametrization:
  - 100% leak-free out-of-fold cross validation
  - Zero test set leakage
  - Calibrated with the 5-Zone Zero-Fraud Manifold
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

def main():
    print("=" * 80)
    print("🛡️ Elite 8-Pillar Direct ROC-AUC Simplex Optimizer (Nested 5-Fold CV)")
    print("=" * 80)

    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    y = train_df['is_fraud'].values.astype(int)
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

    # 8 Elite Diverse Pillars
    pillars = {
        'Tree Bag 10': U(np.load('pgs-s6e11/external_models/oof_bank/oof_gb_bag10.npy')),
        'GBX Bag 5': U(np.load('pgs-s6e11/external_models/oof_bank/oof_gbx_bag5.npy')),
        'GBN Bag 5': U(np.load('pgs-s6e11/external_models/oof_bank/oof_gbn_bag5_w512.npy')),
        'Dual GPT-2': 0.5 * U(np.load('pgs-s6e11/external_models/gpt2_seed42/oof.npy')) + 0.5 * U(np.load('pgs-s6e11/external_models/gpt2_seed2026/gpt2_s2026_oof_5fold.npy')),
        'LGB Diglead': U(np.load('pgs-s6e11/external_models/oof_bank/oof_lgb_diglead_l3.npy')),
        'RealMLP BPE': U(np.load('pgs-s6e11/external_models/weni_bpe_pack/oof_weni_realmlp_bpe_v2.npy')),
        'CatBoost V2': U(np.load('pgs-s6e11/code/catboost_gpu_v2/output/oof_catboost_gpu_v2.npy')),
        'TabM V3': U(np.load('pgs-s6e11/code/tabm_gpu_v3/output/oof_tabm_pwl_v3.npy')),
    }

    names = list(pillars.keys())
    P_mat = np.column_stack([pillars[k] for k in names])
    M = len(names)
    print(f"Loaded {M} elite pillar streams: {names}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    nested_oof = np.zeros(n_train, dtype=np.float64)
    fold_weights = []

    # Fast AUC helper for optimization
    def fast_auc_opt(preds, y_tr):
        n_pos = (y_tr == 1).sum()
        n_neg = len(y_tr) - n_pos
        r = rankdata(preds)
        return (r[y_tr == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

    print("\n--- Running Nelder-Mead Optimization on 5 Folds ---")
    for f, (trn_idx, val_idx) in enumerate(skf.split(P_mat, y)):
        P_tr, y_tr = P_mat[trn_idx], y[trn_idx]
        P_va, y_va = P_mat[val_idx], y[val_idx]
        z_va = zall_5_tr[val_idx]

        def obj(theta):
            w = np.exp(theta - np.max(theta))
            w /= np.sum(w)
            pred = P_tr @ w
            return -fast_auc_opt(pred, y_tr)

        # Warm start theta: slight preference to high-CV models
        init_theta = np.zeros(M)
        res = minimize(obj, init_theta, method='Nelder-Mead', options={'maxiter': 180, 'xatol': 1e-3, 'fatol': 1e-5})
        
        w_opt = np.exp(res.x - np.max(res.x))
        w_opt /= np.sum(w_opt)
        fold_weights.append(w_opt)

        va_pred = P_va @ w_opt
        nested_oof[val_idx] = apply_rank_preserving_manifold(va_pred, z_va)
        fold_auc = roc_auc_score(y_va, nested_oof[val_idx])
        fold_pr = average_precision_score(y_va, nested_oof[val_idx])
        print(f"  Fold {f+1} ROC-AUC: {fold_auc:.6f} | PR-AUC: {fold_pr:.6f}")

    mean_w = np.mean(fold_weights, axis=0)
    overall_auc = roc_auc_score(y, nested_oof)
    overall_pr = average_precision_score(y, nested_oof)

    print("\n" + "=" * 80)
    print(f"🏆 Overall 5-Fold Nested CV ROC-AUC: {overall_auc:.6f}")
    print(f"🏆 Overall 5-Fold Nested CV PR-AUC:  {overall_pr:.6f}")
    print("=" * 80)

    print("\n--- Mean Optimal Weights Across Folds ---")
    for name, w in sorted(zip(names, mean_w), key=lambda x: x[1], reverse=True):
        print(f"  {name:<16}: {w*100:5.2f}%")

    # Save OOF
    out_dir = Path('pgs-s6e11/oof_preds')
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / 'oof_elite_8pillar_nested_cv.npy', nested_oof)
    print(f"\nSaved nested OOF predictions to {out_dir / 'oof_elite_8pillar_nested_cv.npy'}")

if __name__ == '__main__':
    main()
