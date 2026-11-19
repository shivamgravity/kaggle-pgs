"""
Non-Negative Simplex Meta-Stacker with 5-Fold Nested CV for PGS-S6E11.

Tests the hypothesis:
  A non-negative convex combination (w_i >= 0, sum w_i = 1) across diverse model families
  (Trees, Neural PWL TabM, Causal LM, BPE models, GLMs) with 4-zone zero-fraud calibration
  maximizes out-of-fold generalization and produces higher private test stability than
  unconstrained wide linear regression.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.optimize import minimize
from scipy.stats import rankdata
from scipy.special import logit

DATA_DIR = Path('pgs-s6e11/dataset')
BANK_DIR = Path('pgs-s6e11/external_models/oof_bank')
WENI_DIR = Path('pgs-s6e11/external_models/weni_bpe_pack')
TABM_DIR = Path('pgs-s6e11/code/tabm_gpu_v2/output')
GPT2_DIR = Path('pgs-s6e11/external_models/gpt2_seed2026')

U = lambda v: (rankdata(v) - 0.5) / len(v)

def main():
    print("=" * 80)
    print("🧪 Running Non-Negative Simplex Meta-Stacker with 5-Fold Nested CV")
    print("=" * 80)

    # 1. Load train labels and compute 4-zone manifold
    tr = pd.read_csv(DATA_DIR / 'train.csv', usecols=['id', 'is_fraud', 'velocity_score', 'merchant_risk_score',
                                                     'cvv_retry_count', 'prior_disputes', 'is_new_merchant',
                                                     'is_foreign_transaction', 'used_vpn', 'ip_country_mismatch',
                                                     'billing_shipping_mismatch', 'is_ai_generated_scam_attempt'])
    y = tr['is_fraud'].values

    clean = ((tr['cvv_retry_count'] == 0) & (tr['prior_disputes'] == 0) & 
             (~tr['is_new_merchant'].astype(bool)) & (~tr['is_foreign_transaction'].astype(bool)) & 
             (~tr['used_vpn'].astype(bool)) & (~tr['ip_country_mismatch'].astype(bool)) & 
             (~tr['billing_shipping_mismatch'].astype(bool)) & (~tr['is_ai_generated_scam_attempt'].astype(bool))).to_numpy()

    zone1 = tr['merchant_risk_score'].to_numpy() < 13.1
    zone2 = (tr['velocity_score'].to_numpy() <= 4.1) & (tr['merchant_risk_score'].to_numpy() <= 20.0)
    zone34 = clean & (((tr['velocity_score'].to_numpy() <= 4.1) & (tr['merchant_risk_score'].to_numpy() <= 33.8)) |
                      ((tr['velocity_score'].to_numpy() <= 6.0) & (tr['merchant_risk_score'].to_numpy() <= 25.0)))
    zall = zone1 | zone2 | zone34
    print(f"4-Zone Zero-Fraud rows: {zall.sum():,} ({zall.mean():.2%}), frauds inside: {y[zall].sum()}")

    # 2. Select Diverse Member Models
    # Include best from diverse families
    models = {}
    
    # Non-tree neural & LM
    models['TabM_PWL_V2'] = np.load(TABM_DIR / 'oof_tabm_pwl_v2.npy')
    models['GPT2_2026'] = np.load(GPT2_DIR / 'gpt2_s2026_oof_5fold.npy')
    models['MegaLM'] = np.load('pgs-s6e11/external_models/megalm_oof_sha181c.npy')
    
    # BPE-digit representations
    models['BPE_LGB_L4'] = np.load(WENI_DIR / 'oof_weni_bpe_lgb_l4.npy')
    models['BPE_CatBoost'] = np.load(WENI_DIR / 'oof_weni_bpe_cat_d5.npy')
    models['BPE_RealMLP'] = np.load(WENI_DIR / 'oof_weni_realmlp_bpe_v2.npy')
    models['BPE_GLM'] = np.load(WENI_DIR / 'oof_weni_bpe_glm.npy')
    
    # Top tree ensembles from bank
    for bank_name in ['gb_bag5', 'gb_bag10', 'gbx_bag5', 'gbn_bag5_w512', 'lgb_diglead_l3']:
        models[bank_name] = np.load(BANK_DIR / f'oof_{bank_name}.npy')

    names = list(models.keys())
    print(f"\nPooled {len(names)} diverse candidate models:")
    for name in names:
        auc = roc_auc_score(y, models[name])
        print(f"  - {name:20s}: OOF ROC-AUC = {auc:.6f}")

    # Rank-normalize each model into uniform marginals and convert to logits
    print("\nConverting all models into logit space...")
    L = lambda v: logit(np.clip(U(v), 1e-6, 1 - 1e-6))
    OOF_matrix = np.column_stack([L(models[k]) for k in names])

    # 3. 5-Fold Nested Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    nested_preds = np.zeros(len(y))
    nested_fold_aucs = []
    nested_fold_praucs = []
    fold_weights = []

    print("\n--- Running 5-Fold Nested Optimization (Logistic Regression & Nelder-Mead on Logits) ---")
    from sklearn.linear_model import LogisticRegression
    
    for f, (trn_idx, val_idx) in enumerate(skf.split(OOF_matrix, y)):
        X_trn, y_trn = OOF_matrix[trn_idx], y[trn_idx]
        X_val, y_val = OOF_matrix[val_idx], y[val_idx]
        
        # Fit regularized Logistic Regression on logits
        # Positive weights only (via L-BFGS with positive bounds or L1/L2)
        clf = LogisticRegression(C=0.1, max_iter=200, random_state=42, solver='lbfgs')
        clf.fit(X_trn, y_trn)
        
        w_raw = clf.coef_[0]
        # Zero out any negative weights to ensure monotone ensembling
        w_pos = np.maximum(0, w_raw)
        w_norm = w_pos / w_pos.sum()
        fold_weights.append(w_norm)

        val_blend = X_val @ w_norm
        val_calibrated = np.where(zall[val_idx], -1e9, val_blend)
        nested_preds[val_idx] = val_calibrated

        fold_auc = roc_auc_score(y_val, val_calibrated)
        fold_prauc = average_precision_score(y_val, val_calibrated)
        nested_fold_aucs.append(fold_auc)
        nested_fold_praucs.append(fold_prauc)

        print(f"Fold {f}: ROC-AUC = {fold_auc:.6f} | PR-AUC = {fold_prauc:.6f}")

    # Overall nested CV score
    overall_auc = roc_auc_score(y, nested_preds)
    overall_prauc = average_precision_score(y, nested_preds)
    mean_fold_auc = np.mean(nested_fold_aucs)
    mean_fold_prauc = np.mean(nested_fold_praucs)

    print("\n" + "=" * 80)
    print(f"🏆 5-Fold Nested Simplex Meta-Stacker Results:")
    print(f"  Overall OOF ROC-AUC : {overall_auc:.6f} (Mean Fold: {mean_fold_auc:.6f})")
    print(f"  Overall OOF PR-AUC  : {overall_prauc:.6f} (Mean Fold: {mean_fold_prauc:.6f})")
    print("=" * 80)

    # Average weights across folds
    mean_weights = np.mean(fold_weights, axis=0)
    weight_df = pd.DataFrame({'model': names, 'mean_weight': mean_weights}).sort_values('mean_weight', ascending=False)
    print("\nMean Simplex Weights across 5 Folds:")
    print(weight_df.to_string(index=False))

    # Save OOF predictions
    out_dir = Path('pgs-s6e11/oof_preds')
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / 'oof_simplex_meta_stack.npy', nested_preds)
    print(f"\nSaved nested OOF predictions to {out_dir / 'oof_simplex_meta_stack.npy'}")

if __name__ == '__main__':
    main()
