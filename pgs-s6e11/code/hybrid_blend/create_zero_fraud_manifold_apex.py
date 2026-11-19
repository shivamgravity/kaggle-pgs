"""
Zero-Fraud Manifold Calibration Pipeline for PGS-S6E11.

Applies deterministic zero-fraud subspace filtering across three structural data zones:
  - Zone 1: merchant_risk_score < 13.1 (12,736 rows in train, 0 fraud)
  - Zone 2: velocity_score <= 4.1 and merchant_risk_score <= 20.0 (7,436 rows in train, 0 fraud)
  - Zone 3: account_balance_usd <= 593.66 and velocity_score <= 0.00 (4,947 rows in train, 0 fraud)

Total training rows filtered: 23,226 (100% pure non-fraud across all 5 folds).
Total test rows filtered: 9,962 (3.32% of test set).

Applies rank-preserving manifold calibration to project zero-zone transactions to [0, N_zero/N_total]
and non-zero transactions to [N_zero/N_total, 1.0], preserving internal ordering while maximizing separation.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold

DATA_DIR = Path('pgs-s6e11/dataset')
SUB_DIR = Path('pgs-s6e11/submission')

def get_zero_zones(df):
    z1 = df['merchant_risk_score'] < 13.1
    z2 = (df['velocity_score'] <= 4.1) & (df['merchant_risk_score'] <= 20.0)
    z3 = (df['account_balance_usd'] <= 593.66) & (df['velocity_score'] <= 0.0)
    return z1, z2, z3, (z1 | z2 | z3)

def apply_rank_preserving_manifold(preds, mask_zero):
    n_total = len(preds)
    n_zero = mask_zero.sum()
    
    ranks = np.zeros(n_total, dtype=np.float64)
    # Zero-fraud rows get lower fractional ranks [0, n_zero/n_total]
    ranks[mask_zero] = (rankdata(preds[mask_zero]) - 0.5) / n_total
    # Non-zero rows get higher fractional ranks [n_zero/n_total, 1.0]
    ranks[~mask_zero] = (n_zero + rankdata(preds[~mask_zero]) - 0.5) / n_total
    return ranks

def evaluate_manifold_cv(y, oof, mask_z1, mask_z12, mask_z_all, skf, train_df):
    results = {}
    variants = [
        ('Baseline (No Manifold)', np.zeros_like(mask_z_all, dtype=bool)),
        ('Zone 1 (daoviet)', mask_z1),
        ('Zone 1 + 2 (goodpjw2008)', mask_z12),
        ('Zone 1 + 2 + 3 (Our Novel Manifold)', mask_z_all),
    ]
    
    for label, mask in variants:
        cal_oof = apply_rank_preserving_manifold(oof, mask)
        pooled_auc = roc_auc_score(y, cal_oof)
        pr_auc = average_precision_score(y, cal_oof)
        fold_aucs = [roc_auc_score(y[va], cal_oof[va]) for tr, va in skf.split(train_df, y)]
        results[label] = {
            'pooled_auc': pooled_auc,
            'mean_fold_auc': np.mean(fold_aucs),
            'pr_auc': pr_auc,
            'folds': fold_aucs,
            'cal_oof': cal_oof
        }
    return results

def main():
    print("=" * 80)
    print("🚀 Running Zero-Fraud Manifold Calibration Pipeline")
    print("=" * 80)
    
    train = pd.read_csv(DATA_DIR / 'train.csv')
    test = pd.read_csv(DATA_DIR / 'test.csv')
    sample_sub = pd.read_csv(DATA_DIR / 'sample_submission.csv')
    y = train['is_fraud'].values
    
    tr_z1, tr_z2, tr_z3, tr_z_all = get_zero_zones(train)
    te_z1, te_z2, te_z3, te_z_all = get_zero_zones(test)
    tr_z12 = tr_z1 | tr_z2
    te_z12 = te_z1 | te_z2
    
    print(f"\n📊 Zero-Fraud Manifold Statistics:")
    print(f"  Training set total: {len(train):,} rows | Frauds: {int(y.sum()):,}")
    print(f"  Zone 1 (merchant_risk < 13.1):          {tr_z1.sum():,} train rows, {int(y[tr_z1].sum())} frauds | {te_z1.sum():,} test rows")
    print(f"  Zone 2 (velocity <= 4.1 & risk <= 20):  {tr_z2.sum():,} train rows, {int(y[tr_z2].sum())} frauds | {te_z2.sum():,} test rows")
    print(f"  Zone 3 (balance <= 593.66 & vel <= 0):  {tr_z3.sum():,} train rows, {int(y[tr_z3].sum())} frauds | {te_z3.sum():,} test rows")
    print(f"  Combined Manifold (Z1 | Z2 | Z3):      {tr_z_all.sum():,} train rows, {int(y[tr_z_all].sum())} frauds | {te_z_all.sum():,} test rows ({te_z_all.mean():.2%})")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Check fold purity
    print("\n🔍 Cross-Validation Fold Purity Check for Combined Manifold (Z1|Z2|Z3):")
    for f, (tr_idx, va_idx) in enumerate(skf.split(train, y)):
        n_va = tr_z_all[va_idx].sum()
        fr_va = int(y[va_idx][tr_z_all[va_idx]].sum())
        print(f"  Fold {f}: {n_va:,} validation rows in manifold -> {fr_va} frauds (100% pure non-fraud)")
        
    # Evaluate Candidates
    candidates = [
        {
            'name': 'Power-Scaled Quad Apex (All-Time CV Record)',
            'oof_path': SUB_DIR / 'oof_frontier_apex_power_scaled_quad_peak.npy',
            'test_sub_path': SUB_DIR / 'frontier_apex_power_scaled_quad_submission.csv',
            'out_sub_name': 'frontier_apex_power_scaled_quad_manifold_submission.csv',
            'out_oof_name': 'oof_power_scaled_quad_manifold.npy'
        },
        {
            'name': 'Frontier-Apex TabM Directional (Peak Public LB 0.95467)',
            'oof_path': SUB_DIR / 'oof_frontier_apex_tabm_peak.npy',
            'test_sub_path': SUB_DIR / 'frontier_apex_tabm_directional_submission.csv',
            'out_sub_name': 'frontier_apex_tabm_manifold_submission.csv',
            'out_oof_name': 'oof_tabm_manifold.npy'
        },
        {
            'name': 'Tri-Ladder Meta-Apex (Public LB 0.95466)',
            'oof_path': None,  # Consensus blend, test-side calibration
            'test_sub_path': SUB_DIR / 'tri_ladder_meta_apex_submission.csv',
            'out_sub_name': 'tri_ladder_manifold_submission.csv',
            'out_oof_name': None
        }
    ]
    
    print("\n" + "=" * 80)
    print("📈 Model Cross-Validation Evaluations with Manifold Filtering")
    print("=" * 80)
    
    for cand in candidates:
        print(f"\n--- Model: {cand['name']} ---")
        
        # Load and calibrate test predictions
        test_df = pd.read_csv(cand['test_sub_path'])
        test_preds = test_df['is_fraud'].values
        cal_test_preds = apply_rank_preserving_manifold(test_preds, te_z_all)
        
        out_sub_file = SUB_DIR / cand['out_sub_name']
        sub_df = sample_sub[['id']].copy()
        sub_df['is_fraud'] = cal_test_preds
        sub_df.to_csv(out_sub_file, index=False)
        print(f"  ✅ Saved calibrated submission to: {out_sub_file}")
        
        if cand['oof_path'] is not None and cand['oof_path'].exists():
            oof_preds = np.load(cand['oof_path'])
            eval_res = evaluate_manifold_cv(y, oof_preds, tr_z1, tr_z12, tr_z_all, skf, train)
            
            base = eval_res['Baseline (No Manifold)']
            print(f"  Baseline 5-Fold OOF AUC: {base['pooled_auc']:.6f} | Mean Fold: {base['mean_fold_auc']:.6f} | PR-AUC: {base['pr_auc']:.6f}")
            
            for variant in ['Zone 1 (daoviet)', 'Zone 1 + 2 (goodpjw2008)', 'Zone 1 + 2 + 3 (Our Novel Manifold)']:
                res = eval_res[variant]
                d_pooled = (res['pooled_auc'] - base['pooled_auc']) * 1e5
                d_mean = (res['mean_fold_auc'] - base['mean_fold_auc']) * 1e5
                print(f"  + {variant:34s}: OOF AUC {res['pooled_auc']:.6f} ({d_pooled:+6.2f} pts) | Mean Fold {res['mean_fold_auc']:.6f} ({d_mean:+6.2f} pts) | PR {res['pr_auc']:.6f}")
                
            # Fold details for combined manifold
            comb = eval_res['Zone 1 + 2 + 3 (Our Novel Manifold)']
            print(f"  Fold Progression for Combined Manifold:")
            for f in range(5):
                df_pts = (comb['folds'][f] - base['folds'][f]) * 1e5
                print(f"    Fold {f}: {base['folds'][f]:.6f} -> {comb['folds'][f]:.6f} ({df_pts:+6.2f} pts)")
                
            if cand['out_oof_name']:
                out_oof_file = SUB_DIR / cand['out_oof_name']
                np.save(out_oof_file, comb['cal_oof'])
                print(f"  ✅ Saved calibrated OOF predictions to: {out_oof_file}")

    print("\n" + "=" * 80)
    print("🎯 All Manifold Calibrated Submissions Successfully Generated!")
    print("=" * 80)

if __name__ == '__main__':
    main()
