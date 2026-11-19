"""
Podium Top-10 Meta-Apex Pipeline for PGS-S6E11.

Generates high-conviction submissions to break directly into the Top 10 (0.95505+):
  1. Candidate 1: Podium Apex Zone-3 Calibrated Shield (0.95504 base + Zone 3 Zero-Fraud Manifold)
  2. Candidate 2: Sovereign Gram-Schmidt Orthogonal Meta-Apex (0.95504 base + Orthogonal TabM/ResNet Delta + Manifold)
  3. Candidate 3: Tri-Ladder Consensus Apex (80% Podium Apex + 20% Our Frontier TabM Manifold)
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.special import logit

DATA_DIR = Path('pgs-s6e11/dataset')
SUB_DIR = Path('pgs-s6e11/submission')
PBF_DIR = Path('pgs-s6e11/external_models/public_board_files')
GP_DIR = Path('pgs-s6e11/external_models/goodpjw_095504')

U = lambda v: (rankdata(v) - 0.5) / len(v)
L = lambda v: logit(np.clip(U(v), 1e-7, 1 - 1e-7))

def apply_rank_preserving_manifold(preds, mask_zero):
    n_total = len(preds)
    n_zero = mask_zero.sum()
    ranks = np.zeros(n_total, dtype=np.float64)
    ranks[mask_zero] = (rankdata(preds[mask_zero]) - 0.5) / n_total
    ranks[~mask_zero] = (n_zero + rankdata(preds[~mask_zero]) - 0.5) / n_total
    return ranks

def main():
    print("=" * 80)
    print("🚀 Running Podium Top-10 Meta-Apex Pipeline")
    print("=" * 80)

    test = pd.read_csv(DATA_DIR / 'test.csv')
    sample_sub = pd.read_csv(DATA_DIR / 'sample_submission.csv')

    # 3-Zone Zero-Fraud Manifold
    z1 = test['merchant_risk_score'] < 13.1
    z2 = (test['velocity_score'] <= 4.1) & (test['merchant_risk_score'] <= 20.0)
    z3 = (test['account_balance_usd'] <= 593.66) & (test['velocity_score'] <= 0.0)
    z_all = z1 | z2 | z3

    print(f"Total test transactions in 3-zone zero-fraud manifold: {z_all.sum():,} ({z_all.mean():.2%})")

    # Load 0.95504 base
    f_095504 = pd.read_csv(GP_DIR / 'submission.csv')['is_fraud'].values
    our_tabm = pd.read_csv(SUB_DIR / 'frontier_apex_tabm_manifold_submission.csv')['is_fraud'].values

    # Candidate 1: Podium Apex Zone-3 Calibrated Shield
    print("\n--- Generating Candidate 1: Podium Apex Zone-3 Calibrated ---")
    sub1_preds = apply_rank_preserving_manifold(f_095504, z_all)
    sub1_path = SUB_DIR / 'podium_apex_zone3_calibrated_submission.csv'
    sub_df1 = sample_sub[['id']].copy()
    sub_df1['is_fraud'] = sub1_preds
    sub_df1.to_csv(sub1_path, index=False)
    print(f"  ✅ Saved Candidate 1 to: {sub1_path}")

    # Candidate 2: Sovereign Gram-Schmidt Orthogonal Meta-Apex
    print("\n--- Generating Candidate 2: Sovereign Gram-Schmidt Orthogonal Meta-Apex ---")
    prior = L(f_095504)
    delta = L(our_tabm) - prior
    delta_ortho = delta - float(delta @ prior / (prior @ prior)) * prior
    
    # Weight w = 0.08
    w = 0.08
    sub2_raw = U(prior + w * delta_ortho)
    sub2_preds = apply_rank_preserving_manifold(sub2_raw, z_all)
    sub2_path = SUB_DIR / 'sovereign_gram_schmidt_meta_apex_submission.csv'
    sub_df2 = sample_sub[['id']].copy()
    sub_df2['is_fraud'] = sub2_preds
    sub_df2.to_csv(sub2_path, index=False)
    print(f"  ✅ Saved Candidate 2 to: {sub2_path}")

    # Candidate 3: Tri-Ladder Consensus Apex (80% Podium + 20% Our TabM Manifold)
    print("\n--- Generating Candidate 3: Tri-Ladder Consensus Apex ---")
    sub3_raw = U(0.80 * L(sub1_preds) + 0.20 * L(our_tabm))
    sub3_preds = apply_rank_preserving_manifold(sub3_raw, z_all)
    sub3_path = SUB_DIR / 'tri_ladder_top10_consensus_submission.csv'
    sub_df3 = sample_sub[['id']].copy()
    sub_df3['is_fraud'] = sub3_preds
    sub_df3.to_csv(sub3_path, index=False)
    print(f"  ✅ Saved Candidate 3 to: {sub3_path}")

    # Verify correlations
    print("\n📊 Rank Correlations Between Candidates:")
    r12 = np.corrcoef(rankdata(sub1_preds), rankdata(sub2_preds))[0, 1]
    r13 = np.corrcoef(rankdata(sub1_preds), rankdata(sub3_preds))[0, 1]
    r23 = np.corrcoef(rankdata(sub2_preds), rankdata(sub3_preds))[0, 1]
    print(f"  Candidate 1 vs Candidate 2: {r12:.6f}")
    print(f"  Candidate 1 vs Candidate 3: {r13:.6f}")
    print(f"  Candidate 2 vs Candidate 3: {r23:.6f}")
    print(f"  Candidate 1 vs 0.95504 Base: {np.corrcoef(rankdata(sub1_preds), rankdata(f_095504))[0, 1]:.6f}")

    print("\n🎯 All Podium Top-10 Candidates Ready for Submission!")

if __name__ == '__main__':
    main()
