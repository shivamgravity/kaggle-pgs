"""
Top-5 Summit Candidates Pipeline for PGS-S6E11.

Generates 3 high-conviction summit submissions to reach Top 5-10:
  1. Candidate 1: Decoupled Step I + 3-Zone Zero-Fraud Manifold
     (Solves the Coupler Flaw in goodpjw2008 base; preserves Step F basis + extends zero-fraud manifold)
  2. Candidate 2: Sovereign 4D Gram-Schmidt Neural Orthogonal Apex
     (Decoupled Step I + Orthogonal Yekenot TabM + Orthogonal Our GPU TabM + 3-Zone Manifold)
  3. Candidate 3: Sovereign Consensus Guardrail Apex
     (90% Candidate 2 4D Apex + 10% Local CV Peak 0.952713 Power-Scaled Quad + 3-Zone Manifold)
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from scipy.special import logit

DATA_DIR = Path('pgs-s6e11/dataset')
SUB_DIR = Path('pgs-s6e11/submission')
GP_DIR = Path('pgs-s6e11/external_models/goodpjw_095504')
YK_DIR = Path('pgs-s6e11/external_models/yekenot_tabm_pytabkit')

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
    print("🚀 Generating Top-5 Summit Candidates (PGS-S6E11)")
    print("=" * 80)

    test = pd.read_csv(DATA_DIR / 'test.csv')
    sample_sub = pd.read_csv(DATA_DIR / 'sample_submission.csv')
    n_test = len(sample_sub)

    # 1. Compute 3-Zone Zero-Fraud Manifold Mask
    z1 = test['merchant_risk_score'] < 13.1
    z2 = (test['velocity_score'] <= 4.1) & (test['merchant_risk_score'] <= 20.0)
    z3 = (test['account_balance_usd'] <= 593.66) & (test['velocity_score'] <= 0.0)
    z_all = z1 | z2 | z3

    print(f"Total test transactions in 3-zone zero-fraud manifold: {z_all.sum():,} ({z_all.mean():.2%})")

    # 2. Load Core References
    y_e = pd.read_csv(GP_DIR / 'submission_step_e.csv')['is_fraud'].values
    y_f = pd.read_csv(GP_DIR / 'submission_step_f.csv')['is_fraud'].values
    y_g = pd.read_csv(GP_DIR / 'submission_step_g.csv')['is_fraud'].values
    y_095504 = pd.read_csv(GP_DIR / 'submission.csv')['is_fraud'].values
    y_yk_tabm = pd.read_csv(YK_DIR / 'submission.csv')['is_fraud'].values
    y_our_tabm = pd.read_csv(SUB_DIR / 'frontier_apex_tabm_manifold_submission.csv')['is_fraud'].values
    y_peak_cv = pd.read_csv(SUB_DIR / 'frontier_apex_power_scaled_quad_manifold_submission.csv')['is_fraud'].values

    # Convert to logit space
    Le = L(y_e)
    Lf = L(y_f)
    Lg = L(y_g)

    # Basis 1: Foundation Vector v_F
    v_F = Lf - Le
    v22 = Le + 6.0 * v_F

    # Basis 2: Orthogonalized Step G Innovation v_G_perp
    v_G = Lg - L(v22)
    proj_FG = np.dot(v_G, v_F) / np.dot(v_F, v_F)
    v_G_perp = v_G - proj_FG * v_F
    print(f"Gram-Schmidt Projection <v_G, v_F> / ||v_F||^2: {proj_FG:.4f} (Coupler Flaw eliminated)")

    # Construct Decoupled Step I Base: c1 = 5.8, c2 = 2.4
    z_step_i = Le + 5.8 * v_F + 2.4 * v_G_perp

    # Basis 3: Orthogonalized Yekenot TabM v_yk_perp
    delta_yk = L(y_yk_tabm) - z_step_i
    proj_yk_F = np.dot(delta_yk, v_F) / np.dot(v_F, v_F)
    proj_yk_G = np.dot(delta_yk, v_G_perp) / np.dot(v_G_perp, v_G_perp)
    v_yk_perp = delta_yk - proj_yk_F * v_F - proj_yk_G * v_G_perp

    # Basis 4: Orthogonalized Our TabM v_our_perp
    delta_our = L(y_our_tabm) - z_step_i
    proj_our_F = np.dot(delta_our, v_F) / np.dot(v_F, v_F)
    proj_our_G = np.dot(delta_our, v_G_perp) / np.dot(v_G_perp, v_G_perp)
    proj_our_yk = np.dot(delta_our, v_yk_perp) / np.dot(v_yk_perp, v_yk_perp)
    v_our_perp = delta_our - proj_our_F * v_F - proj_our_G * v_G_perp - proj_our_yk * v_yk_perp

    # Verify 4D Orthogonality
    print("\nVerifying 4D Gram-Schmidt Orthogonality Matrix:")
    basis_vectors = [v_F, v_G_perp, v_yk_perp, v_our_perp]
    names = ['v_F', 'v_G_perp', 'v_yk_perp', 'v_our_perp']
    for i in range(len(basis_vectors)):
        for j in range(i + 1, len(basis_vectors)):
            corr = np.corrcoef(basis_vectors[i], basis_vectors[j])[0, 1]
            print(f"  rho({names[i]}, {names[j]}) = {corr:+.8f}")

    # =========================================================================
    # CANDIDATE 1: Decoupled Step I + 3-Zone Zero-Fraud Manifold
    # =========================================================================
    print("\n--- Generating Candidate 1: Decoupled Step I + 3-Zone Zero-Fraud Manifold ---")
    c1_raw = U(z_step_i)
    c1_preds = apply_rank_preserving_manifold(c1_raw, z_all)
    c1_path = SUB_DIR / 'decoupled_step_i_3zone_manifold_submission.csv'
    sub1 = sample_sub[['id']].copy()
    sub1['is_fraud'] = c1_preds
    sub1.to_csv(c1_path, index=False)
    print(f"  ✅ Saved Candidate 1 to: {c1_path}")
    print(f"     Spearman r with 0.95504 Base: {spearmanr(c1_preds, y_095504).correlation:.6f}")

    # =========================================================================
    # CANDIDATE 2: Sovereign 4D Gram-Schmidt Neural Orthogonal Apex
    # =========================================================================
    print("\n--- Generating Candidate 2: Sovereign 4D Gram-Schmidt Neural Apex ---")
    # c1 = 5.8, c2 = 2.4, w_yk = 0.035, w_our = 0.025
    z_4d = z_step_i + 0.035 * v_yk_perp + 0.025 * v_our_perp
    c2_raw = U(z_4d)
    c2_preds = apply_rank_preserving_manifold(c2_raw, z_all)
    c2_path = SUB_DIR / 'sovereign_4d_gram_schmidt_apex_submission.csv'
    sub2 = sample_sub[['id']].copy()
    sub2['is_fraud'] = c2_preds
    sub2.to_csv(c2_path, index=False)
    print(f"  ✅ Saved Candidate 2 to: {c2_path}")
    print(f"     Spearman r with 0.95504 Base: {spearmanr(c2_preds, y_095504).correlation:.6f}")
    print(f"     Spearman r with Candidate 1: {spearmanr(c2_preds, c1_preds).correlation:.6f}")

    # =========================================================================
    # CANDIDATE 3: Sovereign Consensus Guardrail Apex (90% C2 + 10% Local Peak CV)
    # =========================================================================
    print("\n--- Generating Candidate 3: Sovereign Consensus Guardrail Apex ---")
    c3_raw = U(0.90 * L(c2_preds) + 0.10 * L(y_peak_cv))
    c3_preds = apply_rank_preserving_manifold(c3_raw, z_all)
    c3_path = SUB_DIR / 'sovereign_consensus_guardrail_apex_submission.csv'
    sub3 = sample_sub[['id']].copy()
    sub3['is_fraud'] = c3_preds
    sub3.to_csv(c3_path, index=False)
    print(f"  ✅ Saved Candidate 3 to: {c3_path}")
    print(f"     Spearman r with 0.95504 Base: {spearmanr(c3_preds, y_095504).correlation:.6f}")
    print(f"     Spearman r with Candidate 2: {spearmanr(c3_preds, c2_preds).correlation:.6f}")
    print(f"     Rank delta count vs Candidate 2: {(rankdata(c3_preds) != rankdata(c2_preds)).sum():,}")

    # =========================================================================
    # INTEGRITY CHECKS
    # =========================================================================
    print("\n🔍 Running Strict Submission Integrity Checks:")
    for path, p in [(c1_path, c1_preds), (c2_path, c2_preds), (c3_path, c3_preds)]:
        assert len(p) == n_test, f"Length mismatch in {path.name}"
        assert not np.isnan(p).any(), f"NaN found in {path.name}"
        assert (p > 0).all() and (p < 1).all(), f"Boundary violation in {path.name}"
        # Check zero-fraud manifold condition
        zero_max = p[z_all].max()
        nonzero_min = p[~z_all].min()
        assert zero_max < nonzero_min, f"Manifold ordering violated in {path.name}: {zero_max} >= {nonzero_min}"
        print(f"  ✅ {path.name}: verified ({len(p):,} rows, zero-max={zero_max:.8f} < nonzero-min={nonzero_min:.8f})")

    print("\n🎯 All 3 Top-5 Summit Candidates Successfully Created and Verified!")

if __name__ == '__main__':
    main()
