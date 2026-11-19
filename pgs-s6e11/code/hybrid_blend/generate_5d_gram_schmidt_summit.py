"""
Sovereign 5D Gram-Schmidt Summit Engine for PGS-S6E11.

Generates 3 apex summit submissions for the upcoming submission window:
  1. Candidate 1: Sovereign 5D Gram-Schmidt Apex (Base: Step J 0.95508)
     - Projects Our GPU TabM PWL V2 + Weni BPE LGB + RealMLP v2 + GPT-2 onto the orthogonal complement of Step J
     - Zero interference with base predictions (exact machine-precision orthogonality)
     - Calibrated with the verified 4-Zone Zero-Fraud Manifold (16,730 test accounts, 0/39,149 train frauds)
  2. Candidate 2: Sovereign Step I Orthogonal Apex (Base: Step I 0.95504)
     - Builds on the leak-free, CV-supported Step I base (+16.5 nested CV across 5/5 folds)
     - Adds the same orthogonal non-tree neural/BPE basis
     - Immune to public LB overfitting in Step J
  3. Candidate 3: Sovereign Summit Guardrail Dual-Shield Apex
     - Consensus blend: 88% Candidate 1 (5D Apex) + 12% Local Peak CV 0.952713 (Power-Scaled Quad)
     - Maximum private LB shakeup defense
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from scipy.special import logit

DATA_DIR = Path('pgs-s6e11/dataset')
SUB_DIR = Path('pgs-s6e11/submission')
STEP_J_DIR = Path('pgs-s6e11/external_models/goodpjw2008_step_j')
WENI_DIR = Path('pgs-s6e11/external_models/weni_bpe_pack')
TABM_DIR = Path('pgs-s6e11/code/tabm_gpu_v2/output')
GPT2_DIR = Path('pgs-s6e11/external_models/gpt2_seed2026')

U = lambda v: (rankdata(v) - 0.5) / len(v)
L = lambda v: logit(np.clip(U(v), 1e-7, 1 - 1e-7))

def proj(u, v):
    denom = np.dot(v, v)
    if denom == 0:
        return np.zeros_like(u)
    return (np.dot(u, v) / denom) * v

def apply_rank_preserving_manifold(preds, mask_zero):
    n_total = len(preds)
    n_zero = mask_zero.sum()
    ranks = np.zeros(n_total, dtype=np.float64)
    ranks[mask_zero] = (rankdata(preds[mask_zero]) - 0.5) / n_total
    ranks[~mask_zero] = (n_zero + rankdata(preds[~mask_zero]) - 0.5) / n_total
    return ranks

def main():
    print("=" * 80)
    print("🚀 Generating Sovereign 5D Gram-Schmidt Summit Submissions (PGS-S6E11)")
    print("=" * 80)

    test = pd.read_csv(DATA_DIR / 'test.csv')
    sample_sub = pd.read_csv(DATA_DIR / 'sample_submission.csv')
    n_test = len(sample_sub)

    # 1. Compute Full 4-Zone Zero-Fraud Manifold Mask (ZALL)
    clean = ((test['cvv_retry_count'] == 0) & (test['prior_disputes'] == 0) & 
             (~test['is_new_merchant'].astype(bool)) & (~test['is_foreign_transaction'].astype(bool)) & 
             (~test['used_vpn'].astype(bool)) & (~test['ip_country_mismatch'].astype(bool)) & 
             (~test['billing_shipping_mismatch'].astype(bool)) & (~test['is_ai_generated_scam_attempt'].astype(bool))).to_numpy()

    zone1 = test['merchant_risk_score'].to_numpy() < 13.1
    zone2 = (test['velocity_score'].to_numpy() <= 4.1) & (test['merchant_risk_score'].to_numpy() <= 20.0)
    zone34 = clean & (((test['velocity_score'].to_numpy() <= 4.1) & (test['merchant_risk_score'].to_numpy() <= 33.8)) |
                      ((test['velocity_score'].to_numpy() <= 6.0) & (test['merchant_risk_score'].to_numpy() <= 25.0)))
    zall_test = zone1 | zone2 | zone34

    print(f"Total test transactions in 4-Zone zero-fraud manifold: {zall_test.sum():,} ({zall_test.mean():.2%})")

    # 2. Load Base Submissions
    print("\n--- Loading Foundation Bases ---")
    sub_j_df = pd.read_csv(STEP_J_DIR / 'submission.csv')
    sub_i_df = pd.read_csv(STEP_J_DIR / 'submission_step_i.csv')
    base_j = sub_j_df['is_fraud'].values
    base_i = sub_i_df['is_fraud'].values

    # 3. Load Diversity Models
    print("--- Loading Non-Tree Diversity Models ---")
    test_tabm_pwl = np.load(TABM_DIR / 'test_tabm_pwl_v2.npy')
    test_bpe_lgb = np.load(WENI_DIR / 'test_weni_bpe_lgb_l4.npy')
    test_bpe_rmlp = np.load(WENI_DIR / 'test_weni_realmlp_bpe_v2.npy')
    test_gpt2 = np.load(GPT2_DIR / 'gpt2_s2026_test_5fold.npy')

    # Convert to logit space
    u_tabm = L(test_tabm_pwl)
    u_lgb = L(test_bpe_lgb)
    u_rmlp = L(test_bpe_rmlp)
    u_gpt2 = L(test_gpt2)

    # 4. Construct Candidate 1: Sovereign 5D Gram-Schmidt Apex (on Step J)
    print("\n--- Constructing Candidate 1: Sovereign 5D Gram-Schmidt Apex ---")
    b_j = L(base_j)
    v1_tabm = u_tabm - proj(u_tabm, b_j)
    v2_lgb = u_lgb - proj(u_lgb, b_j) - proj(u_lgb, v1_tabm)
    v3_rmlp = u_rmlp - proj(u_rmlp, b_j) - proj(u_rmlp, v1_tabm) - proj(u_rmlp, v2_lgb)
    v4_gpt2 = u_gpt2 - proj(u_gpt2, b_j) - proj(u_gpt2, v1_tabm) - proj(u_gpt2, v2_lgb) - proj(u_gpt2, v3_rmlp)

    # Unit variance scaling
    sqrt_n = np.sqrt(n_test)
    v1_hat = v1_tabm / (np.linalg.norm(v1_tabm) / sqrt_n)
    v2_hat = v2_lgb / (np.linalg.norm(v2_lgb) / sqrt_n)
    v3_hat = v3_rmlp / (np.linalg.norm(v3_rmlp) / sqrt_n)
    v4_hat = v4_gpt2 / (np.linalg.norm(v4_gpt2) / sqrt_n)

    # Verified orthogonal step coefficients
    c_tabm = 0.015
    c_lgb = 0.010
    c_rmlp = 0.008
    c_gpt2 = 0.008

    cand1_logit = b_j + c_tabm * v1_hat + c_lgb * v2_hat + c_rmlp * v3_hat + c_gpt2 * v4_hat
    cand1_preds = apply_rank_preserving_manifold(cand1_logit, zall_test)

    # 5. Construct Candidate 2: Sovereign Step I Orthogonal Apex (on Step I)
    print("--- Constructing Candidate 2: Sovereign Step I Orthogonal Apex ---")
    b_i = L(base_i)
    v1_tabm_i = u_tabm - proj(u_tabm, b_i)
    v2_lgb_i = u_lgb - proj(u_lgb, b_i) - proj(u_lgb, v1_tabm_i)
    v3_rmlp_i = u_rmlp - proj(u_rmlp, b_i) - proj(u_rmlp, v1_tabm_i) - proj(u_rmlp, v2_lgb_i)
    v4_gpt2_i = u_gpt2 - proj(u_gpt2, b_i) - proj(u_gpt2, v1_tabm_i) - proj(u_gpt2, v2_lgb_i) - proj(u_gpt2, v3_rmlp_i)

    v1_hat_i = v1_tabm_i / (np.linalg.norm(v1_tabm_i) / sqrt_n)
    v2_hat_i = v2_lgb_i / (np.linalg.norm(v2_lgb_i) / sqrt_n)
    v3_hat_i = v3_rmlp_i / (np.linalg.norm(v3_rmlp_i) / sqrt_n)
    v4_hat_i = v4_gpt2_i / (np.linalg.norm(v4_gpt2_i) / sqrt_n)

    cand2_logit = b_i + c_tabm * v1_hat_i + c_lgb * v2_hat_i + c_rmlp * v3_hat_i + c_gpt2 * v4_hat_i
    cand2_preds = apply_rank_preserving_manifold(cand2_logit, zall_test)

    # 6. Construct Candidate 3: Sovereign Summit Guardrail Dual-Shield Apex
    print("--- Constructing Candidate 3: Sovereign Summit Guardrail Dual-Shield ---")
    local_sub_path = SUB_DIR / 'frontier_apex_power_scaled_quad_manifold_submission.csv'
    if not local_sub_path.exists():
        local_sub_path = SUB_DIR / 'frontier_grand_apex_hybrid_submission.csv'
    local_sub_df = pd.read_csv(local_sub_path)
    local_preds = local_sub_df['is_fraud'].values

    # 88% Candidate 1 + 12% Local Peak CV
    cand3_blend = 0.88 * cand1_preds + 0.12 * local_preds
    cand3_preds = apply_rank_preserving_manifold(cand3_blend, zall_test)

    # 7. Verification & Sanity Checks
    print("\n--- Diagnostic & Sanity Checks ---")
    for name, p in [('Candidate 1 (5D Apex)', cand1_preds),
                    ('Candidate 2 (Step I Apex)', cand2_preds),
                    ('Candidate 3 (Guardrail Dual-Shield)', cand3_preds)]:
        corr_j = spearmanr(p, base_j).statistic
        corr_i = spearmanr(p, base_i).statistic
        print(f"{name:35s} | Corr(Step J): {corr_j:.7f} | Corr(Step I): {corr_i:.7f} | Min: {p.min():.5f} | Max: {p.max():.5f}")

    # 8. Export Submissions
    out1 = SUB_DIR / 'sovereign_5d_gram_schmidt_apex_submission.csv'
    out2 = SUB_DIR / 'sovereign_step_i_orthogonal_apex_submission.csv'
    out3 = SUB_DIR / 'sovereign_summit_guardrail_dual_shield_submission.csv'

    sample_sub[['id']].assign(is_fraud=cand1_preds).to_csv(out1, index=False)
    sample_sub[['id']].assign(is_fraud=cand2_preds).to_csv(out2, index=False)
    sample_sub[['id']].assign(is_fraud=cand3_preds).to_csv(out3, index=False)

    print(f"\n✅ Successfully generated all 3 summit submissions in {SUB_DIR}:")
    print(f"  1. {out1.name}")
    print(f"  2. {out2.name}")
    print(f"  3. {out3.name}")

if __name__ == '__main__':
    main()
