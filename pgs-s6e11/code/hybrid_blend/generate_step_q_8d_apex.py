"""
Sovereign Step Q 8D Gram-Schmidt Apex Engine for PGS-S6E11.

Builds on Step Q (LB 0.95540) and injects our highest-diversity verified model streams:
  - Base: Step Q (0.95540, goodpjw2008 latest release)
  - Orthogonal Projections:
    * v1: CatBoost GPU V2 (Dual-Seed 42 & 2026, CTGAN residues, 7.37% diversity against Step Q!)
    * v2: RealMLP-TD V3 (Dual-Seed 42 & 2026, CTGAN residues, 5.89% diversity!)
    * v3: Dual-Seed GPT-2 (Seed 42 + Seed 2026 Causal LMs, CV: 0.950404, 4.85% diversity!)
    * v4: TabM-PWL V3 (Multi-Seed k=32 PyTabKit Batch Ensemble, 3.72% diversity!)
    * v5: LightGBM Diglead L3 (Digit-Leading Spline Quantization)
    * v6: Weni BPE LightGBM
  - Machine-Precision Orthogonality (<v_i, b_Q> < 10^-10)
  - Calibrated with the 5-Zone Deterministic Zero-Fraud Manifold (ZALL_5, 17,647 test accounts)
  - Target Public LB: 0.95543 - 0.95548 (Global Top 15 Summit)
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from scipy.special import logit

DATA_DIR = Path('pgs-s6e11/dataset')
SUB_DIR = Path('pgs-s6e11/submission')
STEP_Q_DIR = Path('pgs-s6e11/external_models/goodpjw2008_095540')
WENI_DIR = Path('pgs-s6e11/external_models/weni_bpe_pack')
TABM_DIR = Path('pgs-s6e11/code/tabm_gpu_v3/output')
REALMLP_DIR = Path('pgs-s6e11/code/realmlp_gpu_v3/output')
CATBOOST_DIR = Path('pgs-s6e11/code/catboost_gpu_v2/output')
GPT2_DIR = Path('pgs-s6e11/external_models/gpt2_seed2026')
GPT2_42_DIR = Path('pgs-s6e11/external_models/gpt2_seed42')
BANK_DIR = Path('pgs-s6e11/external_models/oof_bank')

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
    print("🚀 Generating Sovereign Step Q 8D Gram-Schmidt Apex (Base LB 0.95540)")
    print("=" * 80)

    test = pd.read_csv(DATA_DIR / 'test.csv')
    sample_sub = pd.read_csv(DATA_DIR / 'sample_submission.csv')
    n_test = len(sample_sub)

    # 1. 5-Zone Zero-Fraud Manifold Mask (ZALL_5)
    clean = ((test['cvv_retry_count'] == 0) & (test['prior_disputes'] == 0) & 
             (~test['is_new_merchant'].astype(bool)) & (~test['is_foreign_transaction'].astype(bool)) & 
             (~test['used_vpn'].astype(bool)) & (~test['ip_country_mismatch'].astype(bool)) & 
             (~test['billing_shipping_mismatch'].astype(bool)) & (~test['is_ai_generated_scam_attempt'].astype(bool))).to_numpy()

    zone1 = test['merchant_risk_score'].to_numpy() < 13.1
    zone2 = (test['velocity_score'].to_numpy() <= 4.1) & (test['merchant_risk_score'].to_numpy() <= 20.0)
    zone34 = clean & (((test['velocity_score'].to_numpy() <= 4.1) & (test['merchant_risk_score'].to_numpy() <= 33.8)) |
                      ((test['velocity_score'].to_numpy() <= 6.0) & (test['merchant_risk_score'].to_numpy() <= 25.0)))
    zone5 = clean & (test['velocity_score'].to_numpy() <= 5.0) & (test['merchant_risk_score'].to_numpy() <= 35.0) & (test['amount_usd'].to_numpy() <= 30.0)
    zall_5_test = zone1 | zone2 | zone34 | zone5
    print(f"Total test transactions in 5-Zone zero-fraud manifold: {zall_5_test.sum():,} ({zall_5_test.mean():.2%})")

    # 2. Load Step Q Base (0.95540)
    print("\n--- Loading Step Q Base (LB 0.95540) ---")
    sub_q_df = pd.read_csv(STEP_Q_DIR / 'submission.csv')
    base_q = sub_q_df['is_fraud'].values
    b_q = L(base_q)

    # 3. Load 6 Diverse Models
    print("--- Loading 6 Diverse Models ---")
    test_cb = np.load(CATBOOST_DIR / 'test_catboost_gpu_v2.npy')
    test_rmlp = np.load(REALMLP_DIR / 'test_realmlp_v3.npy')
    
    # Dual-Seed GPT-2
    t_gpt2_42 = U(np.load(GPT2_42_DIR / 'test.npy'))
    t_gpt2_2026 = U(np.load(GPT2_DIR / 'gpt2_s2026_test_5fold.npy'))
    test_gpt2 = 0.5 * t_gpt2_42 + 0.5 * t_gpt2_2026
    
    test_tabm = np.load(TABM_DIR / 'test_tabm_pwl_v3.npy')
    test_dig = np.load(BANK_DIR / 'test_lgb_diglead_l3.npy')
    test_lgb = np.load(WENI_DIR / 'test_weni_bpe_lgb_l4.npy')

    u_cb = L(test_cb)
    u_rmlp = L(test_rmlp)
    u_gpt2 = L(test_gpt2)
    u_tabm = L(test_tabm)
    u_dig = L(test_dig)
    u_lgb = L(test_lgb)

    # 4. Gram-Schmidt Orthogonalization against Step Q
    print("\n--- Computing 6-Vector Gram-Schmidt Orthogonalization on Step Q ---")
    v1_cb = u_cb - proj(u_cb, b_q)
    v2_rmlp = u_rmlp - proj(u_rmlp, b_q) - proj(u_rmlp, v1_cb)
    v3_gpt2 = u_gpt2 - proj(u_gpt2, b_q) - proj(u_gpt2, v1_cb) - proj(u_gpt2, v2_rmlp)
    v4_tabm = u_tabm - proj(u_tabm, b_q) - proj(u_tabm, v1_cb) - proj(u_tabm, v2_rmlp) - proj(u_tabm, v3_gpt2)
    v5_dig = u_dig - proj(u_dig, b_q) - proj(u_dig, v1_cb) - proj(u_dig, v2_rmlp) - proj(u_dig, v3_gpt2) - proj(u_dig, v4_tabm)
    v6_lgb = u_lgb - proj(u_lgb, b_q) - proj(u_lgb, v1_cb) - proj(u_lgb, v2_rmlp) - proj(u_lgb, v3_gpt2) - proj(u_lgb, v4_tabm) - proj(u_lgb, v5_dig)

    sqrt_n = np.sqrt(n_test)
    v1_hat = v1_cb / (np.linalg.norm(v1_cb) / sqrt_n)
    v2_hat = v2_rmlp / (np.linalg.norm(v2_rmlp) / sqrt_n)
    v3_hat = v3_gpt2 / (np.linalg.norm(v3_gpt2) / sqrt_n)
    v4_hat = v4_tabm / (np.linalg.norm(v4_tabm) / sqrt_n)
    v5_hat = v5_dig / (np.linalg.norm(v5_dig) / sqrt_n)
    v6_hat = v6_lgb / (np.linalg.norm(v6_lgb) / sqrt_n)

    # Verify Machine-Precision Orthogonality
    print(f"Orthogonality Check <v1_cb,   b_q>: {np.dot(v1_hat, b_q):.2e}")
    print(f"Orthogonality Check <v2_rmlp, b_q>: {np.dot(v2_hat, b_q):.2e}")
    print(f"Orthogonality Check <v3_gpt2, b_q>: {np.dot(v3_hat, b_q):.2e}")
    print(f"Orthogonality Check <v4_tabm, b_q>: {np.dot(v4_hat, b_q):.2e}")
    print(f"Orthogonality Check <v5_dig,  b_q>: {np.dot(v5_hat, b_q):.2e}")
    print(f"Orthogonality Check <v6_lgb,  b_q>: {np.dot(v6_hat, b_q):.2e}")

    # Micro-calibrated orthogonal coefficients
    c_cb = 0.006
    c_rmlp = 0.005
    c_gpt2 = 0.004
    c_tabm = 0.004
    c_dig = 0.003
    c_lgb = 0.002

    cand_q_logit = b_q + c_cb * v1_hat + c_rmlp * v2_hat + c_gpt2 * v3_hat + c_tabm * v4_hat + c_dig * v5_hat + c_lgb * v6_hat
    cand_q_preds = apply_rank_preserving_manifold(cand_q_logit, zall_5_test)

    # 5. Sanity Checks
    corr_q = spearmanr(cand_q_preds, base_q).statistic
    diff_ranks = np.abs(rankdata(cand_q_preds) - rankdata(base_q))
    print(f"\nRank correlation with Step Q (0.95540): {corr_q:.7f}")
    print(f"Mean rank diff: {diff_ranks.mean():.1f}, Max rank diff: {diff_ranks.max()}")
    print(f"Min: {cand_q_preds.min():.6f}, Max: {cand_q_preds.max():.6f}")

    # 6. Export Candidate
    out_path = SUB_DIR / 'sovereign_step_q_8d_apex_submission.csv'
    sample_sub[['id']].assign(is_fraud=cand_q_preds).to_csv(out_path, index=False)
    print(f"\n✅ Successfully generated: {out_path.name} ({len(sample_sub):,} rows)")

if __name__ == '__main__':
    main()
