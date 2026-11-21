"""
================================================================================
Sovereign Step R 9D Gram-Schmidt Apex Engine for PGS-S6E11
================================================================================
Base:
  - Step R (LB 0.95546, goodpjw2008 latest release, Nov 21 13:35 UTC)

Orthogonal Feature Streams (Orthogonalized via Gram-Schmidt against Step R & each other):
  1. v_shield: Sovereign Peak CV Shield (Nested OOF CV 0.95264, LB 0.95300)
  2. v_cb:     CatBoost GPU V2 (Dual-Seed 42 & 2026, CTGAN residues, 7.37% diversity)
  3. v_rmlp:   RealMLP-TD V3 (Dual-Seed 42 & 2026, CTGAN residues, 5.89% diversity)
  4. v_gpt2:   Dual-Seed GPT-2 (Seed 42 + Seed 2026 Causal LMs, 4.85% diversity)
  5. v_tabm:   TabM-PWL V3 (Multi-Seed k=32 PyTabKit Batch Ensemble)
  6. v_dig:    LightGBM Diglead L3 (Digit-Leading Spline Quantization)
  7. v_lgb:    Weni BPE LightGBM (Byte-Pair Encoded Digit Splines)

Machine-Precision Orthogonality (<v_i, b_R> < 10^-10).
Calibrated with the 5-Zone Deterministic Zero-Fraud Manifold (ZALL_5, 17,647 test accounts).
Target Public LB: 0.95548 - 0.95552+ (Global Top 15 Summit)
================================================================================
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from scipy.special import logit

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "dataset"
SUB_DIR = ROOT_DIR / "submission"
STEP_R_DIR = ROOT_DIR / "external_models" / "goodpjw2008_095546"
WENI_DIR = ROOT_DIR / "external_models" / "weni_bpe_pack"
TABM_DIR = ROOT_DIR / "code" / "tabm_gpu_v3" / "output"
REALMLP_DIR = ROOT_DIR / "code" / "realmlp_gpu_v3" / "output"
CATBOOST_DIR = ROOT_DIR / "code" / "catboost_gpu_v2" / "output"
GPT2_DIR = ROOT_DIR / "external_models" / "gpt2_seed2026"
GPT2_42_DIR = ROOT_DIR / "external_models" / "gpt2_seed42"
BANK_DIR = ROOT_DIR / "external_models" / "oof_bank"

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
    print("🚀 GENERATING SOVEREIGN STEP R 9D GRAM-SCHMIDT APEX (Base LB 0.95546)")
    print("=" * 80)

    test = pd.read_csv(DATA_DIR / "test.csv")
    sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
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

    # 2. Load Step R Base (0.95546)
    print("\n--- Loading Step R Base (LB 0.95546) ---")
    sub_r_df = pd.read_csv(STEP_R_DIR / "submission.csv")
    base_r = sub_r_df["is_fraud"].values
    b_r = L(base_r)

    # 3. Load Diverse Models & Sovereign Peak CV Shield
    print("--- Loading Diverse Models & Sovereign CV Shield ---")
    test_shield = pd.read_csv(SUB_DIR / "sovereign_cv95264_shield_submission.csv")["is_fraud"].values
    test_cb = np.load(CATBOOST_DIR / "test_catboost_gpu_v2.npy")
    test_rmlp = np.load(REALMLP_DIR / "test_realmlp_v3.npy")
    
    # Dual-Seed GPT-2
    t_gpt2_42 = U(np.load(GPT2_42_DIR / "test.npy"))
    t_gpt2_2026 = U(np.load(GPT2_DIR / "gpt2_s2026_test_5fold.npy"))
    test_gpt2 = 0.5 * t_gpt2_42 + 0.5 * t_gpt2_2026
    
    test_tabm = np.load(TABM_DIR / "test_tabm_pwl_v3.npy")
    test_dig = np.load(BANK_DIR / "test_lgb_diglead_l3.npy")
    test_lgb = np.load(WENI_DIR / "test_weni_bpe_lgb_l4.npy")

    u_sh = L(test_shield)
    u_cb = L(test_cb)
    u_rmlp = L(test_rmlp)
    u_gpt2 = L(test_gpt2)
    u_tabm = L(test_tabm)
    u_dig = L(test_dig)
    u_lgb = L(test_lgb)

    # 4. Gram-Schmidt Orthogonalization against Step R
    print("\n--- Computing 7-Vector Gram-Schmidt Orthogonalization on Step R ---")
    v1_sh = u_sh - proj(u_sh, b_r)
    v2_cb = u_cb - proj(u_cb, b_r) - proj(u_cb, v1_sh)
    v3_rmlp = u_rmlp - proj(u_rmlp, b_r) - proj(u_rmlp, v1_sh) - proj(u_rmlp, v2_cb)
    v4_gpt2 = u_gpt2 - proj(u_gpt2, b_r) - proj(u_gpt2, v1_sh) - proj(u_gpt2, v2_cb) - proj(u_gpt2, v3_rmlp)
    v5_tabm = u_tabm - proj(u_tabm, b_r) - proj(u_tabm, v1_sh) - proj(u_tabm, v2_cb) - proj(u_tabm, v3_rmlp) - proj(u_tabm, v4_gpt2)
    v6_dig = u_dig - proj(u_dig, b_r) - proj(u_dig, v1_sh) - proj(u_dig, v2_cb) - proj(u_dig, v3_rmlp) - proj(u_dig, v4_gpt2) - proj(u_dig, v5_tabm)
    v7_lgb = u_lgb - proj(u_lgb, b_r) - proj(u_lgb, v1_sh) - proj(u_lgb, v2_cb) - proj(u_lgb, v3_rmlp) - proj(u_lgb, v4_gpt2) - proj(u_lgb, v5_tabm) - proj(u_lgb, v6_dig)

    sqrt_n = np.sqrt(n_test)
    v1_hat = v1_sh / (np.linalg.norm(v1_sh) / sqrt_n)
    v2_hat = v2_cb / (np.linalg.norm(v2_cb) / sqrt_n)
    v3_hat = v3_rmlp / (np.linalg.norm(v3_rmlp) / sqrt_n)
    v4_hat = v4_gpt2 / (np.linalg.norm(v4_gpt2) / sqrt_n)
    v5_hat = v5_tabm / (np.linalg.norm(v5_tabm) / sqrt_n)
    v6_hat = v6_dig / (np.linalg.norm(v6_dig) / sqrt_n)
    v7_hat = v7_lgb / (np.linalg.norm(v7_lgb) / sqrt_n)

    # Verify Machine-Precision Orthogonality
    print(f"Orthogonality Check <v1_sh,   b_r>: {np.dot(v1_hat, b_r):.2e}")
    print(f"Orthogonality Check <v2_cb,   b_r>: {np.dot(v2_hat, b_r):.2e}")
    print(f"Orthogonality Check <v3_rmlp, b_r>: {np.dot(v3_hat, b_r):.2e}")
    print(f"Orthogonality Check <v4_gpt2, b_r>: {np.dot(v4_hat, b_r):.2e}")
    print(f"Orthogonality Check <v5_tabm, b_r>: {np.dot(v5_hat, b_r):.2e}")
    print(f"Orthogonality Check <v6_dig,  b_r>: {np.dot(v6_hat, b_r):.2e}")
    print(f"Orthogonality Check <v7_lgb,  b_r>: {np.dot(v7_hat, b_r):.2e}")

    # Micro-calibrated orthogonal coefficients
    c_sh = 0.006      # Sovereign Peak CV Shield (Nested CV: 0.95264)
    c_cb = 0.005      # CatBoost CTGAN Residue (highest non-tree diversity)
    c_rmlp = 0.004    # RealMLP-TD V3
    c_gpt2 = 0.003    # Dual-Seed GPT-2
    c_tabm = 0.003    # TabM-PWL V3
    c_dig = 0.002     # LGB Diglead
    c_lgb = 0.002     # Weni BPE LGB

    cand_r_logit = b_r + c_sh * v1_hat + c_cb * v2_hat + c_rmlp * v3_hat + c_gpt2 * v4_hat + c_tabm * v5_hat + c_dig * v6_hat + c_lgb * v7_hat
    cand_r_preds = apply_rank_preserving_manifold(cand_r_logit, zall_5_test)

    # 5. Sanity Checks & Rank Statistics
    corr_r = spearmanr(cand_r_preds, base_r).statistic
    diff_ranks = np.abs(rankdata(cand_r_preds) - rankdata(base_r))
    print(f"\n--- Alignment Diagnostics ---")
    print(f"  Spearman correlation with Step R: {corr_r:.6f}")
    print(f"  Mean absolute rank shift:         {diff_ranks.mean():.1f} positions")
    print(f"  Max absolute rank shift:          {diff_ranks.max():.0f} positions")
    print(f"  Transactions shifted > 1,000 ranks: {(diff_ranks > 1000).sum():,} ({(diff_ranks > 1000).mean():.2%})")

    # 6. Save Submission
    sub = sample_sub.copy()
    sub["is_fraud"] = cand_r_preds
    assert len(sub) == n_test
    assert sub["is_fraud"].between(0, 1).all()
    assert not sub["is_fraud"].isna().any()

    SUB_DIR.mkdir(parents=True, exist_ok=True)
    out_sub = SUB_DIR / "sovereign_step_r_9d_apex_submission.csv"
    sub.to_csv(out_sub, index=False)
    print(f"\n✅ Submission successfully saved: {out_sub}")
    print(f"Top 5 predictions:\n{sub.head()}")

if __name__ == "__main__":
    main()
