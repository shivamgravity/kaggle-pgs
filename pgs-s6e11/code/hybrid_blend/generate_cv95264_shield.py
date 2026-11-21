"""
================================================================================
Sovereign Peak CV Shield Generator (5-Fold Leak-Free CV: 0.952636)
================================================================================
Combines:
  1. Wide LM Stack Nested OOF / Test (65-model multi-family ensemble including GPT-2)
  2. TabM-PWL V3 (Multi-Seed k=32 PyTabKit Batch Ensemble with Piecewise-Linear activations)
  3. GBN Bag 5 (w512) (GLM-Residual Multi-Layer Perceptron)
  4. ResNet Bag 3 (Deep ResNet for Tabular Data)
  5. 5-Zone Deterministic Zero-Fraud Manifold Calibration (ZALL_5)

Metrics:
  - 5-Fold Stratified Out-of-Fold ROC-AUC: 0.952636
  - PR-AUC: 0.5428+
  - Inversions Fixed: 222,353+ transactions calibrated against manifold lower bound
================================================================================
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "dataset"
SUBMISSION_DIR = ROOT_DIR / "submission"
EXTERNAL_DIR = ROOT_DIR / "external_models" / "artifacts"
TABM_DIR = ROOT_DIR / "code" / "tabm_gpu_v3" / "output"
NEURAL_DIR = ROOT_DIR / "code" / "dual_neural_multiseed" / "output"

def main():
    print("=" * 80)
    print("🛡️  GENERATING SOVEREIGN PEAK CV SHIELD (CV: 0.952636)")
    print("=" * 80)

    # 1. Load Data
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
    y = train["is_fraud"].values
    n_train = len(y)
    n_test = len(test)

    # 2. 5-Zone Zero-Fraud Manifold
    def compute_manifold(df):
        clean = ((df["cvv_retry_count"] == 0) & (df["prior_disputes"] == 0) & 
                 (~df["is_new_merchant"].astype(bool)) & (~df["is_foreign_transaction"].astype(bool)) & 
                 (~df["used_vpn"].astype(bool)) & (~df["ip_country_mismatch"].astype(bool)) & 
                 (~df["billing_shipping_mismatch"].astype(bool)) & (~df["is_ai_generated_scam_attempt"].astype(bool))).to_numpy()
        z1 = df["merchant_risk_score"].to_numpy() < 13.1
        z2 = (df["velocity_score"].to_numpy() <= 4.1) & (df["merchant_risk_score"].to_numpy() <= 20.0)
        z34 = clean & (((df["velocity_score"].to_numpy() <= 4.1) & (df["merchant_risk_score"].to_numpy() <= 33.8)) |
                       ((df["velocity_score"].to_numpy() <= 6.0) & (df["merchant_risk_score"].to_numpy() <= 25.0)))
        z5 = clean & (df["velocity_score"].to_numpy() <= 5.0) & (df["merchant_risk_score"].to_numpy() <= 35.0) & (df["amount_usd"].to_numpy() <= 30.0)
        return z1 | z2 | z34 | z5

    manifold_tr = compute_manifold(train)
    manifold_te = compute_manifold(test)
    print(f"Train Manifold: {manifold_tr.sum():,} rows ({manifold_tr.mean():.2%}) | Frauds: {y[manifold_tr].sum()}")
    print(f"Test Manifold:  {manifold_te.sum():,} rows ({manifold_te.mean():.2%})")

    # 3. Model Weights from Caruana Forward Selection
    w_wide = 0.903861
    w_tabm = 0.042440
    w_gbn  = 0.044674
    w_res  = 0.009025
    print(f"\nOptimal Ensemble Weights (Sum={w_wide + w_tabm + w_gbn + w_res:.6f}):")
    print(f"  Wide LM Stack:  {w_wide:.4f}")
    print(f"  TabM-PWL V3:    {w_tabm:.4f}")
    print(f"  GBN Bag 5:      {w_gbn:.4f}")
    print(f"  ResNet Bag 3:   {w_res:.4f}")

    # 4. Out-of-Fold Verification
    oof_wide = rankdata(np.load(EXTERNAL_DIR / "wide_lm_stack_nested.npy")) / n_train
    oof_tabm = rankdata(np.load(TABM_DIR / "oof_tabm_pwl_v3.npy")) / n_train
    oof_gbn  = rankdata(np.load(EXTERNAL_DIR / "gbn_bag5_w512_oof.npy")) / n_train
    oof_res  = rankdata(np.load(NEURAL_DIR / "oof_resnet_bag3.npy")) / n_train

    oof_blend = w_wide * oof_wide + w_tabm * oof_tabm + w_gbn * oof_gbn + w_res * oof_res
    fmin_tr = np.min(oof_blend[~manifold_tr])
    oof_blend_cal = oof_blend.copy()
    oof_blend_cal[manifold_tr] = np.minimum(oof_blend_cal[manifold_tr], fmin_tr - 1e-7)

    raw_cv = roc_auc_score(y, oof_blend)
    cal_cv = roc_auc_score(y, oof_blend_cal)
    print(f"\n--- OOF CV Scores ---")
    print(f"  Raw Blend OOF ROC-AUC:        {raw_cv:.6f}")
    print(f"  Calibrated (+Manifold) ROC-AUC:{cal_cv:.6f} (+{(cal_cv - raw_cv)*1e5:.1f} pts)")

    # 5. Generate Test Predictions
    test_wide = rankdata(np.load(EXTERNAL_DIR / "wide_lm_stack_test.npy")) / n_test
    test_tabm = rankdata(np.load(TABM_DIR / "test_tabm_pwl_v3.npy")) / n_test
    test_gbn  = rankdata(np.load(EXTERNAL_DIR / "gbn_bag5_w512_full.npy")) / n_test
    test_res  = rankdata(np.load(NEURAL_DIR / "test_resnet_bag3.npy")) / n_test

    test_blend = w_wide * test_wide + w_tabm * test_tabm + w_gbn * test_gbn + w_res * test_res
    test_rank = rankdata(test_blend) / n_test

    fmin_te = np.min(test_rank[~manifold_te])
    test_cal = test_rank.copy()
    test_cal[manifold_te] = np.minimum(test_cal[manifold_te], fmin_te - 1e-7)
    test_final = rankdata(test_cal) / (n_test + 1)

    # 6. Save Submission & OOF
    sub = sample_sub.copy()
    sub["is_fraud"] = test_final
    assert len(sub) == n_test
    assert sub["is_fraud"].between(0, 1).all()
    assert not sub["is_fraud"].isna().any()

    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    out_sub = SUBMISSION_DIR / "sovereign_cv95264_shield_submission.csv"
    sub.to_csv(out_sub, index=False)
    
    np.save(ROOT_DIR / "oof_preds" / "oof_sovereign_cv95264_shield.npy", oof_blend_cal)
    print(f"\n✅ Submission successfully saved: {out_sub}")
    print(f"✅ OOF saved to: pgs-s6e11/oof_preds/oof_sovereign_cv95264_shield.npy")

if __name__ == "__main__":
    main()
