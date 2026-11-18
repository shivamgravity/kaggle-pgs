#!/usr/bin/env python3
"""
Create Quad-Directional Meta-Apex Submission (All-Time Record 5-Fold CV: 0.952660)
Combines 4 independent, leak-free directional architectural updates in logit-rank space:
1. TabM 16-Head Batch-Ensemble GPU Delta:  d_tabm   = z_tabm - z_dual           (w = 0.030)
2. Tabular ResNet Entity-Embedding Delta:   d_resnet = z_resnet - z_ft           (w = 0.070)
3. 10-Bag GBDT Tree Diversity Delta:        d_tree   = z_gb_bag10 - z_gb_ohk     (w = 0.180)
4. Shallow Tree Regularization Delta:       d_depth  = z_lgb_l3 - z_lgb_l4       (w = 0.050)

Advances 5-Fold Stratified OOF CV to:
- Pooled ROC-AUC: 0.952660 (+6.35 pts over Grand Apex, +3.1 pts over Peak TabM)
- Mean Fold ROC-AUC: 0.952663 (+4.32 pts over Peak TabM)
- PR-AUC: 0.543337
"""

import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()

def rl(v: np.ndarray) -> np.ndarray:
    """Mid-rank logit transformation."""
    r = (rankdata(v) - 0.5) / len(v)
    return np.log(r / (1.0 - r))

def main():
    root = Path(__file__).resolve().parents[2]
    sub_dir = root / "submission"
    dataset_dir = root / "dataset"
    dual_dir = root / "code/dual_neural_multiseed/output"
    tabm_dir = root / "code/tabm_gpu/output"
    bank_dir = root / "external_models/oof_bank"

    print("1. Loading OOF predictions and labels...")
    train = pd.read_csv(dataset_dir / "train.csv", usecols=["id", "is_fraud"])
    y = train["is_fraud"].values.astype(int)

    oof_apex = np.load(sub_dir / "oof_grand_apex_mega_ensemble.npy")
    oof_tabm = np.load(tabm_dir / "oof_tabm.npy")
    oof_dual = np.load(dual_dir / "oof_dual_neural_bag3.npy")
    oof_res = np.load(dual_dir / "oof_resnet_bag3.npy")
    oof_ft = np.load(dual_dir / "oof_ft_transformer_bag3.npy")
    oof_gb = np.load(bank_dir / "oof_gb_bag10.npy")
    oof_ohk = np.load(bank_dir / "oof_gb_ohk.npy")
    oof_l3 = np.load(bank_dir / "oof_lgb_diglead_l3.npy")
    oof_l4 = np.load(bank_dir / "oof_lgb_diglead_l4.npy")

    z_apex = rl(oof_apex)
    d_tabm = rl(oof_tabm) - rl(oof_dual)
    d_resnet = rl(oof_res) - rl(oof_ft)
    d_tree = rl(oof_gb) - rl(oof_ohk)
    d_depth = rl(oof_l3) - rl(oof_l4)

    w_tabm = 0.030
    w_resnet = 0.070
    w_tree = 0.180
    w_depth = 0.050

    z_quad_oof = z_apex + w_tabm * d_tabm + w_resnet * d_resnet + w_tree * d_tree + w_depth * d_depth
    r_quad_oof = (rankdata(z_quad_oof) - 0.5) / len(z_quad_oof)

    auc_pooled = roc_auc_score(y, r_quad_oof)
    pr_pooled = average_precision_score(y, r_quad_oof)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_aucs = [roc_auc_score(y[va_idx], r_quad_oof[va_idx]) for _, va_idx in skf.split(train, y)]
    fold_mean = np.mean(fold_aucs)

    print("=" * 75)
    print("👑 VALIDATION RECORD: QUAD-DIRECTIONAL MULTI-PARADIGM ENSEMBLE")
    print("=" * 75)
    print(f"Pooled 5-Fold ROC-AUC: {auc_pooled:.6f} (+{(auc_pooled - 0.952597)*1e5:+.2f} pts over Grand Apex!)")
    print(f"Mean 5-Fold ROC-AUC:   {fold_mean:.6f} (+{(fold_mean - 0.952620)*1e5:+.2f} pts over Apex TabM!)")
    print(f"PR-AUC:                {pr_pooled:.6f}")
    print("Per-Fold Breakdown:")
    for f in range(5):
        print(f"  Fold {f}: {fold_aucs[f]:.6f}")
    print("=" * 75)

    oof_out_path = sub_dir / "oof_frontier_apex_quad_directional_peak.npy"
    np.save(oof_out_path, r_quad_oof.astype(np.float32))
    print(f"Saved new peak OOF: {oof_out_path}")

    # Build Test Submission
    print("\n2. Building Test Submission...")
    sub_hybrid_path = sub_dir / "frontier_grand_apex_hybrid_submission.csv"
    sub_apex_path = sub_dir / "frontier_apex_tabm_directional_submission.csv"
    sample_sub_path = dataset_dir / "sample_submission.csv"

    sub_hybrid = pd.read_csv(sub_hybrid_path)
    sub_apex = pd.read_csv(sub_apex_path)
    sample_sub = pd.read_csv(sample_sub_path)

    test_tabm = np.load(tabm_dir / "test_tabm.npy")
    test_dual = np.load(dual_dir / "test_dual_neural_bag3.npy")
    test_res = np.load(dual_dir / "test_resnet_bag3.npy")
    test_ft = np.load(dual_dir / "test_ft_transformer_bag3.npy")
    test_gb = np.load(bank_dir / "test_gb_bag10.npy")
    test_ohk = np.load(bank_dir / "test_gb_ohk.npy")
    test_l3 = np.load(bank_dir / "test_lgb_diglead_l3.npy")
    test_l4 = np.load(bank_dir / "test_lgb_diglead_l4.npy")

    d_test_tabm = rl(test_tabm) - rl(test_dual)
    d_test_res = rl(test_res) - rl(test_ft)
    d_test_tree = rl(test_gb) - rl(test_ohk)
    d_test_depth = rl(test_l3) - rl(test_l4)

    z_base = rl(sub_hybrid["is_fraud"].values)
    z_quad_test = z_base + w_tabm * d_test_tabm + w_resnet * d_test_res + w_tree * d_test_tree + w_depth * d_test_depth
    r_quad_test = (rankdata(z_quad_test) - 0.5) / len(z_quad_test)

    sp_apex = spearmanr(r_quad_test, sub_apex["is_fraud"].values)[0]
    sp_hyb = spearmanr(r_quad_test, sub_hybrid["is_fraud"].values)[0]
    print(f"Test rank correlation to ApexTabM (0.95467): {sp_apex:.6f}")
    print(f"Test rank correlation to Hybrid (0.95463):  {sp_hyb:.6f}")

    out_df = pd.DataFrame({
        "id": sample_sub["id"].values,
        "is_fraud": r_quad_test
    })

    assert len(out_df) == 299984
    assert out_df["is_fraud"].isna().sum() == 0
    assert not np.isinf(out_df["is_fraud"]).any()
    assert (out_df["is_fraud"] > 0).all() and (out_df["is_fraud"] < 1).all()

    out_path = sub_dir / "frontier_apex_quad_directional_submission.csv"
    out_df.to_csv(out_path, index=False)
    out_sha = sha256_file(out_path)
    print(f"Generated Submission: {out_path}")
    print(f"File size: {out_path.stat().st_size / (1024*1024):.2f} MB")
    print(f"SHA-256:   {out_sha}")

if __name__ == "__main__":
    main()
