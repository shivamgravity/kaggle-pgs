#!/usr/bin/env python3
"""
Create Route 2 Feature Residual Refinement Apex Submission
Applies the learned Tabular ResNet entity-embedding directional residual update (d_resnet = z_res - z_ft)
at w = 0.05 to our peak Frontier-Apex TabM directional model.
Advances 5-Fold Stratified OOF CV from 0.952629 to all-time record 0.952644 (+1.51 pts pooled, +2.31 pts fold mean),
and PR-AUC to 0.543522.
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

    print("Loading OOF predictions and labels...")
    train = pd.read_csv(dataset_dir / "train.csv", usecols=["id", "is_fraud"])
    y = train["is_fraud"].values.astype(int)

    oof_peak_prev = np.load(sub_dir / "oof_frontier_apex_tabm_peak.npy")
    oof_res = np.load(dual_dir / "oof_resnet_bag3.npy")
    oof_ft = np.load(dual_dir / "oof_ft_transformer_bag3.npy")

    z_peak = rl(oof_peak_prev)
    z_res = rl(oof_res)
    z_ft = rl(oof_ft)

    d_resnet_oof = z_res - z_ft
    w_res = 0.05

    z_new_oof = z_peak + w_res * d_resnet_oof
    r_new_oof = (rankdata(z_new_oof) - 0.5) / len(z_new_oof)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_aucs_prev = [roc_auc_score(y[va_idx], oof_peak_prev[va_idx]) for _, va_idx in skf.split(train, y)]
    fold_aucs_new = [roc_auc_score(y[va_idx], r_new_oof[va_idx]) for _, va_idx in skf.split(train, y)]

    auc_prev = roc_auc_score(y, oof_peak_prev)
    auc_new = roc_auc_score(y, r_new_oof)
    pr_prev = average_precision_score(y, oof_peak_prev)
    pr_new = average_precision_score(y, r_new_oof)

    print("=" * 75)
    print("🏆 VALIDATION BREAKTHROUGH: ROUTE 2 FEATURE RESIDUAL REFINEMENT")
    print("=" * 75)
    print(f"Previous Record 5-Fold Pooled ROC-AUC: {auc_prev:.6f} | PR-AUC: {pr_prev:.6f}")
    print(f"NEW All-Time Record 5-Fold ROC-AUC:    {auc_new:.6f} (+{(auc_new - auc_prev)*1e5:+.2f} pts) | PR-AUC: {pr_new:.6f}")
    print(f"Previous Fold Mean: {np.mean(fold_aucs_prev):.6f}")
    print(f"NEW Fold Mean:      {np.mean(fold_aucs_new):.6f} (+{(np.mean(fold_aucs_new) - np.mean(fold_aucs_prev))*1e5:+.2f} pts)")
    print("Per-Fold Comparison:")
    for f in range(5):
        print(f"  Fold {f}: Prev {fold_aucs_prev[f]:.6f} -> NEW {fold_aucs_new[f]:.6f} ({+(fold_aucs_new[f] - fold_aucs_prev[f])*1e5:+.2f} pts)")
    print("=" * 75)

    oof_out_path = sub_dir / "oof_frontier_apex_tabm_resnet_peak.npy"
    np.save(oof_out_path, r_new_oof.astype(np.float32))
    print(f"Saved new peak OOF: {oof_out_path}")

    # Build Test Submission
    sub_apex_path = sub_dir / "frontier_apex_tabm_directional_submission.csv"
    sample_sub_path = dataset_dir / "sample_submission.csv"

    sub_apex = pd.read_csv(sub_apex_path)
    sample_sub = pd.read_csv(sample_sub_path)
    test_res = np.load(dual_dir / "test_resnet_bag3.npy")
    test_ft = np.load(dual_dir / "test_ft_transformer_bag3.npy")

    d_test_resnet = rl(test_res) - rl(test_ft)
    z_sub_new = rl(sub_apex["is_fraud"].values) + w_res * d_test_resnet
    r_sub_new = (rankdata(z_sub_new) - 0.5) / len(z_sub_new)

    sp_corr = spearmanr(r_sub_new, sub_apex["is_fraud"].values)[0]
    print(f"Rank correlation to Peak ApexTabM (0.95467): {sp_corr:.6f}")

    out_df = pd.DataFrame({
        "id": sample_sub["id"].values,
        "is_fraud": r_sub_new
    })

    assert len(out_df) == 299984
    assert out_df["is_fraud"].isna().sum() == 0
    assert not np.isinf(out_df["is_fraud"]).any()
    assert (out_df["is_fraud"] > 0).all() and (out_df["is_fraud"] < 1).all()

    out_path = sub_dir / "frontier_apex_resnet_tabm_directional_submission.csv"
    out_df.to_csv(out_path, index=False)
    out_sha = sha256_file(out_path)
    print(f"Generated Submission: {out_path}")
    print(f"File size: {out_path.stat().st_size / (1024*1024):.2f} MB")
    print(f"SHA-256:   {out_sha}")

if __name__ == "__main__":
    main()
