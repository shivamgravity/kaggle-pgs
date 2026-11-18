#!/usr/bin/env python3
"""
Generate Frontier-Apex-TabM Directional Submission & Record Peak CV
Applies the first-party TabM directional update (+3.23 pts OOF gain, CV 0.952629)
to our top-performing Frontier-Apex Hybrid submission (0.95463 Public LB, Rank #37).
"""

import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score

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
    tabm_dir = Path(__file__).resolve().parent / "output"
    sub_dir = root / "submission"
    dataset_dir = root / "dataset"

    # 1. Update OOF Benchmark
    print("Loading OOF predictions and labels...")
    train = pd.read_csv(dataset_dir / "train.csv")
    y = train["is_fraud"].values.astype(int)

    oof_apex = np.load(sub_dir / "oof_grand_apex_mega_ensemble.npy")
    oof_tabm = np.load(tabm_dir / "oof_tabm.npy")
    oof_dual = np.load(root / "code/dual_neural_multiseed/output/oof_dual_neural_bag3.npy")

    z_apex = rl(oof_apex)
    z_tabm = rl(oof_tabm)
    z_dual = rl(oof_dual)
    d_tabm_oof = z_tabm - z_dual

    w_step = 0.025
    z_peak_oof = z_apex + w_step * d_tabm_oof
    r_peak_oof = (rankdata(z_peak_oof) - 0.5) / len(z_peak_oof)

    base_auc = roc_auc_score(y, oof_apex)
    peak_auc = roc_auc_score(y, r_peak_oof)
    peak_prauc = average_precision_score(y, r_peak_oof)

    print("=" * 65)
    print(f"Previous Peak Grand Apex CV: {base_auc:.6f}")
    print(f"New Peak Apex+TabM CV:       {peak_auc:.6f} (+{(peak_auc - base_auc)*1e5:+.2f} pts)")
    print(f"New Peak Apex+TabM PR-AUC:   {peak_prauc:.6f}")
    print("=" * 65)

    np.save(sub_dir / "oof_frontier_apex_tabm_peak.npy", r_peak_oof.astype(np.float32))

    # 2. Build Test Submission
    print("\nBuilding Directionally Enhanced Submission...")
    sub_hybrid_path = sub_dir / "frontier_grand_apex_hybrid_submission.csv"
    sample_sub_path = dataset_dir / "sample_submission.csv"

    sub_hybrid = pd.read_csv(sub_hybrid_path)
    sample_sub = pd.read_csv(sample_sub_path)
    test_tabm = np.load(tabm_dir / "test_tabm.npy")
    test_dual = np.load(root / "code/dual_neural_multiseed/output/test_dual_neural_bag3.npy")

    d_test = rl(test_tabm) - rl(test_dual)
    z_sub_peak = rl(sub_hybrid["is_fraud"].values) + w_step * d_test
    r_sub_peak = (rankdata(z_sub_peak) - 0.5) / len(z_sub_peak)

    sp_corr = spearmanr(r_sub_peak, sub_hybrid["is_fraud"].values)[0]
    print(f"Rank correlation to previous 0.95463 Hybrid: {sp_corr:.6f}")

    out_df = pd.DataFrame({
        "id": sample_sub["id"].values,
        "is_fraud": r_sub_peak
    })

    assert len(out_df) == 299984
    assert out_df["is_fraud"].isna().sum() == 0
    assert not np.isinf(out_df["is_fraud"]).any()
    assert (out_df["is_fraud"] > 0).all() and (out_df["is_fraud"] < 1).all()

    out_path = sub_dir / "frontier_apex_tabm_directional_submission.csv"
    out_df.to_csv(out_path, index=False)
    out_sha = sha256_file(out_path)
    print(f"Generated: {out_path}")
    print(f"File size: {out_path.stat().st_size / (1024*1024):.2f} MB")
    print(f"SHA-256:   {out_sha}")

if __name__ == "__main__":
    main()
