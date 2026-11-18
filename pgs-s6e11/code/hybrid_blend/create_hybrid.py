#!/usr/bin/env python3
"""
Create Frontier-GrandApex Hybrid Submission
Combines the top public frontier consensus (0.95465 Public LB, 124-model wide LM stack + TabM/RealMLP)
with our best independent, out-of-fold validated Grand Apex Mega-Ensemble (0.95292 Public LB, 0.952597 5-Fold CV)
using logit-rank blending: 85% Frontier + 15% Grand Apex.
"""

import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()

def rl(v: np.ndarray) -> np.ndarray:
    """Mid-rank logit transformation for invariant rank scaling."""
    r = (rankdata(v) - 0.5) / len(v)
    return np.log(r / (1.0 - r))

def main():
    root = Path(__file__).resolve().parents[2]
    sub_dir = root / "submission"
    dataset_dir = root / "dataset"

    frontier_path = Path("/tmp/busyaprime_output/submission.csv")
    apex_path = sub_dir / "grand_apex_mega_ensemble_submission.csv"
    sample_sub_path = dataset_dir / "sample_submission.csv"

    assert frontier_path.exists(), f"Frontier submission not found: {frontier_path}"
    assert apex_path.exists(), f"Grand Apex submission not found: {apex_path}"
    assert sample_sub_path.exists(), f"Sample submission not found: {sample_sub_path}"

    print(f"Loading Frontier (0.95465): {frontier_path}")
    sub_front = pd.read_csv(frontier_path)
    print(f"Loading Grand Apex (0.95292): {apex_path}")
    sub_apex = pd.read_csv(apex_path)
    sample_sub = pd.read_csv(sample_sub_path)

    # Verification of ID alignment
    assert len(sub_front) == len(sample_sub) == 299984
    assert len(sub_apex) == len(sample_sub) == 299984
    assert (sub_front["id"].values == sample_sub["id"].values).all()
    assert (sub_apex["id"].values == sample_sub["id"].values).all()

    z_front = rl(sub_front["is_fraud"].values)
    z_apex = rl(sub_apex["is_fraud"].values)

    w_front = 0.85
    w_apex = 0.15

    print(f"Blending with weights: {w_front:.2f} Frontier + {w_apex:.2f} Grand Apex in logit-rank space...")
    z_hybrid = w_front * z_front + w_apex * z_apex
    r_hybrid = (rankdata(z_hybrid) - 0.5) / len(z_hybrid)

    sp_f = spearmanr(r_hybrid, sub_front["is_fraud"].values)[0]
    sp_a = spearmanr(r_hybrid, sub_apex["is_fraud"].values)[0]
    print(f"Hybrid rank correlation to Frontier:   {sp_f:.6f}")
    print(f"Hybrid rank correlation to Grand Apex: {sp_a:.6f}")

    out_df = pd.DataFrame({
        "id": sample_sub["id"].values,
        "is_fraud": r_hybrid
    })

    assert len(out_df) == 299984
    assert out_df["is_fraud"].isna().sum() == 0
    assert not np.isinf(out_df["is_fraud"]).any()
    assert (out_df["is_fraud"] > 0).all() and (out_df["is_fraud"] < 1).all()

    out_path = sub_dir / "frontier_grand_apex_hybrid_submission.csv"
    out_df.to_csv(out_path, index=False)
    out_sha = sha256_file(out_path)
    print(f"Successfully generated: {out_path}")
    print(f"File size: {out_path.stat().st_size / (1024*1024):.2f} MB")
    print(f"SHA-256: {out_sha}")

if __name__ == "__main__":
    main()
