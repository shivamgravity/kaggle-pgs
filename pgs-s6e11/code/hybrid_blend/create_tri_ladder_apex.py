#!/usr/bin/env python3
"""
Create Tri-Ladder Meta-Apex Submission (Route 1)
Combines our top-performing Frontier-Apex TabM Directional model (0.95467 Public LB, 0.952629 5-Fold CV)
with the 3-Ladder Consensus (combining goodpjw2008 G13 + BhuvanNR B10 + Areeb Ali P31)
in logit-rank space: 85% ApexTabM + 15% ThreeBases.
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
    """Mid-rank logit transformation."""
    r = (rankdata(v) - 0.5) / len(v)
    return np.log(r / (1.0 - r))

def main():
    root = Path(__file__).resolve().parents[2]
    sub_dir = root / "submission"
    dataset_dir = root / "dataset"

    apex_tabm_path = sub_dir / "frontier_apex_tabm_directional_submission.csv"
    three_bases_path = Path("/tmp/busyaprime_output/candidate_three_bases.csv")
    sample_sub_path = dataset_dir / "sample_submission.csv"

    assert apex_tabm_path.exists(), f"ApexTabM submission not found: {apex_tabm_path}"
    assert three_bases_path.exists(), f"ThreeBases submission not found: {three_bases_path}"
    assert sample_sub_path.exists(), f"Sample submission not found: {sample_sub_path}"

    print(f"Loading ApexTabM Peak (0.95467): {apex_tabm_path}")
    sub_apex = pd.read_csv(apex_tabm_path)
    print(f"Loading ThreeBases Consensus:    {three_bases_path}")
    sub_c3 = pd.read_csv(three_bases_path)
    sample_sub = pd.read_csv(sample_sub_path)

    assert len(sub_apex) == len(sample_sub) == 299984
    assert len(sub_c3) == len(sample_sub) == 299984
    assert (sub_apex["id"].values == sample_sub["id"].values).all()
    assert (sub_c3["id"].values == sample_sub["id"].values).all()

    z_apex = rl(sub_apex["is_fraud"].values)
    z_c3 = rl(sub_c3["is_fraud"].values)

    w_apex = 0.85
    w_c3 = 0.15

    print(f"Blending with weights: {w_apex:.2f} ApexTabM + {w_c3:.2f} ThreeBases in logit-rank space...")
    z_tri = w_apex * z_apex + w_c3 * z_c3
    r_tri = (rankdata(z_tri) - 0.5) / len(z_tri)

    sp_a = spearmanr(r_tri, sub_apex["is_fraud"].values)[0]
    sp_c = spearmanr(r_tri, sub_c3["is_fraud"].values)[0]
    print(f"Tri-Ladder rank correlation to ApexTabM:   {sp_a:.6f}")
    print(f"Tri-Ladder rank correlation to ThreeBases: {sp_c:.6f}")

    out_df = pd.DataFrame({
        "id": sample_sub["id"].values,
        "is_fraud": r_tri
    })

    assert len(out_df) == 299984
    assert out_df["is_fraud"].isna().sum() == 0
    assert not np.isinf(out_df["is_fraud"]).any()
    assert (out_df["is_fraud"] > 0).all() and (out_df["is_fraud"] < 1).all()

    out_path = sub_dir / "tri_ladder_meta_apex_submission.csv"
    out_df.to_csv(out_path, index=False)
    out_sha = sha256_file(out_path)
    print(f"Successfully generated: {out_path}")
    print(f"File size: {out_path.stat().st_size / (1024*1024):.2f} MB")
    print(f"SHA-256:   {out_sha}")

if __name__ == "__main__":
    main()
