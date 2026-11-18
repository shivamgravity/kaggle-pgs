#!/usr/bin/env python3
"""
Create Tri-Ladder Power-Scaled Quad Submission
Combines our all-time record Power-Scaled Quad-Directional model (0.952664 5-Fold CV)
with the 3-Ladder Consensus (goodpjw2008 G13 + BhuvanNR B10 + Areeb Ali P31)
in logit-rank space: 85% PowerScaledQuad + 15% ThreeBases.
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

    quad_sub_path = sub_dir / "frontier_apex_power_scaled_quad_submission.csv"
    three_bases_path = Path("/tmp/busyaprime_output/candidate_three_bases.csv")
    sample_sub_path = dataset_dir / "sample_submission.csv"

    assert quad_sub_path.exists(), f"Quad sub not found: {quad_sub_path}"
    assert three_bases_path.exists(), f"Three bases not found: {three_bases_path}"
    assert sample_sub_path.exists(), f"Sample sub not found: {sample_sub_path}"

    print(f"Loading Power-Scaled Quad Peak (CV 0.952664): {quad_sub_path}")
    quad_sub = pd.read_csv(quad_sub_path)
    print(f"Loading Three-Bases Consensus:               {three_bases_path}")
    three_bases = pd.read_csv(three_bases_path)
    sample_sub = pd.read_csv(sample_sub_path)

    z_quad = rl(quad_sub["is_fraud"].values)
    z_c3 = rl(three_bases["is_fraud"].values)

    w_quad = 0.85
    w_c3 = 0.15

    print(f"Blending with weights: {w_quad:.2f} QuadPeak + {w_c3:.2f} ThreeBases in logit-rank space...")
    z_tri = w_quad * z_quad + w_c3 * z_c3
    r_tri = (rankdata(z_tri) - 0.5) / len(z_tri)

    sp_quad = spearmanr(r_tri, quad_sub["is_fraud"].values)[0]
    sp_c3 = spearmanr(r_tri, three_bases["is_fraud"].values)[0]
    print(f"Correlation to Quad-Peak:   {sp_quad:.6f}")
    print(f"Correlation to Three-Bases: {sp_c3:.6f}")

    out_df = pd.DataFrame({
        "id": sample_sub["id"].values,
        "is_fraud": r_tri
    })

    assert len(out_df) == 299984
    assert out_df["is_fraud"].isna().sum() == 0
    assert not np.isinf(out_df["is_fraud"]).any()
    assert (out_df["is_fraud"] > 0).all() and (out_df["is_fraud"] < 1).all()

    out_path = sub_dir / "tri_ladder_power_scaled_quad_submission.csv"
    out_df.to_csv(out_path, index=False)
    out_sha = sha256_file(out_path)
    print(f"Successfully generated: {out_path}")
    print(f"File size: {out_path.stat().st_size / (1024*1024):.2f} MB")
    print(f"SHA-256:   {out_sha}")

if __name__ == "__main__":
    main()
