#!/usr/bin/env python3
"""
Evaluate First-Party TabM Deep Ensemble OOF & Test Blend
Computes 5-fold ROC-AUC, PR-AUC, rank correlations with our model bank,
and constructs the optimal directional update and hybrid submissions.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score

def rl(v: np.ndarray) -> np.ndarray:
    """Mid-rank logit transformation."""
    r = (rankdata(v) - 0.5) / len(v)
    return np.log(r / (1.0 - r))

def main():
    root = Path(__file__).resolve().parents[2]
    tabm_dir = Path(__file__).resolve().parent / "output"
    sub_dir = root / "submission"
    dataset_dir = root / "dataset"

    oof_tabm_path = tabm_dir / "oof_tabm.npy"
    test_tabm_path = tabm_dir / "test_tabm.npy"

    if not oof_tabm_path.exists():
        print(f"Waiting for TabM output files in: {tabm_dir}")
        return

    print("Loading datasets and OOF predictions...")
    train = pd.read_csv(dataset_dir / "train.csv")
    y = train["is_fraud"].values.astype(int)

    oof_tabm = np.load(oof_tabm_path)
    test_tabm = np.load(test_tabm_path)

    auc_tabm = roc_auc_score(y, oof_tabm)
    prauc_tabm = average_precision_score(y, oof_tabm)
    print("=" * 60)
    print(f"TabM 5-Fold OOF ROC-AUC: {auc_tabm:.6f}")
    print(f"TabM 5-Fold OOF PR-AUC:  {prauc_tabm:.6f}")
    print("=" * 60)

    # Compare with our other OOF models
    oof_stack = np.load("/tmp/busyaprime_output/stack_nested_oof.npy")
    oof_apex = np.load(sub_dir / "oof_grand_apex_mega_ensemble.npy")
    oof_dual = np.load(root / "code/dual_neural_multiseed/output/oof_dual_neural_bag3.npy")
    oof_glm = np.load(root / "code/glm_10fold_boost/output/oof_10f_blend.npy")

    print("\nSpearman Correlation with TabM:")
    print(f"  Wide LM Stack (0.952110):     {spearmanr(oof_tabm, oof_stack)[0]:.5f}")
    print(f"  Grand Apex Ensemble (0.952597): {spearmanr(oof_tabm, oof_apex)[0]:.5f}")
    print(f"  Dual Neural Bag3 (0.943441):  {spearmanr(oof_tabm, oof_dual)[0]:.5f}")
    print(f"  GLM 10-Fold Booster (0.950647): {spearmanr(oof_tabm, oof_glm)[0]:.5f}")

    # Test Directional Delta on Stack
    z_stack = rl(oof_stack)
    z_tabm = rl(oof_tabm)
    z_dual = rl(oof_dual)

    # Directional update: TabM minus Dual Neural
    d_tabm = z_tabm - z_dual
    base_auc = roc_auc_score(y, z_stack)

    print(f"\nBaseline Stack OOF AUC: {base_auc:.6f}")
    for w in [0.01, 0.02, 0.03, 0.05, 0.08, 0.10]:
        score = roc_auc_score(y, z_stack + w * d_tabm)
        diff = (score - base_auc) * 1e5
        print(f"Directional Step w={w:.2f} -> AUC={score:.6f} ({diff:+.2f} pts)")

if __name__ == "__main__":
    main()
