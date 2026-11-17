import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.special import logit, expit
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

print("=" * 75)
print("🧠 PGS-S6E11: MEGA-STACKER ACROSS ALL VALIDATED HIGH-ORDER PARADIGMS")
print("=" * 75)

train_df = pd.read_csv("pgs-s6e11/dataset/train.csv", usecols=["is_fraud"])
y = train_df["is_fraud"].values
N_TR = len(y)

test_sample = pd.read_csv("pgs-s6e11/dataset/sample_submission.csv")
N_TE = len(test_sample)

models_oof = {}
models_test = {}

# 1. Wide Stacks
models_oof["wide_lm_stack"] = np.load("pgs-s6e11/external_models/artifacts/wide_lm_stack_nested.npy")
models_test["wide_lm_stack"] = np.load("pgs-s6e11/external_models/artifacts/wide_lm_stack_test.npy")

models_oof["wide_stack"] = np.load("pgs-s6e11/external_models/artifacts/wide_stack_nested.npy")
models_test["wide_stack"] = np.load("pgs-s6e11/external_models/artifacts/wide_stack_test.npy")

# 2. Language Models (GPT-2)
models_oof["gpt2_s2026"] = np.load("pgs-s6e11/external_models/gpt2_seed2026/gpt2_s2026_oof_5fold.npy")
models_test["gpt2_s2026"] = np.load("pgs-s6e11/external_models/gpt2_seed2026/gpt2_s2026_test_5fold.npy")

models_oof["megalm_gpt2"] = np.load("pgs-s6e11/external_models/megalm_oof_sha181c.npy")
models_test["megalm_gpt2"] = np.load("pgs-s6e11/external_models/megalm_test_sha3168.npy")

models_oof["lm7_gpt2"] = np.load("pgs-s6e11/external_models/lm7_oof_5fold.npy")
models_test["lm7_gpt2"] = np.load("pgs-s6e11/external_models/lm7_test_5fold.npy")

# 3. Our Multi-Paradigm Standalone Pipelines
models_oof["champ_stack52"] = np.load("pgs-s6e11/submission/oof_grand_champion_stack.npy")
models_test["champ_stack52"] = pd.read_csv("pgs-s6e11/submission/grand_champion_meta_stack_submission.csv")["is_fraud"].values

models_oof["triple_te"] = np.load("pgs-s6e11/code/triple_glm_joint_te_boost/output/oof_triple_blend.npy")
models_test["triple_te"] = pd.read_csv("pgs-s6e11/code/triple_glm_joint_te_boost/output/triple_glm_joint_te_blend_submission.csv")["is_fraud"].values

models_oof["velocity_triple"] = np.load("pgs-s6e11/code/velocity_burst_glm_boost/output/oof_velocity_blend.npy")
models_test["velocity_triple"] = pd.read_csv("pgs-s6e11/code/velocity_burst_glm_boost/output/velocity_burst_glm_blend_submission.csv")["is_fraud"].values

models_oof["dual_margin"] = np.load("pgs-s6e11/code/dual_glm_digit_te_boost/output/oof_dual_margin_blend.npy")
models_test["dual_margin"] = pd.read_csv("pgs-s6e11/code/dual_glm_digit_te_boost/output/dual_margin_te_blend_submission.csv")["is_fraud"].values

models_oof["tabular_nn"] = np.load("pgs-s6e11/code/tabular_nn_boost/output/oof_tabular_nn.npy")
models_test["tabular_nn"] = pd.read_csv("pgs-s6e11/code/tabular_nn_boost/output/tabular_nn_submission.csv")["is_fraud"].values

# 4. Neural MLPs & EBMs from Artifacts
models_oof["realmlp_dig"] = np.load("pgs-s6e11/external_models/artifacts/realmlp_dig_oof.npy")
models_test["realmlp_dig"] = np.load("pgs-s6e11/external_models/artifacts/realmlp_dig_test.npy")

models_oof["ebm_i25"] = np.load("pgs-s6e11/external_models/artifacts/ebm_i25_bag4_oof.npy")
models_test["ebm_i25"] = np.load("pgs-s6e11/external_models/artifacts/ebm_i25_bag4_full.npy")

models_oof["gbn_w512"] = np.load("pgs-s6e11/external_models/artifacts/gbn_bag5_w512_oof.npy")
models_test["gbn_w512"] = np.load("pgs-s6e11/external_models/artifacts/gbn_w512_testbag_full.npy")

print(f"Total model paradigms assembled: {len(models_oof)}")
print("-" * 65)
for name, arr in models_oof.items():
    auc = roc_auc_score(y, arr)
    pr = average_precision_score(y, arr)
    print(f"  {name:20s}: ROC-AUC = {auc:.6f} | PR-AUC = {pr:.6f}")
print("-" * 65)

# Nelder-Mead Optimization on Rank-Averages
names = list(models_oof.keys())
oof_ranks = [rankdata(models_oof[m]) / N_TR for m in names]
test_ranks = [rankdata(models_test[m]) / N_TE for m in names]

def simplex_loss(w):
    w = np.maximum(w, 0)
    s = w.sum()
    if s == 0: return 0.0
    w = w / s
    b = sum(w[i] * oof_ranks[i] for i in range(len(names)))
    return -roc_auc_score(y, b)

init_w = np.zeros(len(names))
init_w[names.index("wide_lm_stack")] = 0.80
init_w[names.index("champ_stack52")] = 0.10
init_w[names.index("tabular_nn")] = 0.05
init_w[names.index("velocity_triple")] = 0.05

res = minimize(simplex_loss, init_w, method="Nelder-Mead", options={"maxiter": 400})
best_w = np.maximum(res.x, 0)
best_w /= best_w.sum()

best_oof_blend = sum(best_w[i] * oof_ranks[i] for i in range(len(names)))
best_test_blend = sum(best_w[i] * test_ranks[i] for i in range(len(names)))

auc_blend = roc_auc_score(y, best_oof_blend)
pr_blend = average_precision_score(y, best_oof_blend)

print("\n" + "=" * 75)
print(f"🏆 OPTIMAL PARADIGM SIMPLEX RESULTS")
print("=" * 75)
print(f"5-Fold OOF ROC-AUC : {auc_blend:.6f}")
print(f"5-Fold OOF PR-AUC  : {pr_blend:.6f}")
print("\nNon-zero Model Weights:")
for name, w in sorted(zip(names, best_w), key=lambda x: x[1], reverse=True):
    if w > 0.001:
        print(f"  {name:20s}: {w*100:5.2f}%")
print("=" * 75)

# Save Master Mega-Ensemble Prediction
out_dir = Path("pgs-s6e11/submission")
sub = test_sample.copy()
sub["is_fraud"] = rankdata(best_test_blend) / N_TE
sub_path = out_dir / "master_mega_ensemble_submission.csv"
sub.to_csv(sub_path, index=False)
np.save(out_dir / "oof_master_mega_ensemble.npy", best_oof_blend)
print(f"Generated Master Mega-Ensemble Submission: {sub_path}")
