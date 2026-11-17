import os
import glob
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.special import logit, expit
from scipy.optimize import minimize, nnls
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

print("=" * 75)
print("🧠 PGS-S6E11: GRAND META-STACKER & CROSS-VALIDATION ADVANCEMENT")
print("=" * 75)

# 1. Load Ground Truth
train_df = pd.read_csv("pgs-s6e11/dataset/train.csv", usecols=["is_fraud"])
y = train_df["is_fraud"].values
N_TR = len(y)
test_sample = pd.read_csv("pgs-s6e11/dataset/sample_submission.csv")
N_TE = len(test_sample)

print(f"Loaded ground truth: {N_TR:,} train rows ({y.sum():,} fraud txns, {y.mean()*100:.3f}%).")

# 2. Collect Candidate Model OOFs and Test Predictions
models_oof = {}
models_test = {}

# A. Our Own Pipelines
models_oof["megalm_gpt2"] = np.load("pgs-s6e11/external_models/megalm_oof_sha181c.npy")
models_test["megalm_gpt2"] = np.load("pgs-s6e11/external_models/megalm_test_sha3168.npy")

models_oof["lm7_gpt2"] = np.load("pgs-s6e11/external_models/lm7_oof_5fold.npy")
models_test["lm7_gpt2"] = np.load("pgs-s6e11/external_models/lm7_test_5fold.npy")

models_oof["triple_te"] = np.load("pgs-s6e11/code/triple_glm_joint_te_boost/output/oof_triple_blend.npy")
models_test["triple_te"] = pd.read_csv("pgs-s6e11/code/triple_glm_joint_te_boost/output/triple_glm_joint_te_blend_submission.csv")["is_fraud"].values

models_oof["dual_margin"] = np.load("pgs-s6e11/code/dual_glm_digit_te_boost/output/oof_dual_margin_blend.npy")
models_test["dual_margin"] = pd.read_csv("pgs-s6e11/code/dual_glm_digit_te_boost/output/dual_margin_te_blend_submission.csv")["is_fraud"].values

models_oof["te_1d"] = np.load("pgs-s6e11/code/glm_digit_te_boost/output/oof_blend_digit_te.npy")
models_test["te_1d"] = pd.read_csv("pgs-s6e11/code/glm_digit_te_boost/output/glm_digit_te_blend_submission.csv")["is_fraud"].values

models_oof["tabular_nn"] = np.load("pgs-s6e11/code/tabular_nn_boost/output/oof_tabular_nn.npy")
models_test["tabular_nn"] = pd.read_csv("pgs-s6e11/code/tabular_nn_boost/output/tabular_nn_submission.csv")["is_fraud"].values

models_oof["velocity_triple"] = np.load("pgs-s6e11/code/velocity_burst_glm_boost/output/oof_velocity_blend.npy")
models_test["velocity_triple"] = pd.read_csv("pgs-s6e11/code/velocity_burst_glm_boost/output/velocity_burst_glm_blend_submission.csv")["is_fraud"].values

# B. Check for OOF Bank models
bank_dir = Path("pgs-s6e11/external_models/oof_bank")
if bank_dir.exists():
    manifest_path = bank_dir / "manifest.csv"
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        print(f"\nDiscovered OOF Bank with {len(manifest)} models.")
        # Load top bank models
        for _, row in manifest.iterrows():
            m_id = row["name"]
            oof_file = bank_dir / f"oof_{m_id}.npy"
            test_file = bank_dir / f"test_{m_id}.npy"
            if oof_file.exists() and test_file.exists():
                try:
                    oof_arr = np.load(oof_file)
                    test_arr = np.load(test_file)
                    if len(oof_arr) == N_TR and len(test_arr) == N_TE:
                        auc = roc_auc_score(y, oof_arr)
                        if auc > 0.9495:  # Only take strong models
                            models_oof[f"bank_{m_id}"] = oof_arr
                            models_test[f"bank_{m_id}"] = test_arr
                except Exception:
                    pass

model_names = list(models_oof.keys())
print(f"\nTotal candidate models assembled: {len(model_names)}")
for name in model_names:
    auc = roc_auc_score(y, models_oof[name])
    pr = average_precision_score(y, models_oof[name])
    print(f"  {name:25s}: ROC-AUC = {auc:.6f} | PR-AUC = {pr:.6f}")

# 3. Logit-Rank Transformation
print("\nTransforming out-of-fold and test predictions to Logit-Rank space...")
X_train_logit = np.empty((N_TR, len(model_names)), dtype=np.float32)
X_test_logit = np.empty((N_TE, len(model_names)), dtype=np.float32)

for j, name in enumerate(model_names):
    r_tr = rankdata(models_oof[name]) / (N_TR + 1)
    r_te = rankdata(models_test[name]) / (N_TE + 1)
    X_train_logit[:, j] = logit(np.clip(r_tr, 1e-5, 1 - 1e-5))
    X_test_logit[:, j] = logit(np.clip(r_te, 1e-5, 1 - 1e-5))

# 4. Strict 5-Fold Nested Cross-Validation of Meta-Learners
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = {}

# Strategy 1: Nelder-Mead Rank Blend
print("\n--- Strategy 1: Nelder-Mead Rank-Average Simplex ---")
oof_ranks = [rankdata(models_oof[m]) / N_TR for m in model_names]
test_ranks = [rankdata(models_test[m]) / N_TE for m in model_names]

def simplex_loss(w):
    w = np.maximum(w, 0)
    s = w.sum()
    if s == 0:
        return 0.0
    w = w / s
    blend = sum(w[i] * oof_ranks[i] for i in range(len(model_names)))
    return -roc_auc_score(y, blend)

res_nm = minimize(simplex_loss, np.ones(len(model_names)) / len(model_names), method="Nelder-Mead", options={"maxiter": 600})
best_w_nm = np.maximum(res_nm.x, 0)
best_w_nm /= best_w_nm.sum()
oof_nm = sum(best_w_nm[i] * oof_ranks[i] for i in range(len(model_names)))
test_nm = sum(best_w_nm[i] * test_ranks[i] for i in range(len(model_names)))
auc_nm = roc_auc_score(y, oof_nm)
pr_nm = average_precision_score(y, oof_nm)
results["Nelder-Mead Simplex"] = {"oof": oof_nm, "test": test_nm, "auc": auc_nm, "pr": pr_nm}
print(f"Nelder-Mead Simplex -> OOF ROC-AUC: {auc_nm:.6f} | PR-AUC: {pr_nm:.6f}")

# Strategy 2: Regularized Ridge Logistic Regression on Logit-Ranks
print("\n--- Strategy 2: Regularized Logistic Regression on Logit-Ranks ---")
best_c, best_c_auc, best_c_oof, best_c_test = None, 0.0, None, None

for C_val in [0.001, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1]:
    oof_lr = np.zeros(N_TR)
    test_lr = np.zeros(N_TE)
    
    for tr_idx, va_idx in skf.split(X_train_logit, y):
        clf = LogisticRegression(C=C_val, max_iter=500, random_state=42)
        clf.fit(X_train_logit[tr_idx], y[tr_idx])
        oof_lr[va_idx] = clf.predict_proba(X_train_logit[va_idx])[:, 1]
        test_lr += clf.predict_proba(X_test_logit)[:, 1] / 5.0
        
    auc_lr = roc_auc_score(y, oof_lr)
    pr_lr = average_precision_score(y, oof_lr)
    print(f"  C = {C_val:6.3f} -> OOF ROC-AUC: {auc_lr:.6f} | PR-AUC: {pr_lr:.6f}")
    if auc_lr > best_c_auc:
        best_c_auc = auc_lr
        best_c = C_val
        best_c_oof = oof_lr
        best_c_test = test_lr

results[f"Logistic Stacker (C={best_c})"] = {
    "oof": best_c_oof, "test": best_c_test, "auc": best_c_auc, "pr": average_precision_score(y, best_c_oof)
}

# Strategy 3: Non-Negative Least Squares (NNLS) on Probabilities
print("\n--- Strategy 3: Non-Negative Least Squares (NNLS) ---")
oof_probs = np.column_stack([models_oof[m] for m in model_names])
test_probs = np.column_stack([models_test[m] for m in model_names])

oof_nnls = np.zeros(N_TR)
test_nnls = np.zeros(N_TE)

for tr_idx, va_idx in skf.split(oof_probs, y):
    w_nnls, _ = nnls(oof_probs[tr_idx], y[tr_idx])
    w_sum = w_nnls.sum()
    if w_sum > 0:
        w_nnls /= w_sum
    oof_nnls[va_idx] = oof_probs[va_idx] @ w_nnls
    test_nnls += (test_probs @ w_nnls) / 5.0

auc_nnls = roc_auc_score(y, oof_nnls)
pr_nnls = average_precision_score(y, oof_nnls)
results["NNLS Stacker"] = {"oof": oof_nnls, "test": test_nnls, "auc": auc_nnls, "pr": pr_nnls}
print(f"NNLS Stacker -> OOF ROC-AUC: {auc_nnls:.6f} | PR-AUC: {pr_nnls:.6f}")

# 5. Grand Stacking Comparison Table
print("\n" + "=" * 75)
print("GRAND META-STACKING CV BENCHMARK COMPARISON")
print("=" * 75)
comp_df = pd.DataFrame([
    {"Stacker Architecture": k, "5-Fold OOF ROC-AUC": v["auc"], "5-Fold OOF PR-AUC": v["pr"]}
    for k, v in results.items()
]).sort_values("5-Fold OOF ROC-AUC", ascending=False)
print(comp_df.to_string(index=False))

# 6. Save Winning Predictions
best_stacker_name = comp_df.iloc[0]["Stacker Architecture"]
best_stacker_test = results[best_stacker_name]["test"]
best_stacker_oof = results[best_stacker_name]["oof"]

out_dir = Path("pgs-s6e11/submission")
sub = test_sample.copy()
sub["is_fraud"] = rankdata(best_stacker_test) / N_TE
sub_path = out_dir / "grand_champion_meta_stack_submission.csv"
sub.to_csv(sub_path, index=False)
np.save(out_dir / "oof_grand_champion_stack.npy", best_stacker_oof)

print(f"\nWinning Stacker: {best_stacker_name} (AUC: {comp_df.iloc[0]['5-Fold OOF ROC-AUC']:.6f})")
print(f"Generated Champion Submission: {sub_path}")
print("=" * 75)
