"""
Grand 83-Model Non-Negative Simplex Meta-Stacker for PGS-S6E11.

Evaluates an exhaustive 5-fold nested cross-validation stack across 83 verified model streams:
  - 70 models from OOF Bank (Trees, RealMLPs, EBMs, GLMs, Bags)
  - 7 models from Weni Mohamed BPE Pack (BPE LGB, CatBoost, TabM, RealMLP, GLM)
  - 1 model from GPU TabM-PWL V2
  - 1 model from GPT-2 Sequence LM 5-fold
  - 3 models from Dual Neural (ResNet-bag3, FT-Transformer-bag3, Dual NN)
  - 1 model from Quad Interaction Booster

Features:
  - Strict 5-Fold Nested CV (zero data leakage)
  - Non-negative convex optimization (w_i >= 0, sum w_i = 1)
  - 4-Zone Deterministic Zero-Fraud Manifold Calibration (ZALL, 16,730 accounts)
  - Exports test submission: sovereign_grand_simplex_83m_cv_submission.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from scipy.stats import rankdata
from scipy.special import logit

DATA_DIR = Path('pgs-s6e11/dataset')
SUB_DIR = Path('pgs-s6e11/submission')
OOF_DIR = Path('pgs-s6e11/oof_preds')
OOF_DIR.mkdir(parents=True, exist_ok=True)
SUB_DIR.mkdir(parents=True, exist_ok=True)

U = lambda v: (rankdata(v) - 0.5) / len(v)
L = lambda v: logit(np.clip(U(v), 1e-6, 1 - 1e-6))

def apply_rank_preserving_manifold(preds, mask_zero):
    n_total = len(preds)
    n_zero = mask_zero.sum()
    ranks = np.zeros(n_total, dtype=np.float64)
    ranks[mask_zero] = (rankdata(preds[mask_zero]) - 0.5) / n_total
    ranks[~mask_zero] = (n_zero + rankdata(preds[~mask_zero]) - 0.5) / n_total
    return ranks

def collect_streams():
    candidates = []
    
    # 1. OOF bank (70 models)
    bank = Path('pgs-s6e11/external_models/oof_bank')
    if bank.exists():
        for p in sorted(bank.glob('oof_*.npy')):
            name = p.name[4:-4]
            t = bank / f'test_{name}.npy'
            if t.exists():
                candidates.append((f'bank:{name}', p, t))

    # 2. Weni BPE pack (7 models)
    weni = Path('pgs-s6e11/external_models/weni_bpe_pack')
    if weni.exists():
        for p in sorted(weni.glob('oof_*.npy')):
            name = p.name[4:-4]
            t = weni / f'test_{name}.npy'
            if t.exists():
                candidates.append((f'weni:{name}', p, t))

    # 3. TabM-PWL v2
    tabm = Path('pgs-s6e11/code/tabm_gpu_v2/output')
    if (tabm / 'oof_tabm_pwl_v2.npy').exists() and (tabm / 'test_tabm_pwl_v2.npy').exists():
        candidates.append(('tabm:pwl_v2', tabm / 'oof_tabm_pwl_v2.npy', tabm / 'test_tabm_pwl_v2.npy'))

    # 4. GPT-2 Seed 2026
    gpt2 = Path('pgs-s6e11/external_models/gpt2_seed2026')
    if (gpt2 / 'gpt2_s2026_oof_5fold.npy').exists() and (gpt2 / 'gpt2_s2026_test_5fold.npy').exists():
        candidates.append(('lm:gpt2_s2026', gpt2 / 'gpt2_s2026_oof_5fold.npy', gpt2 / 'gpt2_s2026_test_5fold.npy'))

    # 5. Dual Neural Multiseed (3 models)
    dn = Path('pgs-s6e11/code/dual_neural_multiseed/output')
    if dn.exists():
        for p in sorted(dn.glob('oof_*.npy')):
            name = p.name[4:-4]
            t = dn / f'test_{name}.npy'
            if t.exists():
                candidates.append((f'dual_nn:{name}', p, t))

    # 6. Quad GLM Boost
    quad = Path('pgs-s6e11/code/quad_glm_boost/output')
    if (quad / 'oof_quad_glm_boost.npy').exists() and (quad / 'test_quad_glm_boost.npy').exists():
        candidates.append(('quad_glm:boost', quad / 'oof_quad_glm_boost.npy', quad / 'test_quad_glm_boost.npy'))

    return candidates

def main():
    print("=" * 80)
    print("🏆 Grand 83-Model Non-Negative Simplex Meta-Stacker with Nested 5-Fold CV")
    print("=" * 80)

    # 1. Load train labels and compute 4-zone zero-fraud manifold
    train_df = pd.read_csv(DATA_DIR / 'train.csv', usecols=[
        'is_fraud', 'velocity_score', 'merchant_risk_score',
        'cvv_retry_count', 'prior_disputes', 'is_new_merchant',
        'is_foreign_transaction', 'used_vpn', 'ip_country_mismatch',
        'billing_shipping_mismatch', 'is_ai_generated_scam_attempt'
    ])
    y = train_df['is_fraud'].values
    n_train = len(y)

    test_df = pd.read_csv(DATA_DIR / 'test.csv', usecols=[
        'velocity_score', 'merchant_risk_score',
        'cvv_retry_count', 'prior_disputes', 'is_new_merchant',
        'is_foreign_transaction', 'used_vpn', 'ip_country_mismatch',
        'billing_shipping_mismatch', 'is_ai_generated_scam_attempt'
    ])
    sample_sub = pd.read_csv(DATA_DIR / 'sample_submission.csv')
    n_test = len(sample_sub)

    # Compute 4-zone masks
    def get_zall(df):
        clean = ((df['cvv_retry_count'] == 0) & (df['prior_disputes'] == 0) & 
                 (~df['is_new_merchant'].astype(bool)) & (~df['is_foreign_transaction'].astype(bool)) & 
                 (~df['used_vpn'].astype(bool)) & (~df['ip_country_mismatch'].astype(bool)) & 
                 (~df['billing_shipping_mismatch'].astype(bool)) & (~df['is_ai_generated_scam_attempt'].astype(bool))).to_numpy()
        zone1 = df['merchant_risk_score'].to_numpy() < 13.1
        zone2 = (df['velocity_score'].to_numpy() <= 4.1) & (df['merchant_risk_score'].to_numpy() <= 20.0)
        zone34 = clean & (((df['velocity_score'].to_numpy() <= 4.1) & (df['merchant_risk_score'].to_numpy() <= 33.8)) |
                          ((df['velocity_score'].to_numpy() <= 6.0) & (df['merchant_risk_score'].to_numpy() <= 25.0)))
        return zone1 | zone2 | zone34

    zall_tr = get_zall(train_df)
    zall_te = get_zall(test_df)
    print(f"4-Zone Manifold: {zall_tr.sum():,} train rows (frauds: {y[zall_tr].sum()}), {zall_te.sum():,} test accounts ({zall_te.mean():.2%})")

    # 2. Collect and load all matching streams
    raw_streams = collect_streams()
    print(f"\nDiscovered {len(raw_streams)} streams. Verifying arrays...")

    names, oof_list, test_list = [], [], []
    for tag, oof_p, test_p in raw_streams:
        o = np.load(oof_p)
        t = np.load(test_p)
        if len(o) == n_train and len(t) == n_test and not np.isnan(o).any() and not np.isnan(t).any():
            names.append(tag)
            oof_list.append(o)
            test_list.append(t)

    K = len(names)
    print(f"Loaded {K} valid, complete model streams.")

    # Convert all OOFs and Tests into Logit-Rank Space
    print("\nTransforming all models to Logit-Rank representation...")
    OOF_logit = np.column_stack([L(o) for o in oof_list])
    TEST_logit = np.column_stack([L(t) for t in test_list])

    # 3. Stratified 5-Fold Nested Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    nested_preds = np.zeros(n_train, dtype=np.float64)
    fold_aucs = []
    fold_praucs = []
    fold_weights = []

    print("\n--- Fitting Non-Negative Simplex Stacker across 5 Folds ---")
    for f, (trn_idx, val_idx) in enumerate(skf.split(OOF_logit, y)):
        X_tr, y_tr = OOF_logit[trn_idx], y[trn_idx]
        X_va, y_va = OOF_logit[val_idx], y[val_idx]

        # Fit Non-Negative Least Squares / Non-Negative Linear Regression
        nnls = LinearRegression(positive=True, fit_intercept=False)
        nnls.fit(X_tr, y_tr)
        w_raw = nnls.coef_

        if w_raw.sum() > 0:
            w_norm = w_raw / w_raw.sum()
        else:
            w_norm = np.ones(K) / K

        fold_weights.append(w_norm)

        # Predict on validation fold
        va_raw = X_va @ w_norm
        va_calibrated = apply_rank_preserving_manifold(va_raw, zall_tr[val_idx])
        nested_preds[val_idx] = va_calibrated

        f_auc = roc_auc_score(y_va, va_calibrated)
        f_prauc = average_precision_score(y_va, va_calibrated)
        fold_aucs.append(f_auc)
        fold_praucs.append(f_prauc)

        n_active = (w_norm > 0.001).sum()
        print(f"Fold {f}: ROC-AUC = {f_auc:.6f} | PR-AUC = {f_prauc:.6f} | Active Models: {n_active}/{K}")

    overall_auc = roc_auc_score(y, nested_preds)
    overall_prauc = average_precision_score(y, nested_preds)
    mean_auc = np.mean(fold_aucs)
    mean_prauc = np.mean(fold_praucs)

    print("\n" + "=" * 80)
    print("🏆 GRAND 83-MODEL NESTED SIMPLEX STACK RESULTS:")
    print(f"  Overall OOF ROC-AUC : {overall_auc:.6f}")
    print(f"  Mean Fold ROC-AUC   : {mean_auc:.6f}")
    print(f"  Overall OOF PR-AUC  : {overall_prauc:.6f}")
    print(f"  Mean Fold PR-AUC    : {mean_prauc:.6f}")
    print("=" * 80)

    # Analyze Mean Weights across Folds
    mean_w = np.mean(fold_weights, axis=0)
    w_df = pd.DataFrame({'model': names, 'mean_weight': mean_w}).sort_values('mean_weight', ascending=False)
    active_df = w_df[w_df['mean_weight'] > 0.001].copy()
    active_df['pct'] = active_df['mean_weight'] / active_df['mean_weight'].sum() * 100

    print(f"\nTop Active Models in Convex Simplex ({len(active_df)} out of {K}):")
    print(active_df.head(20).to_string(index=False))

    # Family grouping
    families = {'Trees': 0.0, 'Neural (TabM/RealMLP/ResNet)': 0.0, 'Language Models': 0.0, 'EBM': 0.0, 'GLM': 0.0}
    for _, row in active_df.iterrows():
        m = row['model']
        w = row['mean_weight']
        if 'lm:' in m or 'gpt2' in m:
            families['Language Models'] += w
        elif 'tabm' in m or 'realmlp' in m or 'resnet' in m or 'transformer' in m or 'dual_nn' in m:
            families['Neural (TabM/RealMLP/ResNet)'] += w
        elif 'ebm' in m:
            families['EBM'] += w
        elif 'glm' in m:
            families['GLM'] += w
        else:
            families['Trees'] += w

    print("\nFamily Weight Distribution:")
    for fam, w in families.items():
        print(f"  - {fam:32s}: {w*100:.2f}%")

    # Save OOF Predictions
    np.save(OOF_DIR / 'oof_grand_simplex_83m.npy', nested_preds)
    print(f"\n✅ Saved nested OOF predictions to {OOF_DIR / 'oof_grand_simplex_83m.npy'}")

    # Generate Test Submission
    print("\nGenerating final stacked test predictions...")
    test_raw = TEST_logit @ mean_w
    test_calibrated = apply_rank_preserving_manifold(test_raw, zall_te)

    sub_path = SUB_DIR / 'sovereign_grand_simplex_83m_cv_submission.csv'
    sample_sub[['id']].assign(is_fraud=test_calibrated).to_csv(sub_path, index=False)
    print(f"✅ Saved test submission to {sub_path} ({len(sample_sub):,} rows)")

if __name__ == '__main__':
    main()
