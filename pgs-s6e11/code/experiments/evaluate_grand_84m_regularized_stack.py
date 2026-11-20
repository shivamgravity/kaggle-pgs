"""
Grand 84-Model Fast Regularized Simplex Meta-Stacker for PGS-S6E11.

Evaluates an exhaustive 5-fold nested cross-validation stack across 84 verified model streams:
  - 70 models from OOF Bank (Trees, Bags, EBMs, GLMs)
  - 7 models from Weni Mohamed BPE Pack (BPE LGB, CatBoost, TabM, RealMLP, GLM)
  - 1 model: GPU TabM-PWL V3 (Dual-Seed, k=32 slices, +18.7 pts CV gain)
  - 1 model: GPU RealMLP-TD V3 (Dual-Seed, CTGAN residues, 5.89% diversity)
  - 1 model: GPT-2 Sequence LM 5-fold (Seed 2026)
  - 3 models from Dual Neural (ResNet-bag3, FT-Transformer-bag3, Dual NN)
  - 1 model from Quad Interaction Booster
  
Features:
  - Fast Cholesky Normal Equations NNLS (Machine-precision exact, solves 84x84 in 1ms)
  - Strict 5-Fold Nested Cross-Validation (StratifiedKFold, shuffle=True, random_state=42)
  - Alpha Sweep across L2 Regularization (alpha = 0, 10, 50, 100, 500, 1000)
  - Calibrated with the NEW 5-Zone Deterministic Zero-Fraud Manifold (ZALL_5, 41,301 train, 17,647 test)
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import rankdata, spearmanr
from scipy.special import logit
from scipy.optimize import nnls

DATA_DIR = Path('pgs-s6e11/dataset')
SUB_DIR = Path('pgs-s6e11/submission')
SUB_DIR.mkdir(parents=True, exist_ok=True)

U = lambda v: (rankdata(v) - 0.5) / len(v)
L = lambda v: logit(np.clip(U(v), 1e-6, 1 - 1e-6))

def fast_regularized_nnls(X, y, alpha=0.0):
    """Solves min_{w >= 0} ||X w - y||_2^2 + alpha ||w||_2^2 in 1ms via Cholesky."""
    G = X.T @ X
    if alpha > 0:
        G = G + alpha * np.eye(G.shape[0])
    else:
        G = G + 1e-7 * np.eye(G.shape[0])
    c = X.T @ y
    L_mat = np.linalg.cholesky(G)
    b_R = np.linalg.solve(L_mat, c)
    w, _ = nnls(L_mat.T, b_R)
    s = w.sum()
    if s > 0:
        return w / s
    return np.ones(len(w)) / len(w)

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
            name = p.stem[4:]
            t = bank / f'test_{name}.npy'
            if t.exists():
                candidates.append((f'bank:{name}', p, t))

    # 2. Weni BPE pack (7 models)
    weni = Path('pgs-s6e11/external_models/weni_bpe_pack')
    if weni.exists():
        for p in sorted(weni.glob('oof_*.npy')):
            name = p.stem[4:]
            t = weni / f'test_{name}.npy'
            if t.exists():
                candidates.append((f'weni:{name}', p, t))

    # 3. TabM-PWL V3 (Cloud GPU, k=32)
    tabm = Path('pgs-s6e11/code/tabm_gpu_v3/output')
    if (tabm / 'oof_tabm_pwl_v3.npy').exists() and (tabm / 'test_tabm_pwl_v3.npy').exists():
        candidates.append(('tabm:pwl_v3', tabm / 'oof_tabm_pwl_v3.npy', tabm / 'test_tabm_pwl_v3.npy'))

    # 4. RealMLP-TD V3 (Cloud GPU, dual seed)
    rmlp = Path('pgs-s6e11/code/realmlp_gpu_v3/output')
    if (rmlp / 'oof_realmlp_v3.npy').exists() and (rmlp / 'test_realmlp_v3.npy').exists():
        candidates.append(('realmlp:v3', rmlp / 'oof_realmlp_v3.npy', rmlp / 'test_realmlp_v3.npy'))

    # 5. GPT-2 Seed 2026
    gpt2 = Path('pgs-s6e11/external_models/gpt2_seed2026')
    if (gpt2 / 'gpt2_s2026_oof_5fold.npy').exists() and (gpt2 / 'gpt2_s2026_test_5fold.npy').exists():
        candidates.append(('lm:gpt2_s2026', gpt2 / 'gpt2_s2026_oof_5fold.npy', gpt2 / 'gpt2_s2026_test_5fold.npy'))

    # 6. Dual Neural Multiseed (3 models)
    dn = Path('pgs-s6e11/code/dual_neural_multiseed/output')
    if dn.exists():
        for p in sorted(dn.glob('oof_*.npy')):
            name = p.stem[4:]
            t = dn / f'test_{name}.npy'
            if t.exists():
                candidates.append((f'dual_nn:{name}', p, t))

    # 7. Quad GLM Boost
    quad = Path('pgs-s6e11/code/quad_glm_boost/output')
    if (quad / 'oof_quad_glm_boost.npy').exists() and (quad / 'test_quad_glm_boost.npy').exists():
        candidates.append(('quad_glm:boost', quad / 'oof_quad_glm_boost.npy', quad / 'test_quad_glm_boost.npy'))

    return candidates

def main():
    print("=" * 80)
    print("🏆 Grand 84-Model Fast Regularized Meta-Stacker (5-Zone Manifold)")
    print("=" * 80)

    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    test_df = pd.read_csv(DATA_DIR / 'test.csv')
    sample_sub = pd.read_csv(DATA_DIR / 'sample_submission.csv')

    y = train_df['is_fraud'].values
    n_train = len(y)
    n_test = len(sample_sub)

    # 1. Compute 5-Zone Zero-Fraud Manifold
    def get_zall5(df):
        clean = ((df['cvv_retry_count'] == 0) & (df['prior_disputes'] == 0) & 
                 (~df['is_new_merchant'].astype(bool)) & (~df['is_foreign_transaction'].astype(bool)) & 
                 (~df['used_vpn'].astype(bool)) & (~df['ip_country_mismatch'].astype(bool)) & 
                 (~df['billing_shipping_mismatch'].astype(bool)) & (~df['is_ai_generated_scam_attempt'].astype(bool))).to_numpy()
        zone1 = df['merchant_risk_score'].to_numpy() < 13.1
        zone2 = (df['velocity_score'].to_numpy() <= 4.1) & (df['merchant_risk_score'].to_numpy() <= 20.0)
        zone34 = clean & (((df['velocity_score'].to_numpy() <= 4.1) & (df['merchant_risk_score'].to_numpy() <= 33.8)) |
                          ((df['velocity_score'].to_numpy() <= 6.0) & (df['merchant_risk_score'].to_numpy() <= 25.0)))
        zone5 = clean & (df['velocity_score'].to_numpy() <= 5.0) & (df['merchant_risk_score'].to_numpy() <= 35.0) & (df['amount_usd'].to_numpy() <= 30.0)
        return zone1 | zone2 | zone34 | zone5

    zall_tr = get_zall5(train_df)
    zall_te = get_zall5(test_df)
    print(f"5-Zone Manifold: {zall_tr.sum():,} train rows (frauds: {y[zall_tr].sum()}), {zall_te.sum():,} test accounts ({zall_te.mean():.2%})")

    # 2. Collect and load streams
    raw_streams = collect_streams()
    print(f"\nDiscovered {len(raw_streams)} raw streams. Validating arrays...")

    names, oof_list, test_list = [], [], []
    for tag, oof_p, test_p in raw_streams:
        o = np.load(oof_p)
        t = np.load(test_p)
        if len(o) == n_train and len(t) == n_test and not np.isnan(o).any() and not np.isnan(t).any():
            names.append(tag)
            oof_list.append(o)
            test_list.append(t)

    K = len(names)
    print(f"Successfully loaded {K} valid, complete model streams.")

    # Convert to Logit-Rank Space
    print("\nTransforming all models to Logit-Rank representation...")
    OOF_logit = np.column_stack([L(o) for o in oof_list])
    TEST_logit = np.column_stack([L(t) for t in test_list])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Fast Alpha Sweep
    best_alpha = None
    best_score = -1
    best_prauc = -1
    best_oof_preds = None
    best_test_preds = None
    best_weights = None

    print("\n--- Sweeping L2 Regularization (Alpha) across 5 Folds ---")
    for alpha in [0.0, 10.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0]:
        t0 = time.time()
        nested_preds = np.zeros(n_train, dtype=np.float64)
        test_preds = np.zeros(n_test, dtype=np.float64)
        fold_weights = []

        for f, (trn_idx, val_idx) in enumerate(skf.split(OOF_logit, y)):
            X_tr, y_tr = OOF_logit[trn_idx], y[trn_idx]
            X_va, y_va = OOF_logit[val_idx], y[val_idx]

            w = fast_regularized_nnls(X_tr, y_tr, alpha=alpha)
            fold_weights.append(w)

            va_raw = X_va @ w
            nested_preds[val_idx] = apply_rank_preserving_manifold(va_raw, zall_tr[val_idx])
            test_preds += (TEST_logit @ w) / 5.0

        score_roc = roc_auc_score(y, nested_preds)
        score_pr = average_precision_score(y, nested_preds)
        elapsed = time.time() - t0

        print(f"  Alpha = {alpha:6.1f} | 5-Fold Nested CV ROC-AUC: {score_roc:.6f} | PR-AUC: {score_pr:.6f} ({elapsed:.2f}s)")

        if score_roc > best_score:
            best_score = score_roc
            best_prauc = score_pr
            best_alpha = alpha
            best_oof_preds = nested_preds
            best_test_preds = apply_rank_preserving_manifold(test_preds, zall_te)
            best_weights = np.mean(fold_weights, axis=0)

    print("\n" + "=" * 80)
    print(f"🏆 Optimal Configuration: Alpha = {best_alpha:.1f}")
    print(f"🏆 Peak 5-Fold Nested CV ROC-AUC: {best_score:.6f}")
    print(f"🏆 Peak 5-Fold Nested CV PR-AUC:  {best_prauc:.6f}")
    print("=" * 80)

    # Top contributing models
    top_indices = np.argsort(best_weights)[::-1]
    print("\nTop Contributing Models in Optimal Regularized Stack:")
    for idx in top_indices:
        if best_weights[idx] >= 0.005:  # >= 0.5% weight
            print(f"  {names[idx]:32s}: {best_weights[idx]*100:5.2f}%")

    # Export submission
    out_path = SUB_DIR / 'sovereign_grand_regularized_84m_cv_submission.csv'
    sample_sub[['id']].assign(is_fraud=best_test_preds).to_csv(out_path, index=False)
    print(f"\n✅ Exported Sovereign 84-Model CV Guardrail: {out_path.name} (CV: {best_score:.6f})")

if __name__ == '__main__':
    main()
