"""
Ridge and ElasticNet Regularized Meta-Stacker across 83 models.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import rankdata
from scipy.special import logit

DATA_DIR = Path('pgs-s6e11/dataset')
SUB_DIR = Path('pgs-s6e11/submission')
OOF_DIR = Path('pgs-s6e11/oof_preds')

U = lambda v: (rankdata(v) - 0.5) / len(v)
L = lambda v: logit(np.clip(U(v), 1e-6, 1 - 1e-6))

def apply_rank_preserving_manifold(preds, mask_zero):
    n_total = len(preds)
    n_zero = mask_zero.sum()
    ranks = np.zeros(n_total, dtype=np.float64)
    ranks[mask_zero] = (rankdata(preds[mask_zero]) - 0.5) / n_total
    ranks[~mask_zero] = (n_zero + rankdata(preds[~mask_zero]) - 0.5) / n_total
    return ranks

def main():
    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    y = train_df['is_fraud'].values

    clean = ((train_df['cvv_retry_count'] == 0) & (train_df['prior_disputes'] == 0) & 
             (~train_df['is_new_merchant'].astype(bool)) & (~train_df['is_foreign_transaction'].astype(bool)) & 
             (~train_df['used_vpn'].astype(bool)) & (~train_df['ip_country_mismatch'].astype(bool)) & 
             (~train_df['billing_shipping_mismatch'].astype(bool)) & (~train_df['is_ai_generated_scam_attempt'].astype(bool))).to_numpy()
    zone1 = train_df['merchant_risk_score'].to_numpy() < 13.1
    zone2 = (train_df['velocity_score'].to_numpy() <= 4.1) & (train_df['merchant_risk_score'].to_numpy() <= 20.0)
    zone34 = clean & (((train_df['velocity_score'].to_numpy() <= 4.1) & (train_df['merchant_risk_score'].to_numpy() <= 33.8)) |
                      ((train_df['velocity_score'].to_numpy() <= 6.0) & (train_df['merchant_risk_score'].to_numpy() <= 25.0)))
    zall = zone1 | zone2 | zone34

    from evaluate_grand_83m_simplex_stack import collect_streams
    streams = collect_streams()

    names, oofs, tests = [], [], []
    for tag, op, tp in streams:
        o = np.load(op)
        t = np.load(tp)
        if len(o) == len(y) and len(t) == 299984 and not np.isnan(o).any() and not np.isnan(t).any():
            names.append(tag)
            oofs.append(o)
            tests.append(t)

    K = len(names)
    print(f"Loaded {K} streams. Building logit matrix...")
    X = np.column_stack([L(o) for o in oofs])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for alpha in [500.0, 2000.0, 5000.0, 10000.0, 25000.0, 50000.0, 100000.0]:
        preds = np.zeros(len(y), dtype=np.float64)
        fold_aucs = []
        for trn, val in skf.split(X, y):
            model = Ridge(alpha=alpha, positive=True, fit_intercept=False)
            model.fit(X[trn], y[trn])
            coef = model.coef_
            if coef.sum() > 0:
                w = coef / coef.sum()
            else:
                w = np.ones(K) / K
            raw = X[val] @ w
            cal = apply_rank_preserving_manifold(raw, zall[val])
            preds[val] = cal
            fold_aucs.append(roc_auc_score(y[val], cal))

        overall_auc = roc_auc_score(y, preds)
        overall_prauc = average_precision_score(y, preds)
        print(f"Ridge alpha={alpha:8.1f} | Mean Fold AUC: {np.mean(fold_aucs):.6f} | Overall AUC: {overall_auc:.6f} | PR-AUC: {overall_prauc:.6f}")

if __name__ == '__main__':
    main()
