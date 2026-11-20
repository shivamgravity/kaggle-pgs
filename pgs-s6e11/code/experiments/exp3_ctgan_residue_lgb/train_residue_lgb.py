"""
Phase 2 Experiment 2.1: CTGAN Decimal Residues & Multi-Scale Target Encoding LightGBM.

Tests the hypothesis:
  In synthetic CTGAN data, Gaussian mixture mode normalization creates subtle
  floating-point residues, quantization seams, and decimal ending artifacts.
  Extracting these residues alongside multi-scale smoothed target encodings (m=10, 50, 200)
  improves individual tree CV beyond the 0.9515 collinearity ceiling.
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import rankdata
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('pgs-s6e11/dataset')
OUT_DIR = Path('pgs-s6e11/code/experiments/exp3_ctgan_residue_lgb/output')
OUT_DIR.mkdir(parents=True, exist_ok=True)

U = lambda v: (rankdata(v) - 0.5) / len(v)

def compute_zall(df):
    clean = ((df['cvv_retry_count'] == 0) & (df['prior_disputes'] == 0) & 
             (~df['is_new_merchant'].astype(bool)) & (~df['is_foreign_transaction'].astype(bool)) & 
             (~df['used_vpn'].astype(bool)) & (~df['ip_country_mismatch'].astype(bool)) & 
             (~df['billing_shipping_mismatch'].astype(bool)) & (~df['is_ai_generated_scam_attempt'].astype(bool))).to_numpy()
    zone1 = df['merchant_risk_score'].to_numpy() < 13.1
    zone2 = (df['velocity_score'].to_numpy() <= 4.1) & (df['merchant_risk_score'].to_numpy() <= 20.0)
    zone34 = clean & (((df['velocity_score'].to_numpy() <= 4.1) & (df['merchant_risk_score'].to_numpy() <= 33.8)) |
                      ((df['velocity_score'].to_numpy() <= 6.0) & (df['merchant_risk_score'].to_numpy() <= 25.0)))
    return zone1 | zone2 | zone34

def apply_rank_preserving_manifold(preds, mask_zero):
    n_total = len(preds)
    n_zero = mask_zero.sum()
    ranks = np.zeros(n_total, dtype=np.float64)
    ranks[mask_zero] = (rankdata(preds[mask_zero]) - 0.5) / n_total
    ranks[~mask_zero] = (n_zero + rankdata(preds[~mask_zero]) - 0.5) / n_total
    return ranks

def main():
    print("=" * 80)
    print("🔬 Phase 2 Experiment 2.1: CTGAN Decimal Residues & Multi-Scale Target Encoding")
    print("=" * 80)

    t0 = time.time()
    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    test_df = pd.read_csv(DATA_DIR / 'test.csv')
    sample_sub = pd.read_csv(DATA_DIR / 'sample_submission.csv')

    y = train_df['is_fraud'].values
    n_tr = len(train_df)
    n_te = len(test_df)
    print(f"Loaded {n_tr:,} train and {n_te:,} test rows in {time.time() - t0:.2f}s.")

    zall_tr = compute_zall(train_df)
    zall_te = compute_zall(test_df)
    print(f"4-Zone Zero-Fraud: {zall_tr.sum():,} train ({zall_tr.mean():.2%}), {zall_te.sum():,} test ({zall_te.mean():.2%})")

    # Combine temporarily for consistent categorical & feature construction
    dfs = [train_df, test_df]
    for df in dfs:
        # 1. CTGAN Decimal & Residue Features
        # Amount residues
        amt = df['amount_usd']
        df['amt_cents'] = np.round((amt % 1) * 100).astype(int).clip(0, 99)
        df['amt_tenths'] = np.floor((amt % 1) * 10).astype(int).clip(0, 9)
        df['amt_last_digit'] = (np.floor(amt).astype(int) % 10).clip(0, 9)
        df['amt_mod10'] = (amt % 10.0).astype(np.float32)
        df['amt_mod50'] = (amt % 50.0).astype(np.float32)
        df['is_round_dollar'] = (amt % 1.0 == 0).astype(np.int8)
        df['is_99_cents'] = (df['amt_cents'] == 99).astype(np.int8)
        df['is_quarter_cent'] = (df['amt_cents'] % 25 == 0).astype(np.int8)

        # Velocity score residues
        vel = df['velocity_score']
        df['vel_residue_0p25'] = (vel % 0.25).astype(np.float32)
        df['vel_residue_0p1'] = (vel % 0.1).astype(np.float32)
        df['vel_is_integer'] = (vel % 1.0 == 0).astype(np.int8)

        # Merchant risk score quantization
        mrs = df['merchant_risk_score']
        df['mrs_decile'] = (mrs // 10).astype(int).clip(0, 10)
        df['mrs_residue_1p0'] = (mrs % 1.0).astype(np.float32)

        # Distance & Time residues
        df['dist_mod10'] = (df['distance_from_home_km'] % 10.0).astype(np.float32)
        df['hours_mod24'] = (df['hours_since_last_txn'] % 24.0).astype(np.float32)
        df['hours_mod6'] = (df['hours_since_last_txn'] % 6.0).astype(np.float32)

        # Multi-way Discrete Interaction Keys
        df['cat_x_auth'] = df['merchant_category'].astype(str) + '_' + df['auth_method'].astype(str)
        df['cat_x_device'] = df['merchant_category'].astype(str) + '_' + df['device_type'].astype(str)
        df['sec_triplet'] = (df['is_foreign_transaction'].astype(str) + '_' + 
                             df['used_vpn'].astype(str) + '_' + 
                             df['ip_country_mismatch'].astype(str))
        df['cents_x_cat'] = df['amt_cents'].astype(str) + '_' + df['merchant_category'].astype(str)

    # Base numerical features
    base_num = [
        'amount_usd', 'hours_since_last_txn', 'txn_count_last_24h',
        'distance_from_home_km', 'card_age_months', 'customer_age',
        'account_balance_usd', 'cvv_retry_count', 'velocity_score',
        'merchant_risk_score', 'prior_disputes'
    ]

    residue_num = [
        'amt_cents', 'amt_tenths', 'amt_last_digit', 'amt_mod10', 'amt_mod50',
        'is_round_dollar', 'is_99_cents', 'is_quarter_cent',
        'vel_residue_0p25', 'vel_residue_0p1', 'vel_is_integer',
        'mrs_decile', 'mrs_residue_1p0',
        'dist_mod10', 'hours_mod24', 'hours_mod6'
    ]

    base_cat = [
        'merchant_category', 'card_type', 'auth_method', 'channel', 'device_type',
        'time_of_day_hour', 'day_of_week', 'is_new_merchant',
        'is_foreign_transaction', 'used_vpn', 'ip_country_mismatch',
        'billing_shipping_mismatch', 'is_ai_generated_scam_attempt'
    ]

    inter_cat = ['cat_x_auth', 'cat_x_device', 'sec_triplet', 'cents_x_cat']

    target_encode_cols = ['merchant_category', 'auth_method', 'channel', 'device_type',
                          'cat_x_auth', 'cat_x_device', 'sec_triplet', 'cents_x_cat']

    print(f"Features: {len(base_num)} base num, {len(residue_num)} residue num, {len(base_cat)} base cat, {len(inter_cat)} inter cat.")

    # Convert categoricals to category dtype for LightGBM
    all_cats = base_cat + inter_cat
    for c in all_cats:
        train_df[c] = train_df[c].astype('category')
        test_df[c] = test_df[c].astype('category')

    feature_cols = base_num + residue_num + all_cats

    # 5-Fold Stratified Cross Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(n_tr, dtype=np.float64)
    test_preds = np.zeros(n_te, dtype=np.float64)

    # Multi-scale target encodings (m=10, 50, 200)
    global_prior = y.mean()

    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'max_depth': 5,
        'learning_rate': 0.05,
        'min_child_samples': 40,
        'subsample': 0.8,
        'colsample_bytree': 0.75,
        'random_state': 42,
        'n_estimators': 1200,
        'verbose': -1,
        'n_jobs': 4
    }

    fold_aucs = []
    fold_praucs = []

    print("\n--- Training 5-Fold Stratified LightGBM with CTGAN Residues ---")
    for fold, (trn_idx, val_idx) in enumerate(skf.split(train_df, y)):
        f_t0 = time.time()
        X_tr = train_df.iloc[trn_idx][feature_cols].copy()
        y_tr = y[trn_idx]
        X_va = train_df.iloc[val_idx][feature_cols].copy()
        y_va = y[val_idx]
        X_te = test_df[feature_cols].copy()

        # In-fold multi-scale target encodings
        for col in target_encode_cols:
            stats = train_df.iloc[trn_idx].groupby(col, observed=True)['is_fraud'].agg(['count', 'mean'])
            for m in [10, 50, 200]:
                smoothed = (stats['count'] * stats['mean'] + m * global_prior) / (stats['count'] + m)
                te_map = smoothed.to_dict()
                X_tr[f'{col}_te_m{m}'] = X_tr[col].map(te_map).fillna(global_prior).astype(np.float32)
                X_va[f'{col}_te_m{m}'] = X_va[col].map(te_map).fillna(global_prior).astype(np.float32)
                X_te[f'{col}_te_m{m}'] = X_te[col].map(te_map).fillna(global_prior).astype(np.float32)

        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )

        val_pred = model.predict_proba(X_va)[:, 1]
        tst_pred = model.predict_proba(X_te)[:, 1]

        oof_preds[val_idx] = val_pred
        test_preds += tst_pred / 5.0

        f_auc = roc_auc_score(y_va, val_pred)
        f_prauc = average_precision_score(y_va, val_pred)
        fold_aucs.append(f_auc)
        fold_praucs.append(f_prauc)
        print(f"Fold {fold} finished in {time.time() - f_t0:.1f}s | Best Iter: {model.best_iteration_} | ROC-AUC: {f_auc:.6f} | PR-AUC: {f_prauc:.6f}")

    # Calibration with 4-zone manifold
    oof_cal = apply_rank_preserving_manifold(oof_preds, zall_tr)
    test_cal = apply_rank_preserving_manifold(test_preds, zall_te)

    overall_raw_auc = roc_auc_score(y, oof_preds)
    overall_cal_auc = roc_auc_score(y, oof_cal)
    overall_cal_prauc = average_precision_score(y, oof_cal)

    print("\n" + "=" * 80)
    print("🏆 EXPERIMENT 2.1 RESULTS (CTGAN Residues + Multi-Scale TE LightGBM):")
    print(f"  Raw OOF ROC-AUC        : {overall_raw_auc:.6f} (Mean Fold: {np.mean(fold_aucs):.6f})")
    print(f"  4-Zone Calibrated AUC  : {overall_cal_auc:.6f} (Gain: +{(overall_cal_auc - overall_raw_auc)*1e5:.1f} pts)")
    print(f"  4-Zone Calibrated PR-AUC: {overall_cal_prauc:.6f}")
    print("=" * 80)

    # Save arrays
    np.save(OUT_DIR / 'oof_ctgan_residue_lgb.npy', oof_preds.astype(np.float32))
    np.save(OUT_DIR / 'test_ctgan_residue_lgb.npy', test_preds.astype(np.float32))
    np.save(OUT_DIR / 'oof_ctgan_residue_lgb_calibrated.npy', oof_cal.astype(np.float32))
    np.save(OUT_DIR / 'test_ctgan_residue_lgb_calibrated.npy', test_cal.astype(np.float32))

    # Export test submission
    sub = sample_sub[['id']].assign(is_fraud=test_cal)
    sub.to_csv(OUT_DIR / 'ctgan_residue_lgb_submission.csv', index=False)
    print(f"Saved artifacts to {OUT_DIR}/")

if __name__ == '__main__':
    main()
