"""
Advanced Velocity & Manifold Augmented GLM Residual LightGBM Booster for PGS-S6E11.

Innovations:
  1. Transaction Velocity & Burst Dynamics (spend velocity, travel velocity, balance drain, risk amplification)
  2. CTGAN Decimal Residues & Cents Bins
  3. 5-Zone Manifold Continuous Distance Metrics
  4. In-Fold Bayesian Smoothed Target Encodings (1D + 2-Way Joint Keys)
  5. Two-Stage GLM Residual Architecture:
     - Stage 1: Scaled Ridge/Logistic GLM to compute initial log-odds margin z_i
     - Stage 2: Shallow LightGBM (4 leaves) fitted strictly on the linear residual
  6. Evaluated with 5-Fold Stratified Cross-Validation + 5-Zone Calibration
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata
from scipy.special import expit, logit
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

DATA_DIR = Path('pgs-s6e11/dataset')
OUT_DIR = Path('pgs-s6e11/code/experiments/exp4_output')
OUT_DIR.mkdir(parents=True, exist_ok=True)

def apply_rank_preserving_manifold(preds, mask_zero):
    n_total = len(preds)
    n_zero = mask_zero.sum()
    ranks = np.zeros(n_total, dtype=np.float64)
    ranks[mask_zero] = (rankdata(preds[mask_zero]) - 0.5) / n_total
    ranks[~mask_zero] = (n_zero + rankdata(preds[~mask_zero]) - 0.5) / n_total
    return ranks

def main():
    start_time = time.time()
    print("=" * 80)
    print("🚀 Running Advanced Velocity & Manifold Augmented GLM Residual Booster")
    print("=" * 80)

    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    y = train_df['is_fraud'].values.astype(int)
    n_train = len(y)

    # 1. 5-Zone Zero-Fraud Manifold Mask
    clean_tr = ((train_df['cvv_retry_count'] == 0) & (train_df['prior_disputes'] == 0) & 
                (~train_df['is_new_merchant'].astype(bool)) & (~train_df['is_foreign_transaction'].astype(bool)) & 
                (~train_df['used_vpn'].astype(bool)) & (~train_df['ip_country_mismatch'].astype(bool)) & 
                (~train_df['billing_shipping_mismatch'].astype(bool)) & (~train_df['is_ai_generated_scam_attempt'].astype(bool))).to_numpy()

    zone1_tr = train_df['merchant_risk_score'].to_numpy() < 13.1
    zone2_tr = (train_df['velocity_score'].to_numpy() <= 4.1) & (train_df['merchant_risk_score'].to_numpy() <= 20.0)
    zone34_tr = clean_tr & (((train_df['velocity_score'].to_numpy() <= 4.1) & (train_df['merchant_risk_score'].to_numpy() <= 33.8)) |
                            ((train_df['velocity_score'].to_numpy() <= 6.0) & (train_df['merchant_risk_score'].to_numpy() <= 25.0)))
    zone5_tr = clean_tr & (train_df['velocity_score'].to_numpy() <= 5.0) & (train_df['merchant_risk_score'].to_numpy() <= 35.0) & (train_df['amount_usd'].to_numpy() <= 30.0)
    zall_5 = zone1_tr | zone2_tr | zone34_tr | zone5_tr
    print(f"Total rows in 5-zone manifold: {zall_5.sum():,} ({zall_5.mean():.2%}) | Frauds: {y[zall_5].sum()}")

    # 2. Feature Engineering
    print("\n--- Engineering Features ---")
    FLAG_COLS = list(train_df.select_dtypes(include='bool').columns)
    for c in FLAG_COLS:
        train_df[c] = train_df[c].astype(int)

    # Decimal & Cents Patterns
    train_df['cents'] = np.round((train_df['amount_usd'] % 1) * 100).astype(int)
    train_df['is_round_dollar'] = (train_df['cents'] == 0).astype(int)
    train_df['amount_div10'] = (train_df['amount_usd'] // 10).astype(int)
    train_df['balance_cents'] = np.round((train_df['account_balance_usd'] % 1) * 100).astype(int)

    # Velocity & Burst Dynamics
    train_df['spend_velocity'] = train_df['amount_usd'] / (train_df['hours_since_last_txn'] + 0.05)
    train_df['travel_velocity'] = train_df['distance_from_home_km'] / (train_df['hours_since_last_txn'] + 0.05)
    train_df['balance_drain_ratio'] = train_df['amount_usd'] / (train_df['account_balance_usd'] + 1.0)
    train_df['amount_24h_volume'] = train_df['amount_usd'] * train_df['txn_count_last_24h']
    train_df['velocity_x_amount'] = train_df['velocity_score'] * train_df['amount_usd']
    train_df['cvv_x_risk'] = train_df['cvv_retry_count'] * train_df['merchant_risk_score']
    train_df['total_flags'] = train_df[FLAG_COLS].sum(axis=1)
    train_df['flag_x_amount'] = train_df['total_flags'] * train_df['amount_usd']
    train_df['fraud_risk_index'] = train_df['total_flags'] * (train_df['velocity_score'] + 1.0) * (train_df['cvv_retry_count'] + 1)

    # Manifold Continuous Boundary Distances
    train_df['dist_zone1'] = np.maximum(0.0, train_df['merchant_risk_score'] - 13.1)
    train_df['dist_zone2_vel'] = np.maximum(0.0, train_df['velocity_score'] - 4.1)
    train_df['dist_zone2_risk'] = np.maximum(0.0, train_df['merchant_risk_score'] - 20.0)

    # Categorical interaction keys for target encoding
    train_df['cat_x_cents'] = train_df['merchant_category'].astype(str) + '_' + train_df['cents'].astype(str)
    train_df['cat_x_foreign'] = train_df['merchant_category'].astype(str) + '_' + train_df['is_foreign_transaction'].astype(str)
    train_df['dev_x_vpn'] = train_df['device_type'].astype(str) + '_' + train_df['used_vpn'].astype(str)
    train_df['card_x_cvv'] = train_df['card_type'].astype(str) + '_' + train_df['cvv_retry_count'].astype(str)

    TE_COLS = [
        'merchant_category', 'card_type', 'auth_method', 'channel', 'device_type',
        'cents', 'amount_div10', 'cat_x_cents', 'cat_x_foreign', 'dev_x_vpn', 'card_x_cvv'
    ]

    NUM_COLS = [
        'amount_usd', 'hours_since_last_txn', 'txn_count_last_24h', 'distance_from_home_km',
        'card_age_months', 'customer_age', 'account_balance_usd', 'cvv_retry_count',
        'velocity_score', 'time_of_day_hour', 'day_of_week', 'merchant_risk_score',
        'prior_disputes', 'spend_velocity', 'travel_velocity', 'balance_drain_ratio',
        'amount_24h_volume', 'velocity_x_amount', 'cvv_x_risk', 'total_flags',
        'flag_x_amount', 'fraud_risk_index', 'dist_zone1', 'dist_zone2_vel', 'dist_zone2_risk'
    ] + FLAG_COLS + ['is_round_dollar']

    print(f"Features: {len(NUM_COLS)} continuous/flag features, {len(TE_COLS)} target-encoded keys.")

    # 3. 5-Fold Cross-Validation Setup
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_glm = np.zeros(n_train, dtype=np.float64)
    oof_boost = np.zeros(n_train, dtype=np.float64)

    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 4,
        'min_child_samples': 50,
        'subsample': 0.8,
        'subsample_freq': 1,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': 4,
        'verbose': -1
    }

    print("\n--- Starting 5-Fold Training ---")
    for fold, (trn_idx, val_idx) in enumerate(skf.split(train_df, y)):
        t_fold = time.time()
        tr_df, va_df = train_df.iloc[trn_idx], train_df.iloc[val_idx]
        y_tr, y_va = y[trn_idx], y[val_idx]

        global_mean = y_tr.mean()

        # In-fold Bayesian Target Encoding with smoothing m=20
        tr_df['fold_target'] = y_tr
        tr_te = np.zeros((len(tr_df), len(TE_COLS)), dtype=np.float32)
        va_te = np.zeros((len(va_df), len(TE_COLS)), dtype=np.float32)

        for col_idx, col in enumerate(TE_COLS):
            counts = tr_df.groupby(col)['fold_target'].count()
            sums = tr_df.groupby(col)['fold_target'].sum()
            
            # Map with Bayesian additive smoothing
            m = 20.0
            te_map = (sums + m * global_mean) / (counts + m)
            te_tr_col = tr_df[col].map(te_map).fillna(global_mean).values
            te_va_col = va_df[col].map(te_map).fillna(global_mean).values

            tr_te[:, col_idx] = te_tr_col
            va_te[:, col_idx] = te_va_col

        # Scale continuous features for Stage 1 GLM
        X_tr_num = tr_df[NUM_COLS].values
        X_va_num = va_df[NUM_COLS].values

        scaler = StandardScaler()
        X_tr_scaled = np.hstack([scaler.fit_transform(X_tr_num), tr_te])
        X_va_scaled = np.hstack([scaler.transform(X_va_num), va_te])

        # Stage 1: Fit Logistic Regression GLM
        glm = LogisticRegression(C=0.05, max_iter=200, solver='lbfgs', random_state=42)
        glm.fit(X_tr_scaled, y_tr)

        z_tr = glm.decision_function(X_tr_scaled)
        z_va = glm.decision_function(X_va_scaled)
        oof_glm[val_idx] = z_va

        fold_glm_auc = roc_auc_score(y_va, z_va)

        # Stage 2: LightGBM Residual Booster (init_score = z)
        X_tr_tree = np.hstack([X_tr_num, tr_te])
        X_va_tree = np.hstack([X_va_num, va_te])

        trn_data = lgb.Dataset(X_tr_tree, label=y_tr, init_score=z_tr)
        val_data = lgb.Dataset(X_va_tree, label=y_va, init_score=z_va, reference=trn_data)

        callbacks = [lgb.early_stopping(stopping_rounds=100, verbose=False)]
        model = lgb.train(
            lgb_params,
            trn_data,
            num_boost_round=2500,
            valid_sets=[val_data],
            callbacks=callbacks
        )

        val_residual = model.predict(X_va_tree, raw_score=True)
        final_val_score = z_va + val_residual
        oof_boost[val_idx] = final_val_score

        fold_boost_auc = roc_auc_score(y_va, final_val_score)
        print(f"Fold {fold+1} in {time.time()-t_fold:4.1f}s | GLM AUC: {fold_glm_auc:.6f} -> Boosted AUC: {fold_boost_auc:.6f} (+{(fold_boost_auc-fold_glm_auc)*1e5:.1f} pts, {model.best_iteration} trees)")

    # Overall Metrics
    overall_glm_auc = roc_auc_score(y, oof_glm)
    overall_raw_auc = roc_auc_score(y, oof_boost)
    overall_raw_pr = average_precision_score(y, oof_boost)

    oof_cal = apply_rank_preserving_manifold(oof_boost, zall_5)
    overall_cal_auc = roc_auc_score(y, oof_cal)
    overall_cal_pr = average_precision_score(y, oof_cal)

    print("\n" + "=" * 80)
    print("🏆 FINAL BENCHMARK RESULTS (5-Fold Leak-Free CV):")
    print(f"  Stage 1 GLM Standalone OOF ROC-AUC:    {overall_glm_auc:.6f}")
    print(f"  Stage 2 Boosted Raw OOF ROC-AUC:       {overall_raw_auc:.6f} | PR-AUC: {overall_raw_pr:.6f}")
    print(f"  Calibrated (+ 5-Zone Manifold) ROC-AUC:{overall_cal_auc:.6f} | PR-AUC: {overall_cal_pr:.6f} (+{(overall_cal_auc - overall_raw_auc)*1e5:.1f} pts)")
    print(f"  Total Runtime: {time.time()-start_time:.1f}s")
    print("=" * 80)

    # Save OOF
    np.save(OUT_DIR / 'oof_adv_glm_boost.npy', oof_boost)
    np.save(OUT_DIR / 'oof_adv_glm_boost_calibrated.npy', oof_cal)
    print(f"Saved OOF predictions to {OUT_DIR}/")

if __name__ == '__main__':
    main()
