import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logit, expit
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

print("=" * 75)
print("🧠 PGS-S6E11: QUAD-INTERACTION HIGH-ORDER GLM RESIDUAL BOOSTER")
print("=" * 75)

# 1. Load Data
DATA_DIR = Path("pgs-s6e11/dataset")
OUTPUT_DIR = Path("pgs-s6e11/code/quad_glm_boost/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")
sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")

TARGET = "is_fraud"
y = train_df[TARGET].values
N_TR = len(train_df)
N_TE = len(test_df)
print(f"Loaded {N_TR:,} train rows ({y.sum():,.0f} fraud, {y.mean()*100:.3f}%) and {N_TE:,} test rows.")

# 2. High-Order Feature Creation
print("Constructing 1D, 2-Way, 3-Way, and 4-Way Discrete Keys...")
for df in [train_df, test_df]:
    # 1D Digit Keys
    df["cents"] = np.round((df["amount_usd"] % 1) * 100).astype(int).clip(0, 99)
    df["first_dec"] = np.floor((df["amount_usd"] % 1) * 10).astype(int).clip(0, 9)
    df["last_digit"] = (np.floor(df["amount_usd"]).astype(int) % 10).clip(0, 9)
    df["amount_div10"] = (np.floor(df["amount_usd"]).astype(int) // 10).clip(0, 99)
    df["hours_div5"] = (df["hours_since_last_txn"] // 5).astype(int).clip(0, 20)
    df["dist_div10"] = (df["distance_from_home_km"] // 10).astype(int).clip(0, 30)
    df["risk_binned"] = (df["merchant_risk_score"] // 10).astype(int).clip(0, 10)
    df["time_quarter"] = (df["time_of_day_hour"] // 6).astype(int).clip(0, 3)
    
    # 2-Way Joint Keys
    df["cat_x_risk"] = df["merchant_category"] + "_" + df["risk_binned"].astype(str)
    df["chan_x_auth"] = df["channel"] + "_" + df["auth_method"]
    df["dev_x_auth"] = df["device_type"] + "_" + df["auth_method"]
    df["cents_x_cat"] = df["cents"].astype(str) + "_" + df["merchant_category"]
    df["time_x_cat"] = df["time_of_day_hour"].astype(str) + "_" + df["merchant_category"]
    df["for_x_ip"] = df["is_foreign_transaction"].astype(str) + "_" + df["ip_country_mismatch"].astype(str)
    df["vpn_x_ip"] = df["used_vpn"].astype(str) + "_" + df["ip_country_mismatch"].astype(str)
    
    # 3-Way & 4-Way Complex Fraud Keys
    df["tri_channel"] = df["device_type"] + "_" + df["channel"] + "_" + df["auth_method"]
    df["tri_context"] = df["merchant_category"] + "_" + df["time_quarter"].astype(str) + "_" + df["risk_binned"].astype(str)
    df["quad_security"] = (
        df["is_foreign_transaction"].astype(int).astype(str) + "_" +
        df["used_vpn"].astype(int).astype(str) + "_" +
        df["ip_country_mismatch"].astype(int).astype(str) + "_" +
        df["is_ai_generated_scam_attempt"].astype(int).astype(str)
    )

DISCRETE_KEYS = [
    # 1D
    "cents", "first_dec", "last_digit", "amount_div10", "hours_div5", "dist_div10",
    "merchant_category", "card_type", "auth_method", "channel", "device_type",
    "time_of_day_hour", "day_of_week",
    # 2-Way
    "cat_x_risk", "chan_x_auth", "dev_x_auth", "cents_x_cat", "time_x_cat", "for_x_ip", "vpn_x_ip",
    # 3-Way & 4-Way
    "tri_channel", "tri_context", "quad_security"
]

NUM_COLS = [
    "amount_usd", "hours_since_last_txn", "txn_count_last_24h",
    "distance_from_home_km", "card_age_months", "customer_age",
    "account_balance_usd", "cvv_retry_count", "velocity_score",
    "merchant_risk_score", "prior_disputes"
]

print(f"Total discrete keys to encode: {len(DISCRETE_KEYS)} across multi-scale smoothings (m=20, m=100)")

# 3. Stratified 5-Fold Setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_glm_margins = np.zeros(N_TR, dtype=np.float32)
test_glm_margins = np.zeros(N_TE, dtype=np.float32)

oof_lgb_boost = np.zeros(N_TR, dtype=np.float32)
test_lgb_boost = np.zeros(N_TE, dtype=np.float32)

global_prior = y.mean()
prior_logit = logit(global_prior)

lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 4,
    "max_depth": 3,
    "learning_rate": 0.05,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": 4
}

fold_glm_aucs = []
fold_boost_aucs = []

start_time = time.time()

for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df, y)):
    f_start = time.time()
    print(f"\n--- Fold {fold + 1} / 5 ---")
    
    y_tr, y_va = y[tr_idx], y[va_idx]
    fold_prior = y_tr.mean()
    
    # Build Multi-Scale Out-Of-Fold Target Encodings
    tr_feats = []
    va_feats = []
    te_feats = []
    
    for key in DISCRETE_KEYS:
        s_tr = train_df.iloc[tr_idx][key]
        s_va = train_df.iloc[va_idx][key]
        s_te = test_df[key]
        
        # Aggregate statistics
        stats = pd.DataFrame({"y": y_tr}).groupby(s_tr)["y"].agg(["count", "mean"])
        
        for m in [20, 100]:
            smoothed = (stats["count"] * stats["mean"] + m * fold_prior) / (stats["count"] + m)
            lookup = smoothed.to_dict()
            
            p_tr = s_tr.map(lookup).fillna(fold_prior).values
            p_va = s_va.map(lookup).fillna(fold_prior).values
            p_te = s_te.map(lookup).fillna(fold_prior).values
            
            # Convert to logit space with clipping
            tr_feats.append(logit(np.clip(p_tr, 1e-4, 1 - 1e-4)))
            va_feats.append(logit(np.clip(p_va, 1e-4, 1 - 1e-4)))
            te_feats.append(logit(np.clip(p_te, 1e-4, 1 - 1e-4)))
            
    # Add standardized continuous features
    scaler = StandardScaler()
    tr_num = scaler.fit_transform(train_df.iloc[tr_idx][NUM_COLS].values)
    va_num = scaler.transform(train_df.iloc[va_idx][NUM_COLS].values)
    te_num = scaler.transform(test_df[NUM_COLS].values)
    
    X_tr_glm = np.column_stack(tr_feats + [tr_num])
    X_va_glm = np.column_stack(va_feats + [va_num])
    X_te_glm = np.column_stack(te_feats + [te_num])
    
    # Fit GLM with exact Newton-Cholesky solver
    glm = LogisticRegression(solver="newton-cholesky", C=0.05, max_iter=150, random_state=42)
    glm.fit(X_tr_glm, y_tr)
    
    # Extract linear margins
    z_tr = glm.decision_function(X_tr_glm)
    z_va = glm.decision_function(X_va_glm)
    z_te = glm.decision_function(X_te_glm)
    
    oof_glm_margins[va_idx] = z_va
    test_glm_margins += z_te / 5.0
    
    auc_glm = roc_auc_score(y_va, expit(z_va))
    fold_glm_aucs.append(auc_glm)
    print(f"  GLM alone (Newton-Cholesky) -> Val ROC-AUC: {auc_glm:.6f}")
    
    # Boost 4-leaf LightGBM from GLM Margin
    dtr = lgb.Dataset(X_tr_glm, label=y_tr, init_score=z_tr)
    dva = lgb.Dataset(X_va_glm, label=y_va, init_score=z_va, reference=dtr)
    
    model_lgb = lgb.train(
        lgb_params,
        dtr,
        num_boost_round=600,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    
    # Combined prediction: GLM Margin + Tree Output
    lgb_va_raw = model_lgb.predict(X_va_glm, raw_score=True)
    lgb_te_raw = model_lgb.predict(X_te_glm, raw_score=True)
    
    z_boost_va = z_va + lgb_va_raw
    z_boost_te = z_te + lgb_te_raw
    
    oof_lgb_boost[va_idx] = expit(z_boost_va)
    test_lgb_boost += expit(z_boost_te) / 5.0
    
    auc_boost = roc_auc_score(y_va, oof_lgb_boost[va_idx])
    fold_boost_aucs.append(auc_boost)
    print(f"  GLM + 4-Leaf LightGBM Boost -> Val ROC-AUC: {auc_boost:.6f} (+{auc_boost - auc_glm:+.6f}) [Elapsed: {(time.time() - f_start):.1f}s]")

# 4. Overall Evaluation
glm_auc_total = roc_auc_score(y, expit(oof_glm_margins))
glm_pr_total = average_precision_score(y, expit(oof_glm_margins))
boost_auc_total = roc_auc_score(y, oof_lgb_boost)
boost_pr_total = average_precision_score(y, oof_lgb_boost)

print("\n" + "=" * 75)
print("🏆 QUAD-INTERACTION GLM RESULTS (5-Fold Stratified CV)")
print("=" * 75)
print(f"GLM Alone (Newton-Cholesky) : OOF ROC-AUC = {glm_auc_total:.6f} | PR-AUC = {glm_pr_total:.6f}")
print(f"GLM + 4-Leaf LightGBM Boost  : OOF ROC-AUC = {boost_auc_total:.6f} | PR-AUC = {boost_pr_total:.6f}")
print(f"Total Execution Time: {(time.time() - start_time)/60:.2f} minutes")
print("=" * 75)

# 5. Save Artifacts
np.save(OUTPUT_DIR / "oof_quad_glm_margin.npy", oof_glm_margins)
np.save(OUTPUT_DIR / "test_quad_glm_margin.npy", test_glm_margins)
np.save(OUTPUT_DIR / "oof_quad_glm_boost.npy", oof_lgb_boost)
np.save(OUTPUT_DIR / "test_quad_glm_boost.npy", test_lgb_boost)

sub = sample_sub.copy()
sub["is_fraud"] = rankdata(test_lgb_boost) / N_TE
sub_path = OUTPUT_DIR / "quad_glm_boost_submission.csv"
sub.to_csv(sub_path, index=False)
print(f"\nSaved OOF predictions to {OUTPUT_DIR}")
print(f"Generated Submission: {sub_path}")
