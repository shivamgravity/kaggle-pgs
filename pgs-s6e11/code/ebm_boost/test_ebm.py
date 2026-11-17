import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from interpret.glassbox import ExplainableBoostingClassifier
import warnings
warnings.filterwarnings("ignore")

print("=" * 75)
print("🧠 PGS-S6E11: PAIRWISE EXPLAINABLE BOOSTING MACHINE (EBM) TEST")
print("=" * 75)

DATA_DIR = Path("pgs-s6e11/dataset")
train_df = pd.read_csv(DATA_DIR / "train.csv")
y = train_df["is_fraud"].values
N_TR = len(train_df)

# Feature Engineering
train_df["cents"] = np.round((train_df["amount_usd"] % 1) * 100).astype(int).clip(0, 99)
train_df["first_dec"] = np.floor((train_df["amount_usd"] % 1) * 10).astype(int).clip(0, 9)
train_df["last_digit"] = (np.floor(train_df["amount_usd"]).astype(int) % 10).clip(0, 9)
train_df["spend_velocity"] = train_df["amount_usd"] / (train_df["hours_since_last_txn"] + 0.01)
train_df["travel_speed_kmh"] = train_df["distance_from_home_km"] / (train_df["hours_since_last_txn"] + 0.01)

FEATURES = [
    "amount_usd", "hours_since_last_txn", "account_balance_usd",
    "distance_from_home_km", "merchant_risk_score", "velocity_score",
    "spend_velocity", "travel_speed_kmh",
    "cents", "first_dec", "last_digit",
    "merchant_category", "card_type", "auth_method", "channel",
    "device_type", "time_of_day_hour", "day_of_week",
    "is_foreign_transaction", "used_vpn", "ip_country_mismatch"
]

print(f"Features: {len(FEATURES)}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df, y)):
    if fold > 0:
        break
    print(f"\nTraining EBM on Fold 0 ({len(tr_idx):,} train rows, {len(va_idx):,} val rows)...")
    t0 = time.time()
    ebm = ExplainableBoostingClassifier(
        interactions=10,
        outer_bags=2,
        inner_bags=0,
        learning_rate=0.05,
        max_bins=128,
        n_jobs=4,
        random_state=42
    )
    ebm.fit(train_df.iloc[tr_idx][FEATURES], y[tr_idx])
    val_probs = ebm.predict_proba(train_df.iloc[va_idx][FEATURES])[:, 1]
    auc = roc_auc_score(y[va_idx], val_probs)
    pr = average_precision_score(y[va_idx], val_probs)
    print(f"Fold 0 EBM -> ROC-AUC: {auc:.6f} | PR-AUC: {pr:.6f} (Elapsed: {(time.time() - t0):.1f}s)")
