import json
from pathlib import Path

nb = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

def add_md(src):
    nb["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in src.split("\n")]
    })

def add_code(src):
    nb["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in src.split("\n")]
    })

# Cell 0: Title & Strategy
add_md("""# 🧠 PGS-S6E11: Feature Tokenizer Transformer (FT-Transformer)
### Kaggle Playground Series Season 6 Episode 11: Credit Card Fraud Detection

**Why FT-Transformer is the Ultimate Orthogonal Deep Tabular Architecture:**
1. **Feature Tokenizer**: Each continuous feature (amount, spend velocity, travel speed, account balance) is projected via dedicated linear weights into a $d=64$ dimensional token vector.
2. **Discrete Digit & Categorical Embeddings**: Cents (0-99), first decimal (0-9), last digit, leading digits, and security flags are embedded into the same token space.
3. **Multi-Head Self-Attention**: Rather than hardcoding pairwise feature combinations, the Transformer computes pairwise attention between all 35 feature tokens simultaneously.
4. **Non-Tree Inductive Bias**: While tree splitters partition orthogonal hyperplanes with axis-aligned steps, FT-Transformer models smooth contextual interactions with continuous attention gradients.""")

# Cell 1: Imports & Setup
add_code("""import os
import gc
import glob
import time
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Executing on device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
""")

# Cell 2: Configuration
add_code("""class Config:
    SEED = 42
    N_FOLDS = 5
    BATCH_SIZE = 1024
    EPOCHS = 12
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    PATIENCE = 4
    
    # Model Hyperparameters
    D_TOKEN = 64
    N_HEADS = 4
    N_LAYERS = 3
    FFN_RATIO = 4.0
    DROPOUT = 0.15
    
    # Path resolution via dynamic search
    train_candidates = (
        glob.glob("/kaggle/input/**/train.csv", recursive=True)
        + glob.glob("**/train.csv", recursive=True)
        + glob.glob("../../dataset/train.csv")
        + glob.glob("pgs-s6e11/dataset/train.csv")
    )
    if train_candidates:
        DATA_DIR = Path(train_candidates[0]).parent
    else:
        DATA_DIR = Path("pgs-s6e11/dataset")
        
    OUTPUT_DIR = Path(".")

def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(Config.SEED)
print(f"Configuration initialized. Data Dir: {Config.DATA_DIR}")
""")

# Cell 3: Data Loading & Feature Engineering
add_code("""# Load data
train_df = pd.read_csv(Config.DATA_DIR / "train.csv")
test_df = pd.read_csv(Config.DATA_DIR / "test.csv")
sample_sub = pd.read_csv(Config.DATA_DIR / "sample_submission.csv")

TARGET = "is_fraud"
ID_COL = "id"

print(f"Loaded train shape: {train_df.shape}, test shape: {test_df.shape}")

def engineer_features(df):
    df = df.copy()
    
    # 1. Digit & Cents Features
    df["cents"] = np.round((df["amount_usd"] % 1) * 100).astype(int).clip(0, 99)
    df["first_decimal"] = np.floor((df["amount_usd"] % 1) * 10).astype(int).clip(0, 9)
    df["last_digit"] = (np.floor(df["amount_usd"]).astype(int) % 10).clip(0, 9)
    df["amount_int_div10"] = (np.floor(df["amount_usd"]).astype(int) // 10).clip(0, 99)
    df["is_round_dollar"] = (df["cents"] == 0).astype(int)
    
    df["amount_str"] = df["amount_usd"].astype(str)
    df["amount_leading_digit"] = df["amount_str"].str[0].astype(int).clip(1, 9)
    
    df["bal_str"] = np.maximum(df["account_balance_usd"], 0).astype(int).astype(str)
    df["balance_leading_digit"] = df["bal_str"].str[0].replace({"-": "0"}).astype(int).clip(0, 9)
    df.drop(columns=["amount_str", "bal_str"], inplace=True)
    
    # 2. Velocity & Burst Features
    df["spend_velocity"] = df["amount_usd"] / (df["hours_since_last_txn"] + 0.01)
    df["travel_speed_kmh"] = df["distance_from_home_km"] / (df["hours_since_last_txn"] + 0.01)
    df["balance_drain_ratio"] = df["amount_usd"] / (np.maximum(df["account_balance_usd"], 0) + 1.0)
    df["cvv_x_risk"] = (df["cvv_retry_count"] + 1) * df["merchant_risk_score"]
    
    # 3. Security Red Flag Composites
    flag_cols = [
        "is_foreign_transaction", "is_new_merchant", "used_vpn",
        "ip_country_mismatch", "billing_shipping_mismatch", "is_ai_generated_scam_attempt"
    ]
    df["total_flags"] = df[flag_cols].astype(int).sum(axis=1)
    df["fraud_risk_index"] = df["total_flags"] * df["merchant_risk_score"]
    
    return df

print("Engineering features...")
train_df = engineer_features(train_df)
test_df = engineer_features(test_df)

# Feature Categorization
CAT_COLS = [
    "merchant_category", "card_type", "auth_method", "channel", "device_type",
    "time_of_day_hour", "day_of_week", "cents", "first_decimal", "last_digit",
    "amount_int_div10", "amount_leading_digit", "balance_leading_digit", "is_round_dollar",
    "is_foreign_transaction", "is_new_merchant", "used_vpn", "ip_country_mismatch",
    "billing_shipping_mismatch", "is_ai_generated_scam_attempt", "total_flags"
]

NUM_COLS = [
    "amount_usd", "hours_since_last_txn", "txn_count_last_24h",
    "distance_from_home_km", "card_age_months", "customer_age",
    "account_balance_usd", "cvv_retry_count", "velocity_score",
    "merchant_risk_score", "prior_disputes", "spend_velocity",
    "travel_speed_kmh", "balance_drain_ratio", "cvv_x_risk", "fraud_risk_index"
]

# Integer encode categoricals
cat_maps = {}
for col in CAT_COLS:
    train_df[col] = train_df[col].astype(str)
    test_df[col] = test_df[col].astype(str)
    unique_vals = sorted(train_df[col].unique().tolist())
    val_to_idx = {val: idx for idx, val in enumerate(unique_vals)}
    train_df[col] = train_df[col].map(val_to_idx).fillna(0).astype(int)
    test_df[col] = test_df[col].map(lambda x: val_to_idx.get(x, 0)).astype(int)
    cat_maps[col] = val_to_idx

emb_cardinalities = [len(cat_maps[col]) for col in CAT_COLS]
print(f"Categorical features: {len(CAT_COLS)} (Cardinalities: {emb_cardinalities})")
print(f"Continuous features: {len(NUM_COLS)}")

# Quantile transform numerical features for robust attention stability
scaler = QuantileTransformer(n_quantiles=2000, output_distribution="normal", random_state=Config.SEED)
train_num_scaled = scaler.fit_transform(train_df[NUM_COLS].values).astype(np.float32)
test_num_scaled = scaler.transform(test_df[NUM_COLS].values).astype(np.float32)

train_cat = train_df[CAT_COLS].values.astype(np.int64)
test_cat = test_df[CAT_COLS].values.astype(np.int64)
y_all = train_df[TARGET].values.astype(np.float32)
""")

# Cell 4: FT-Transformer Architecture Definition
add_code("""class TabularDataset(Dataset):
    def __init__(self, cat_data, num_data, targets=None):
        self.cat_data = torch.tensor(cat_data, dtype=torch.long)
        self.num_data = torch.tensor(num_data, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32) if targets is not None else None

    def __len__(self):
        return len(self.cat_data)

    def __getitem__(self, idx):
        if self.targets is not None:
            return self.cat_data[idx], self.num_data[idx], self.targets[idx]
        return self.cat_data[idx], self.num_data[idx]

class FeatureTokenizer(nn.Module):
    def __init__(self, cardinalities, n_num, d_token):
        super().__init__()
        # Categorical tokenizers (Lookup tables)
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(card, d_token) for card in cardinalities
        ])
        for emb in self.cat_embeddings:
            nn.init.kaiming_uniform_(emb.weight, a=math.sqrt(5))
            
        # Numerical tokenizers (Dedicated linear projection per continuous feature)
        self.num_weight = nn.Parameter(torch.empty(n_num, d_token))
        self.num_bias = nn.Parameter(torch.empty(n_num, d_token))
        nn.init.kaiming_uniform_(self.num_weight, a=math.sqrt(5))
        nn.init.zeros_(self.num_bias)
        
        # Learnable [CLS] Token
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_token))
        nn.init.kaiming_uniform_(self.cls_token, a=math.sqrt(5))

    def forward(self, x_cat, x_num):
        B = x_cat.size(0)
        
        # 1. Embed categoricals -> (B, n_cat, d_token)
        cat_tokens = [emb(x_cat[:, i]).unsqueeze(1) for i, emb in enumerate(self.cat_embeddings)]
        cat_tokens = torch.cat(cat_tokens, dim=1)
        
        # 2. Embed numericals -> (B, n_num, d_token)
        # x_num is (B, n_num) -> (B, n_num, 1) * (1, n_num, d_token) + (1, n_num, d_token)
        num_tokens = x_num.unsqueeze(-1) * self.num_weight.unsqueeze(0) + self.num_bias.unsqueeze(0)
        
        # 3. Prepend [CLS] token -> (B, 1 + n_cat + n_num, d_token)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, cat_tokens, num_tokens], dim=1)
        return tokens

class TransformerBlock(nn.Module):
    def __init__(self, d_token, n_heads, ffn_ratio=4.0, dropout=0.15):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_token)
        self.mha = nn.MultiheadAttention(d_token, n_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        
        d_ffn = int(d_token * ffn_ratio)
        self.ln2 = nn.LayerNorm(d_token)
        self.ffn = nn.Sequential(
            nn.Linear(d_token, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_token),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Pre-LN Self-Attention
        res = x
        x_norm = self.ln1(x)
        attn_out, _ = self.mha(x_norm, x_norm, x_norm)
        x = res + self.drop1(attn_out)
        
        # Pre-LN FeedForward
        x = x + self.ffn(self.ln2(x))
        return x

class FTTransformer(nn.Module):
    def __init__(self, cardinalities, n_num, d_token=64, n_heads=4, n_layers=3, ffn_ratio=4.0, dropout=0.15):
        super().__init__()
        self.tokenizer = FeatureTokenizer(cardinalities, n_num, d_token)
        self.layers = nn.ModuleList([
            TransformerBlock(d_token, n_heads, ffn_ratio, dropout) for _ in range(n_layers)
        ])
        self.ln_head = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, 1)

    def forward(self, x_cat, x_num):
        tokens = self.tokenizer(x_cat, x_num)
        for layer in self.layers:
            tokens = layer(tokens)
            
        # Extract representation of [CLS] token
        cls_rep = self.ln_head(tokens[:, 0])
        logits = self.head(cls_rep).squeeze(-1)
        return logits
""")

# Cell 5: Stratified 5-Fold Training Loop
add_code("""N_TR = len(train_df)
N_TE = len(test_df)
skf = StratifiedKFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)

oof_preds = np.zeros(N_TR, dtype=np.float32)
test_preds = np.zeros(N_TE, dtype=np.float32)

test_dataset = TabularDataset(test_cat, test_num_scaled)
test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE * 2, shuffle=False, num_workers=2)

pos_weight = torch.tensor([(N_TR - y_all.sum()) / y_all.sum()], device=device)
print(f"Starting 5-Fold FT-Transformer Training (pos_weight: {pos_weight.item():.2f})...")

for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, y_all)):
    fold_start = time.time()
    print(f"\\n--- Fold {fold + 1} / {Config.N_FOLDS} ---")
    
    # Create datasets & loaders
    train_dataset = TabularDataset(train_cat[train_idx], train_num_scaled[train_idx], y_all[train_idx])
    val_dataset = TabularDataset(train_cat[val_idx], train_num_scaled[val_idx], y_all[val_idx])
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE * 2, shuffle=False, num_workers=2, pin_memory=True)
    
    # Initialize FT-Transformer
    model = FTTransformer(
        cardinalities=emb_cardinalities,
        n_num=len(NUM_COLS),
        d_token=Config.D_TOKEN,
        n_heads=Config.N_HEADS,
        n_layers=Config.N_LAYERS,
        ffn_ratio=Config.FFN_RATIO,
        dropout=Config.DROPOUT
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=1e-5)
    scaler_amp = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    
    best_val_auc = 0.0
    best_val_preds = None
    patience_cnt = 0
    best_state = None
    
    for epoch in range(Config.EPOCHS):
        model.train()
        total_loss = 0.0
        
        for x_cat_b, x_num_b, y_b in train_loader:
            x_cat_b = x_cat_b.to(device, non_blocking=True)
            x_num_b = x_num_b.to(device, non_blocking=True)
            y_b = y_b.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(x_cat_b, x_num_b)
                loss = criterion(logits, y_b)
                
            scaler_amp.scale(loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()
            
            total_loss += loss.item() * len(y_b)
            
        scheduler.step()
        train_loss = total_loss / len(train_idx)
        
        # Validation
        model.eval()
        val_preds_list = []
        with torch.no_grad():
            for x_cat_b, x_num_b, _ in val_loader:
                x_cat_b = x_cat_b.to(device, non_blocking=True)
                x_num_b = x_num_b.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    logits = model(x_cat_b, x_num_b)
                    probs = torch.sigmoid(logits)
                val_preds_list.append(probs.cpu().numpy())
                
        val_preds = np.concatenate(val_preds_list)
        val_auc = roc_auc_score(y_all[val_idx], val_preds)
        val_pr = average_precision_score(y_all[val_idx], val_preds)
        
        print(f"  Epoch {epoch + 1:2d} | Train Loss: {train_loss:.5f} | Val ROC-AUC: {val_auc:.6f} | Val PR-AUC: {val_pr:.6f}")
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_preds = val_preds
            patience_cnt = 0
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= Config.PATIENCE:
                print(f"  Early stopping triggered at epoch {epoch + 1}")
                break
                
    print(f"Fold {fold + 1} Best Val ROC-AUC: {best_val_auc:.6f} (Elapsed: {(time.time() - fold_start)/60:.2f} min)")
    oof_preds[val_idx] = best_val_preds
    
    # Test inference using best model
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    fold_test_preds = []
    with torch.no_grad():
        for x_cat_b, x_num_b in test_loader:
            x_cat_b = x_cat_b.to(device, non_blocking=True)
            x_num_b = x_num_b.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(x_cat_b, x_num_b)
                probs = torch.sigmoid(logits)
            fold_test_preds.append(probs.cpu().numpy())
    test_preds += np.concatenate(fold_test_preds) / Config.N_FOLDS

overall_auc = roc_auc_score(y_all, oof_preds)
overall_pr = average_precision_score(y_all, oof_preds)
print("\\n" + "=" * 65)
print(f"🏆 FT-Transformer 5-Fold OOF ROC-AUC: {overall_auc:.6f}")
print(f"📈 FT-Transformer 5-Fold OOF PR-AUC:  {overall_pr:.6f}")
print("=" * 65)
""")

# Cell 6: Save Artifacts & Generate Submissions
add_code("""# Save Out-Of-Fold predictions
np.save(Config.OUTPUT_DIR / "oof_ft_transformer.npy", oof_preds)
np.save(Config.OUTPUT_DIR / "test_ft_transformer.npy", test_preds)

# Generate Tabular Transformer Standalone Submission
sub_transformer = sample_sub.copy()
sub_transformer["is_fraud"] = test_preds
sub_transformer.to_csv(Config.OUTPUT_DIR / "ft_transformer_submission.csv", index=False)

print(f"Saved OOF predictions to: {Config.OUTPUT_DIR / 'oof_ft_transformer.npy'}")
print(f"Saved Test predictions to: {Config.OUTPUT_DIR / 'test_ft_transformer.npy'}")
print(f"Saved Standalone Submission: {Config.OUTPUT_DIR / 'ft_transformer_submission.csv'}")
""")

output_path = Path("pgs-s6e11/code/tabular_transformer/model.ipynb")
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w") as f:
    json.dump(nb, f, indent=1)

print(f"Successfully generated notebook: {output_path}")
