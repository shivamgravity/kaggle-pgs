import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.10.12"
    }
}
cells = []

# Cell 0: Header
cells.append(nbf.v4.new_markdown_cell("""# 🧠 PGS-S6E11: Multi-Seed Bagged Dual-Neural Architecture (ResNet + FT-Transformer)
### Kaggle Playground Series Season 6 Episode 11: Credit Card Fraud Detection

**Overview & Strategy:**
1. **Multi-Seed Ensembling across Diverse Neural Paradigms**:
   - Neural network training is sensitive to weight initialization and SGD batch trajectories.
   - By training across 3 distinct seeds (`42`, `100`, `2026`) across 5 stratified folds:
     - **Tabular Deep ResNet**: Pre-activation residual blocks with learned entity embeddings for cents, digits, and categories.
     - **Feature Tokenizer Transformer (FT-Transformer)**: Dedicated continuous tokenizers + categorical embeddings into a shared token space with 3 Pre-LN Transformer self-attention blocks.
2. **FlashAttention & AMP FP16 Acceleration**:
   - Uses PyTorch's native `F.scaled_dot_product_attention` and `torch.cuda.amp.autocast()`.
   - Extremely fast convergence on Kaggle GPU.
3. **Outputs Generated**:
   - `oof_resnet_bag3.npy`, `test_resnet_bag3.npy`
   - `oof_ft_transformer_bag3.npy`, `test_ft_transformer_bag3.npy`
   - `oof_dual_neural_bag3.npy`, `test_dual_neural_bag3.npy`
   - `dual_neural_bag3_submission.csv`
"""))

# Cell 1: Imports & GPU Environment
cells.append(nbf.v4.new_code_cell("""import os
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} | Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
"""))

# Cell 2: Configuration
cells.append(nbf.v4.new_code_cell("""class Config:
    SEEDS = [42, 100, 2026]
    N_FOLDS = 5
    BATCH_SIZE = 1024
    
    # ResNet parameters
    RESNET_EPOCHS = 12
    RESNET_LR = 2e-3
    
    # FT-Transformer parameters
    TRANSFORMER_EPOCHS = 10
    TRANSFORMER_LR = 1e-3
    D_TOKEN = 64
    N_HEADS = 4
    N_LAYERS = 3
    DROPOUT = 0.15
    
    # Data path resolution
    train_candidates = (
        glob.glob("/kaggle/input/**/train.csv", recursive=True)
        + glob.glob("**/train.csv", recursive=True)
        + glob.glob("../../dataset/train.csv")
        + glob.glob("pgs-s6e11/dataset/train.csv")
    )
    if not train_candidates:
        raise FileNotFoundError("Could not find train.csv in candidate paths.")
    
    TRAIN_PATH = Path(train_candidates[0])
    DATA_DIR = TRAIN_PATH.parent
    TEST_PATH = DATA_DIR / "test.csv"
    SUB_PATH = DATA_DIR / "sample_submission.csv"
    OUTPUT_DIR = Path(".")
    
    print(f"Resolved Data Directory: {DATA_DIR}")
    print(f"Multi-Seed Configuration: {SEEDS} across {N_FOLDS} folds.")
"""))

# Cell 3: Feature Engineering & Preprocessing
cells.append(nbf.v4.new_code_cell("""train_df = pd.read_csv(Config.TRAIN_PATH)
test_df = pd.read_csv(Config.TEST_PATH)
sample_sub = pd.read_csv(Config.SUB_PATH)

TARGET, ID_COL = "is_fraud", "id"
FLAG_COLS = list(train_df.select_dtypes(include=["bool"]).columns)
for df in (train_df, test_df):
    df[FLAG_COLS] = df[FLAG_COLS].astype("int8")

y = train_df[TARGET].astype(int).values
N_TR, N_TE = len(train_df), len(test_df)
print(f"Loaded {N_TR:,} train rows ({y.sum():,} fraud, {y.mean()*100:.3f}%) and {N_TE:,} test rows.")

# 1. Feature Engineering
cat_means = train_df.groupby("merchant_category")["amount_usd"].transform("mean")
cat_stds = train_df.groupby("merchant_category")["amount_usd"].transform("std")
cat_mean_map = train_df.groupby("merchant_category")["amount_usd"].mean().to_dict()
cat_std_map = train_df.groupby("merchant_category")["amount_usd"].std().to_dict()

for df in (train_df, test_df):
    # Velocity & Burst
    df["spend_velocity"] = df["amount_usd"] / (df["hours_since_last_txn"] + 0.05)
    df["travel_speed_kmh"] = df["distance_from_home_km"] / (df["hours_since_last_txn"] + 0.05)
    df["balance_drain_ratio"] = df["amount_usd"] / (df["account_balance_usd"] + 1.0)
    df["amount_24h_volume"] = df["amount_usd"] * df["txn_count_last_24h"]
    df["velocity_x_amount"] = df["velocity_score"] * df["amount_usd"]
    df["cvv_x_risk"] = df["cvv_retry_count"] * df["merchant_risk_score"]
    df["total_flags"] = df[FLAG_COLS].sum(axis=1)
    
    # Category Relative
    c_mean = df["merchant_category"].map(cat_mean_map).fillna(100.0)
    c_std = df["merchant_category"].map(cat_std_map).fillna(50.0)
    df["amount_to_cat_ratio"] = df["amount_usd"] / (c_mean + 1.0)
    df["amount_cat_zscore"] = (df["amount_usd"] - c_mean) / (c_std + 1e-4)

    # Digit Extractions
    df["cents"] = np.round((df["amount_usd"] % 1) * 100).astype(int).clip(0, 99)
    df["first_dec"] = np.floor((df["amount_usd"] % 1) * 10).astype(int).clip(0, 9)
    df["last_digit"] = (np.floor(df["amount_usd"]).astype(int) % 10).clip(0, 9)
    df["amount_div10"] = (np.floor(df["amount_usd"]).astype(int) // 10).clip(0, 99)
    df["hours_div5"] = (df["hours_since_last_txn"] // 5).astype(int).clip(0, 20)
    df["dist_div10"] = (df["distance_from_home_km"] // 10).astype(int).clip(0, 30)
    df["risk_binned"] = (df["merchant_risk_score"] // 10).astype(int).clip(0, 10)

DISCRETE_COLS = [
    "cents", "first_dec", "last_digit", "amount_div10", "hours_div5", "dist_div10", "risk_binned",
    "merchant_category", "card_type", "auth_method", "channel", "device_type",
    "time_of_day_hour", "day_of_week", "total_flags"
]

CONTINUOUS_COLS = [
    "amount_usd", "hours_since_last_txn", "txn_count_last_24h", "distance_from_home_km",
    "card_age_months", "customer_age", "account_balance_usd", "cvv_retry_count",
    "velocity_score", "merchant_risk_score", "prior_disputes",
    "spend_velocity", "travel_speed_kmh", "balance_drain_ratio",
    "amount_to_cat_ratio", "amount_cat_zscore"
]

# Map categoricals to contiguous integers
cat_maps = {}
for col in DISCRETE_COLS:
    vals = sorted(train_df[col].astype(str).unique())
    cat_maps[col] = {v: i for i, v in enumerate(vals)}
    train_df[col] = train_df[col].astype(str).map(cat_maps[col]).fillna(0).astype(int)
    test_df[col] = test_df[col].astype(str).map(cat_maps[col]).fillna(0).astype(int)

CARDINALITIES = [len(cat_maps[col]) for col in DISCRETE_COLS]
print(f"Discrete Features: {len(DISCRETE_COLS)} | Continuous Features: {len(CONTINUOUS_COLS)}")
"""))

# Cell 4: Neural Model Architectures (ResNet & FT-Transformer)
cells.append(nbf.v4.new_code_cell("""# 1. Tabular Deep ResNet
class ResNetBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2):
        super().__init__()
        self.skip = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
        self.fc1 = nn.Linear(in_features, out_features)
        self.ln1 = nn.LayerNorm(out_features)
        self.act1 = nn.SiLU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(out_features, out_features)
        self.ln2 = nn.LayerNorm(out_features)
        self.act2 = nn.SiLU()
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        res = self.skip(x)
        h = self.drop1(self.act1(self.ln1(self.fc1(x))))
        h = self.drop2(self.act2(self.ln2(self.fc2(h))))
        return h + res

class TabularResNet(nn.Module):
    def __init__(self, cardinalities, n_cont, hidden_dims=[256, 128, 64], dropout=0.2):
        super().__init__()
        # Embedding dimensions: min(16, max(4, card // 2))
        emb_dims = [min(16, max(4, c // 2)) for c in cardinalities]
        self.embeddings = nn.ModuleList([
            nn.Embedding(card, dim) for card, dim in zip(cardinalities, emb_dims)
        ])
        total_in = sum(emb_dims) + n_cont
        self.input_layer = nn.Sequential(
            nn.Linear(total_in, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        self.block1 = ResNetBlock(hidden_dims[0], hidden_dims[0], dropout=dropout)
        self.block2 = ResNetBlock(hidden_dims[0], hidden_dims[1], dropout=dropout)
        self.block3 = ResNetBlock(hidden_dims[1], hidden_dims[2], dropout=dropout / 2)
        self.head = nn.Linear(hidden_dims[2], 1)

    def forward(self, x_cat, x_num):
        emb_outs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(emb_outs + [x_num], dim=1)
        h = self.input_layer(x)
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        return self.head(h).squeeze(-1)

# 2. Feature Tokenizer Transformer (FT-Transformer)
class FTTransformer(nn.Module):
    def __init__(self, cardinalities, n_cont, d_token=64, n_heads=4, n_layers=3, dropout=0.15):
        super().__init__()
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(card, d_token) for card in cardinalities
        ])
        self.cont_weight = nn.Parameter(torch.empty(n_cont, d_token))
        self.cont_bias = nn.Parameter(torch.empty(n_cont, d_token))
        nn.init.kaiming_uniform_(self.cont_weight, a=math.sqrt(5))
        nn.init.zeros_(self.cont_bias)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.normal_(self.cls_token, std=0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=int(d_token * 4),
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, 1)

    def forward(self, x_cat, x_num):
        B = x_cat.size(0)
        cat_tokens = torch.stack([emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)], dim=1)
        cont_tokens = x_num.unsqueeze(-1) * self.cont_weight.unsqueeze(0) + self.cont_bias.unsqueeze(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, cat_tokens, cont_tokens], dim=1)
        out = self.transformer(tokens)
        cls_out = self.norm(out[:, 0])
        return self.head(cls_out).squeeze(-1)

print("Architectures compiled: TabularResNet and FT-Transformer.")
"""))

# Cell 5: Tabular Dataset & DataLoaders
cells.append(nbf.v4.new_code_cell("""class TabularDataset(Dataset):
    def __init__(self, cat_data, num_data, labels=None):
        self.cat_data = torch.as_tensor(cat_data, dtype=torch.long)
        self.num_data = torch.as_tensor(num_data, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.float32) if labels is not None else None

    def __len__(self):
        return len(self.cat_data)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.cat_data[idx], self.num_data[idx], self.labels[idx]
        return self.cat_data[idx], self.num_data[idx]
"""))

# Cell 6: Multi-Seed Training Pipeline
cells.append(nbf.v4.new_code_cell("""# Containers for predictions
cat_train = train_df[DISCRETE_COLS].values
cat_test = test_df[DISCRETE_COLS].values

oof_resnet_seeds = []
test_resnet_seeds = []

oof_ft_seeds = []
test_ft_seeds = []

skf = StratifiedKFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=42)

for seed_idx, SEED in enumerate(Config.SEEDS):
    print("=" * 75)
    print(f"🚀 TRAINING MULTI-SEED RUN: SEED {SEED} ({seed_idx + 1}/{len(Config.SEEDS)})")
    print("=" * 75)
    
    oof_resnet_seed = np.zeros(N_TR, dtype=np.float32)
    test_resnet_seed = np.zeros(N_TE, dtype=np.float32)
    
    oof_ft_seed = np.zeros(N_TR, dtype=np.float32)
    test_ft_seed = np.zeros(N_TE, dtype=np.float32)
    
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df, y)):
        fold_t0 = time.time()
        
        # Preprocessing Quantile + Standard Scaling
        qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal", subsample=150_000, random_state=SEED)
        sc = StandardScaler()
        
        num_tr = sc.fit_transform(qt.fit_transform(train_df.iloc[tr_idx][CONTINUOUS_COLS].values))
        num_va = sc.transform(qt.transform(train_df.iloc[va_idx][CONTINUOUS_COLS].values))
        num_te = sc.transform(qt.transform(test_df[CONTINUOUS_COLS].values))
        
        tr_ds = TabularDataset(cat_train[tr_idx], num_tr, y[tr_idx])
        va_ds = TabularDataset(cat_train[va_idx], num_va, y[va_idx])
        te_ds = TabularDataset(cat_test, num_te)
        
        tr_loader = DataLoader(tr_ds, batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=True, num_workers=2)
        va_loader = DataLoader(va_ds, batch_size=Config.BATCH_SIZE * 2, shuffle=False, pin_memory=True, num_workers=2)
        te_loader = DataLoader(te_ds, batch_size=Config.BATCH_SIZE * 2, shuffle=False, pin_memory=True, num_workers=2)
        
        # ---------------------------------------------------------
        # A. TRAIN TABULAR RESNET
        # ---------------------------------------------------------
        torch.manual_seed(SEED + fold)
        model_resnet = TabularResNet(CARDINALITIES, len(CONTINUOUS_COLS)).to(device)
        criterion = nn.BCEWithLogitsLoss()
        opt_resnet = torch.optim.AdamW(model_resnet.parameters(), lr=Config.RESNET_LR, weight_decay=1e-4)
        sched_resnet = torch.optim.lr_scheduler.OneCycleLR(
            opt_resnet, max_lr=Config.RESNET_LR, total_steps=len(tr_loader) * Config.RESNET_EPOCHS,
            pct_start=0.15, anneal_strategy="cos"
        )
        scaler = torch.cuda.amp.GradScaler()
        
        best_auc_rn = 0.0
        best_weights_rn = None
        for epoch in range(Config.RESNET_EPOCHS):
            model_resnet.train()
            for x_c, x_n, y_b in tr_loader:
                x_c, x_n, y_b = x_c.to(device), x_n.to(device), y_b.to(device)
                opt_resnet.zero_grad()
                with torch.cuda.amp.autocast():
                    out = model_resnet(x_c, x_n)
                    loss = criterion(out, y_b)
                scaler.scale(loss).backward()
                scaler.step(opt_resnet)
                scaler.update()
                sched_resnet.step()
                
            # Val eval
            model_resnet.eval()
            val_preds = []
            with torch.no_grad():
                for x_c, x_n, _ in va_loader:
                    x_c, x_n = x_c.to(device), x_n.to(device)
                    with torch.cuda.amp.autocast():
                        val_preds.append(torch.sigmoid(model_resnet(x_c, x_n)).cpu().numpy())
            val_p = np.concatenate(val_preds)
            auc = roc_auc_score(y[va_idx], val_p)
            if auc > best_auc_rn:
                best_auc_rn = auc
                best_weights_rn = {k: v.cpu() for k, v in model_resnet.state_dict().items()}
                
        model_resnet.load_state_dict({k: v.to(device) for k, v in best_weights_rn.items()})
        model_resnet.eval()
        
        # Predict Fold
        val_preds = []
        with torch.no_grad():
            for x_c, x_n, _ in va_loader:
                x_c, x_n = x_c.to(device), x_n.to(device)
                with torch.cuda.amp.autocast():
                    val_preds.append(torch.sigmoid(model_resnet(x_c, x_n)).cpu().numpy())
        oof_resnet_seed[va_idx] = np.concatenate(val_preds)
        
        test_preds = []
        with torch.no_grad():
            for x_c, x_n in te_loader:
                x_c, x_n = x_c.to(device), x_n.to(device)
                with torch.cuda.amp.autocast():
                    test_preds.append(torch.sigmoid(model_resnet(x_c, x_n)).cpu().numpy())
        test_resnet_seed += np.concatenate(test_preds) / Config.N_FOLDS
        
        del model_resnet, opt_resnet, sched_resnet
        torch.cuda.empty_cache()
        gc.collect()
        
        # ---------------------------------------------------------
        # B. TRAIN FT-TRANSFORMER
        # ---------------------------------------------------------
        torch.manual_seed(SEED + 500 + fold)
        model_ft = FTTransformer(CARDINALITIES, len(CONTINUOUS_COLS), d_token=Config.D_TOKEN, n_heads=Config.N_HEADS, n_layers=Config.N_LAYERS).to(device)
        opt_ft = torch.optim.AdamW(model_ft.parameters(), lr=Config.TRANSFORMER_LR, weight_decay=1e-4)
        sched_ft = torch.optim.lr_scheduler.OneCycleLR(
            opt_ft, max_lr=Config.TRANSFORMER_LR, total_steps=len(tr_loader) * Config.TRANSFORMER_EPOCHS,
            pct_start=0.15, anneal_strategy="cos"
        )
        
        best_auc_ft = 0.0
        best_weights_ft = None
        for epoch in range(Config.TRANSFORMER_EPOCHS):
            model_ft.train()
            for x_c, x_n, y_b in tr_loader:
                x_c, x_n, y_b = x_c.to(device), x_n.to(device), y_b.to(device)
                opt_ft.zero_grad()
                with torch.cuda.amp.autocast():
                    out = model_ft(x_c, x_n)
                    loss = criterion(out, y_b)
                scaler.scale(loss).backward()
                scaler.step(opt_ft)
                scaler.update()
                sched_ft.step()
                
            model_ft.eval()
            val_preds = []
            with torch.no_grad():
                for x_c, x_n, _ in va_loader:
                    x_c, x_n = x_c.to(device), x_n.to(device)
                    with torch.cuda.amp.autocast():
                        val_preds.append(torch.sigmoid(model_ft(x_c, x_n)).cpu().numpy())
            val_p = np.concatenate(val_preds)
            auc = roc_auc_score(y[va_idx], val_p)
            if auc > best_auc_ft:
                best_auc_ft = auc
                best_weights_ft = {k: v.cpu() for k, v in model_ft.state_dict().items()}
                
        model_ft.load_state_dict({k: v.to(device) for k, v in best_weights_ft.items()})
        model_ft.eval()
        
        val_preds = []
        with torch.no_grad():
            for x_c, x_n, _ in va_loader:
                x_c, x_n = x_c.to(device), x_n.to(device)
                with torch.cuda.amp.autocast():
                    val_preds.append(torch.sigmoid(model_ft(x_c, x_n)).cpu().numpy())
        oof_ft_seed[va_idx] = np.concatenate(val_preds)
        
        test_preds = []
        with torch.no_grad():
            for x_c, x_n in te_loader:
                x_c, x_n = x_c.to(device), x_n.to(device)
                with torch.cuda.amp.autocast():
                    test_preds.append(torch.sigmoid(model_ft(x_c, x_n)).cpu().numpy())
        test_ft_seed += np.concatenate(test_preds) / Config.N_FOLDS
        
        del model_ft, opt_ft, sched_ft
        torch.cuda.empty_cache()
        gc.collect()
        
        fold_elapsed = time.time() - fold_t0
        print(f"Fold {fold+1}/5 Done [{fold_elapsed:.1f}s] -> ResNet AUC: {best_auc_rn:.5f} | FT-Trans AUC: {best_auc_ft:.5f}")
        
    auc_rn_seed = roc_auc_score(y, oof_resnet_seed)
    auc_ft_seed = roc_auc_score(y, oof_ft_seed)
    print(f"--- SEED {SEED} FINAL: ResNet OOF AUC: {auc_rn_seed:.6f} | FT-Trans OOF AUC: {auc_ft_seed:.6f} ---")
    
    oof_resnet_seeds.append(oof_resnet_seed)
    test_resnet_seeds.append(test_resnet_seed)
    oof_ft_seeds.append(oof_ft_seed)
    test_ft_seeds.append(test_ft_seed)

# Compute 3-Seed Bagged Predictions
oof_resnet_bag3 = np.mean(oof_resnet_seeds, axis=0)
test_resnet_bag3 = np.mean(test_resnet_seeds, axis=0)

oof_ft_bag3 = np.mean(oof_ft_seeds, axis=0)
test_ft_bag3 = np.mean(test_ft_seeds, axis=0)

# Dual-Neural 6-Model Master Bag (60% FT + 40% ResNet on Ranks)
r_ft = rankdata(oof_ft_bag3) / N_TR
r_rn = rankdata(oof_resnet_bag3) / N_TR
oof_dual_bag = 0.6 * r_ft + 0.4 * r_rn

r_ft_te = rankdata(test_ft_bag3) / N_TE
r_rn_te = rankdata(test_resnet_bag3) / N_TE
test_dual_bag = 0.6 * r_ft_te + 0.4 * r_rn_te

print("=" * 75)
print("🏆 FINAL MULTI-SEED ENSEMBLE EVALUATIONS (5-FOLD CV)")
print("=" * 75)
print(f"ResNet 3-Seed Bag OOF ROC-AUC       : {roc_auc_score(y, oof_resnet_bag3):.6f}")
print(f"FT-Transformer 3-Seed Bag OOF ROC-AUC: {roc_auc_score(y, oof_ft_bag3):.6f}")
print(f"Dual-Neural 6-Model Bag OOF ROC-AUC : {roc_auc_score(y, oof_dual_bag):.6f}")
print(f"Dual-Neural 6-Model Bag OOF PR-AUC  : {average_precision_score(y, oof_dual_bag):.6f}")

# Save Outputs
np.save(Config.OUTPUT_DIR / "oof_resnet_bag3.npy", oof_resnet_bag3)
np.save(Config.OUTPUT_DIR / "test_resnet_bag3.npy", test_resnet_bag3)

np.save(Config.OUTPUT_DIR / "oof_ft_transformer_bag3.npy", oof_ft_bag3)
np.save(Config.OUTPUT_DIR / "test_ft_transformer_bag3.npy", test_ft_bag3)

np.save(Config.OUTPUT_DIR / "oof_dual_neural_bag3.npy", oof_dual_bag)
np.save(Config.OUTPUT_DIR / "test_dual_neural_bag3.npy", test_dual_bag)

sub = sample_sub.copy()
sub["is_fraud"] = rankdata(test_dual_bag) / N_TE
sub_path = Config.OUTPUT_DIR / "dual_neural_bag3_submission.csv"
sub.to_csv(sub_path, index=False)
print(f"Generated Submission: {sub_path} | Shape: {sub.shape}")
"""))

# Build Notebook
nb.cells = cells
out_notebook = Path("pgs-s6e11/code/dual_neural_multiseed/model.ipynb")
with open(out_notebook, "w") as f:
    nbf.write(nb, f)

print(f"Successfully constructed {out_notebook} with {len(cells)} cells.")
