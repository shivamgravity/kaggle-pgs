import nbformat as nbf
from pathlib import Path
import json

nb = nbf.v4.new_notebook()
cells = []

# Cell 0: Markdown Header
cells.append(nbf.v4.new_markdown_cell('''# 🧠 PGS-S6E11: Tabular Deep ResNet with Learned Entity Embeddings
### Kaggle Playground Series Season 6 Episode 11: Credit Card Fraud Detection

**Why a Tabular Neural Network is a Strategic Game-Changer:**
1. **Orthogonal Inductive Bias**:
   - Decision trees partition feature space using axis-aligned orthogonal splits.
   - Deep neural networks construct continuous, curved hyperspherical decision surfaces.
   - In Kaggle tabular competitions, blending neural networks with GBDT ensembles provides massive leaderboard gains because their error residuals are largely uncorrelated (Spearman rank correlation ~`0.92 - 0.95`).
2. **Learned Entity Embeddings for Digit Artifacts & Categoricals**:
   - Rather than one-hot encoding or scalar target encoding, the network learns a continuous latent representation for:
     - `cents` (100 distinct cents values mapped to 12D embedding)
     - `first_decimal` & `last_digit` (10 values each mapped to 4D embeddings)
     - Categorical features (`merchant_category`, `card_type`, `auth_method`, `channel`, `device_type`)
     - Temporal keys (`time_of_day_hour`, `day_of_week`)
3. **Deep Residual MLP Architecture**:
   - Quantile-normalized continuous and velocity features.
   - Pre-activation residual blocks with LayerNorm, Swish (SiLU) activations, and Dropout regularization.
   - Trained with PyTorch on Kaggle GPU using `AdamW` and `OneCycleLR` across 5 stratified folds.
'''))

# Cell 1: Imports & GPU Check
cells.append(nbf.v4.new_code_cell('''import os
import gc
import time
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler, QuantileTransformer

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch Version: {torch.__version__} | Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} (Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB)")
'''))

# Cell 2: Config
cells.append(nbf.v4.new_code_cell('''class Config:
    SEED = 42
    N_FOLDS = 5
    BATCH_SIZE = 1024
    EPOCHS = 18
    LR = 2e-3
    WEIGHT_DECAY = 1e-4
    PATIENCE = 5
    
    # Path resolution
    if os.path.exists("/kaggle/input/playground-series-s6e11"):
        DATA_DIR = Path("/kaggle/input/playground-series-s6e11")
        OUTPUT_DIR = Path(".")
    elif os.path.exists("/kaggle/input/competitions/playground-series-s6e11"):
        DATA_DIR = Path("/kaggle/input/competitions/playground-series-s6e11")
        OUTPUT_DIR = Path(".")
    else:
        DATA_DIR = Path("../../dataset")
        if not (DATA_DIR / "train.csv").exists():
            DATA_DIR = Path("pgs-s6e11/dataset")
        OUTPUT_DIR = Path(".")

print(f"Config initialized: {Config.N_FOLDS} Folds, Batch: {Config.BATCH_SIZE}, Epochs: {Config.EPOCHS}")
'''))

# Cell 3: Feature Engineering & Preprocessing
cells.append(nbf.v4.new_code_cell('''train_df = pd.read_csv(Config.DATA_DIR / "train.csv")
test_df = pd.read_csv(Config.DATA_DIR / "test.csv")
sample_sub = pd.read_csv(Config.DATA_DIR / "sample_submission.csv")

TARGET, ID_COL = "is_fraud", "id"
FLAG_COLS = list(train_df.select_dtypes(include="bool").columns)

for df in (train_df, test_df):
    df[FLAG_COLS] = df[FLAG_COLS].astype("int8")

y = train_df[TARGET].astype(int).values

# 1. Velocity & Burst Features
for df in (train_df, test_df):
    df["spend_velocity"] = df["amount_usd"] / (df["hours_since_last_txn"] + 0.05)
    df["travel_speed_kmh"] = df["distance_from_home_km"] / (df["hours_since_last_txn"] + 0.05)
    df["balance_drain_ratio"] = df["amount_usd"] / (df["account_balance_usd"] + 1.0)
    df["amount_24h_volume"] = df["amount_usd"] * df["txn_count_last_24h"]
    df["velocity_x_amount"] = df["velocity_score"] * df["amount_usd"]
    df["cvv_x_risk"] = df["cvv_retry_count"] * df["merchant_risk_score"]
    df["total_flags"] = df[FLAG_COLS].sum(axis=1)
    df["flag_x_amount"] = df["total_flags"] * df["amount_usd"]
    df["fraud_risk_index"] = df["total_flags"] * (df["velocity_score"] + 1.0) * (df["cvv_retry_count"] + 1)

# Category relative amounts
cat_medians = train_df.groupby("merchant_category")["amount_usd"].median()
cat_means = train_df.groupby("merchant_category")["amount_usd"].mean()
cat_stds = train_df.groupby("merchant_category")["amount_usd"].std()

for df in (train_df, test_df):
    df["amount_to_cat_median"] = df["amount_usd"] / (df["merchant_category"].map(cat_medians) + 1.0)
    df["amount_cat_zscore"] = (df["amount_usd"] - df["merchant_category"].map(cat_means)) / (df["merchant_category"].map(cat_stds) + 1e-5)

# 2. Digit Extraction for Embeddings
for df in (train_df, test_df):
    df["cents"] = np.round((df["amount_usd"] % 1) * 100).astype(int)
    df["first_decimal"] = np.floor((df["amount_usd"] % 1) * 10).astype(int)
    df["last_digit"] = (np.floor(df["amount_usd"]).astype(int)) % 10

# Categorical and Embedding specs
EMB_SPECS = {
    "merchant_category": 12,
    "card_type": 4,
    "auth_method": 4,
    "channel": 4,
    "device_type": 4,
    "time_of_day_hour": 8,
    "day_of_week": 4,
    "cents": 16,
    "first_decimal": 4,
    "last_digit": 4,
}

both = pd.concat([train_df, test_df], ignore_index=True)
cat_maps = {}
for col in EMB_SPECS:
    cats = sorted(both[col].dropna().unique())
    cat_map = {val: i for i, val in enumerate(cats)}
    cat_maps[col] = cat_map
    train_df[col] = train_df[col].map(cat_map).fillna(0).astype(int)
    test_df[col] = test_df[col].map(cat_map).fillna(0).astype(int)

del both; gc.collect()

VELOCITY_COLS = [
    "spend_velocity", "travel_speed_kmh", "balance_drain_ratio",
    "amount_24h_volume", "velocity_x_amount", "cvv_x_risk",
    "total_flags", "flag_x_amount", "fraud_risk_index",
    "amount_to_cat_median", "amount_cat_zscore"
]

NUM_COLS = [
    "amount_usd", "hours_since_last_txn", "txn_count_last_24h",
    "distance_from_home_km", "card_age_months", "customer_age",
    "account_balance_usd", "cvv_retry_count", "velocity_score",
    "merchant_risk_score", "prior_disputes"
] + VELOCITY_COLS

CONTINUOUS_COLS = NUM_COLS + FLAG_COLS
EMBEDDING_COLS = list(EMB_SPECS.keys())

print(f"Continuous features: {len(CONTINUOUS_COLS)}, Embedding features: {len(EMBEDDING_COLS)}")
for col in EMBEDDING_COLS:
    print(f"  {col:20s}: {len(cat_maps[col])} categories -> dim {EMB_SPECS[col]}")
'''))

# Cell 4: PyTorch Dataset & Tabular ResNet Model Architecture
cells.append(nbf.v4.new_code_cell('''class TabularDataset(Dataset):
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

class ResNetBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.ln1 = nn.LayerNorm(out_dim)
        self.act1 = nn.SiLU()
        self.drop1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.ln2 = nn.LayerNorm(out_dim)
        self.act2 = nn.SiLU()
        self.drop2 = nn.Dropout(dropout)
        
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        res = self.skip(x)
        h = self.drop1(self.act1(self.ln1(self.fc1(x))))
        h = self.drop2(self.act2(self.ln2(self.fc2(h))))
        return h + res

class TabularResNet(nn.Module):
    def __init__(self, emb_cardinalities, emb_dims, num_cont_features, hidden_dims=[256, 128, 64], dropout=0.2):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_classes, dim)
            for num_classes, dim in zip(emb_cardinalities, emb_dims)
        ])
        
        total_emb_dim = sum(emb_dims)
        total_in_dim = total_emb_dim + num_cont_features
        
        self.input_layer = nn.Sequential(
            nn.Linear(total_in_dim, hidden_dims[0]),
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
        x_emb = torch.cat(emb_outs, dim=1)
        x = torch.cat([x_emb, x_num], dim=1)
        
        h = self.input_layer(x)
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        out = self.head(h).squeeze(-1)
        return out

print("TabularResNet Architecture defined successfully.")
'''))

# Cell 5: 5-Fold Training Loop
cells.append(nbf.v4.new_code_cell('''N_TR = len(train_df)
N_TE = len(test_df)
skf = StratifiedKFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)

oof_preds = np.zeros(N_TR, dtype=np.float32)
test_preds = np.zeros(N_TE, dtype=np.float32)

cards = [len(cat_maps[col]) for col in EMBEDDING_COLS]
dims = [EMB_SPECS[col] for col in EMBEDDING_COLS]
cat_train = train_df[EMBEDDING_COLS].values
cat_test = test_df[EMBEDDING_COLS].values

print("=" * 75)
print("Starting 5-Fold Deep Tabular ResNet Training with Entity Embeddings...")
print("=" * 75)

total_start = time.time()

for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df, y)):
    fold_t0 = time.time()
    
    # 1. Normalize continuous features per fold using QuantileTransformer + StandardScaler
    qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal", subsample=150_000, random_state=Config.SEED)
    sc = StandardScaler()
    
    num_tr = sc.fit_transform(qt.fit_transform(train_df.iloc[tr_idx][CONTINUOUS_COLS].values))
    num_va = sc.transform(qt.transform(train_df.iloc[va_idx][CONTINUOUS_COLS].values))
    num_te = sc.transform(qt.transform(test_df[CONTINUOUS_COLS].values))
    
    tr_dataset = TabularDataset(cat_train[tr_idx], num_tr, y[tr_idx])
    va_dataset = TabularDataset(cat_train[va_idx], num_va, y[va_idx])
    te_dataset = TabularDataset(cat_test, num_te)
    
    tr_loader = DataLoader(tr_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=True, num_workers=2)
    va_loader = DataLoader(va_dataset, batch_size=Config.BATCH_SIZE * 2, shuffle=False, pin_memory=True, num_workers=2)
    te_loader = DataLoader(te_dataset, batch_size=Config.BATCH_SIZE * 2, shuffle=False, pin_memory=True, num_workers=2)
    
    # 2. Instantiate Model
    torch.manual_seed(Config.SEED + fold)
    model = TabularResNet(cards, dims, len(CONTINUOUS_COLS), hidden_dims=[256, 128, 64], dropout=0.2).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    
    total_steps = len(tr_loader) * Config.EPOCHS
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=Config.LR, total_steps=total_steps,
        pct_start=0.15, anneal_strategy="cos"
    )
    
    best_auc = 0.0
    best_state = None
    no_improve = 0
    
    for epoch in range(1, Config.EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for x_c, x_n, targets in tr_loader:
            x_c, x_n, targets = x_c.to(device), x_n.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(x_c, x_n)
            loss = criterion(logits, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item() * len(targets)
        
        # Validation
        model.eval()
        va_preds = []
        with torch.no_grad():
            for x_c, x_n, _ in va_loader:
                x_c, x_n = x_c.to(device), x_n.to(device)
                logits = model(x_c, x_n)
                va_preds.append(torch.sigmoid(logits).cpu().numpy())
        
        va_preds = np.concatenate(va_preds)
        val_auc = roc_auc_score(y[va_idx], va_preds)
        val_pr = average_precision_score(y[va_idx], va_preds)
        
        if val_auc > best_auc:
            best_auc = val_auc
            best_pr = val_pr
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= Config.PATIENCE:
                break
    
    # Load Best Model for OOF and Test Inference
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    
    # Compute Final OOF
    va_preds = []
    with torch.no_grad():
        for x_c, x_n, _ in va_loader:
            x_c, x_n = x_c.to(device), x_n.to(device)
            va_preds.append(torch.sigmoid(model(x_c, x_n)).cpu().numpy())
    oof_preds[va_idx] = np.concatenate(va_preds)
    
    # Compute Test Predictions
    fold_te_preds = []
    with torch.no_grad():
        for x_c, x_n in te_loader:
            x_c, x_n = x_c.to(device), x_n.to(device)
            fold_te_preds.append(torch.sigmoid(model(x_c, x_n)).cpu().numpy())
    test_preds += np.concatenate(fold_te_preds) / Config.N_FOLDS
    
    print(f"Fold {fold+1}/{Config.N_FOLDS} ({time.time()-fold_t0:.1f}s) | Best Val ROC-AUC: {best_auc:.6f} | PR-AUC: {best_pr:.6f}")
    del model, optimizer, scheduler, tr_loader, va_loader, te_loader; gc.collect()
    torch.cuda.empty_cache()

overall_auc = roc_auc_score(y, oof_preds)
overall_pr = average_precision_score(y, oof_preds)
print("=" * 75)
print(f"Deep Tabular ResNet 5-Fold CV Completed in {(time.time()-total_start)/60:.2f} mins.")
print(f"Overall OOF ROC-AUC: {overall_auc:.6f} | OOF PR-AUC: {overall_pr:.6f}")
print("=" * 75)
'''))

# Cell 6: Save Artifacts and Generate Submission
cells.append(nbf.v4.new_code_cell('''# Save Out-Of-Fold predictions
np.save(Config.OUTPUT_DIR / "oof_tabular_nn.npy", oof_preds)

# Generate Tabular Neural Network Submission
sub_nn = sample_sub.copy()
sub_nn["is_fraud"] = test_preds
sub_nn.to_csv(Config.OUTPUT_DIR / "tabular_nn_submission.csv", index=False)

print(f"Saved: oof_tabular_nn.npy and tabular_nn_submission.csv in {Config.OUTPUT_DIR}")

# Plot ROC & PR Curves
from sklearn.metrics import roc_curve, precision_recall_curve

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

fpr, tpr, _ = roc_curve(y, oof_preds)
ax1.plot(fpr, tpr, label=f"Tabular ResNet (AUC = {overall_auc:.5f})", color="royalblue", lw=2)
ax1.plot([0, 1], [0, 1], "k--", alpha=0.5)
ax1.set_title("Neural Network ROC Curve", fontsize=12, fontweight="bold")
ax1.set_xlabel("False Positive Rate")
ax1.set_ylabel("True Positive Rate")
ax1.legend(loc="lower right")
ax1.grid(True, alpha=0.3)

p_curve, r_curve, _ = precision_recall_curve(y, oof_preds)
ax2.plot(r_curve, p_curve, label=f"Tabular ResNet (PR = {overall_pr:.5f})", color="crimson", lw=2)
ax2.set_title("Neural Network Precision-Recall Curve", fontsize=12, fontweight="bold")
ax2.set_xlabel("Recall")
ax2.set_ylabel("Precision")
ax2.legend(loc="upper right")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(Config.OUTPUT_DIR / "tabular_nn_evaluation_curves.png", dpi=200)
plt.show()
'''))

nb['cells'] = cells
nb['metadata'] = {
    'kernelspec': {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'python3'
    },
    'language_info': {
        'name': 'python',
        'version': '3.10.12'
    }
}

# Strip outputs/execution_count from markdown cells and validate
for cell in nb['cells']:
    if cell['cell_type'] == 'markdown':
        cell.pop('outputs', None)
        cell.pop('execution_count', None)

nbf.validate(nb)

target_path = Path('pgs-s6e11/code/tabular_nn_boost/model.ipynb')
with open(target_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f'Successfully generated and validated {target_path}')

# Write kernel-metadata.json
meta = {
    'id': 'shivamgravity/pgs-s6e11-tabular-nn-boost',
    'title': 'PGS S6E11 Tabular NN Boost',
    'code_file': 'model.ipynb',
    'language': 'python',
    'kernel_type': 'notebook',
    'is_private': True,
    'enable_gpu': True,
    'enable_tpu': False,
    'enable_internet': False,
    'dataset_sources': [],
    'competition_sources': ['playground-series-s6e11'],
    'kernel_sources': []
}

meta_path = Path('pgs-s6e11/code/tabular_nn_boost/kernel-metadata.json')
with open(meta_path, 'w', encoding='utf-8') as f:
    json.dump(meta, f, indent=2)

print(f'Successfully wrote metadata: {meta_path}')
