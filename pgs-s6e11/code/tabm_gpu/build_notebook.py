#!/usr/bin/env python3
"""
Build Kaggle Notebook for TabM GPU Training (5-Fold Stratified CV)
Uses pytabkit TabM_D_Classifier on GPU with domain-engineered features:
Benford first digits, decimal extraction, interaction target encoding, and quantile binning.
"""

import json
from pathlib import Path

def create_notebook():
    cells = []

    # Cell 1: Install pytabkit
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "%%time\n",
            "!pip install -q pytabkit\n"
        ]
    })

    # Cell 2: Imports & Seeds
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os, random, time, warnings\n",
            "from importlib.metadata import version\n",
            "import numpy as np, pandas as pd\n",
            "import torch\n",
            "from pytabkit import TabM_D_Classifier\n",
            "from sklearn.metrics import roc_auc_score, average_precision_score\n",
            "from sklearn.model_selection import StratifiedKFold\n",
            "from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder\n",
            "from scipy.stats import rankdata\n",
            "\n",
            "warnings.filterwarnings('ignore')\n",
            "print('PyTorch version: ', torch.__version__)\n",
            "print('CUDA available:  ', torch.cuda.is_available())\n",
            "if torch.cuda.is_available():\n",
            "    print('GPU device:      ', torch.cuda.get_device_name(0))\n",
            "print('PyTabKit version:', version('pytabkit'))\n",
            "\n",
            "def seed_everything(seed: int = 42):\n",
            "    random.seed(seed)\n",
            "    np.random.seed(seed)\n",
            "    torch.manual_seed(seed)\n",
            "    if torch.cuda.is_available():\n",
            "        torch.cuda.manual_seed_all(seed)\n",
            "\n",
            "seed_everything(42)\n"
        ]
    })

    # Cell 3: Load Data
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "%%time\n",
            "train_path = '/kaggle/input/playground-series-s6e11/train.csv'\n",
            "test_path = '/kaggle/input/playground-series-s6e11/test.csv'\n",
            "\n",
            "train = pd.read_csv(train_path)\n",
            "test = pd.read_csv(test_path)\n",
            "print(f'Train shape: {train.shape}, Test shape: {test.shape}')\n",
            "\n",
            "ID = 'id'\n",
            "TARGET = 'is_fraud'\n",
            "\n",
            "train_id = train[ID]\n",
            "test_id = test[ID]\n",
            "y = train[TARGET].values.astype(int)\n",
            "\n",
            "X = train.drop(columns=[ID, TARGET])\n",
            "X_test = test.drop(columns=[ID])\n",
            "del train, test\n",
            "\n",
            "print(f'Fraud rate: {y.mean():.4%}')\n"
        ]
    })

    # Cell 4: Feature Engineering
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "%%time\n",
            "cat_cols = X.select_dtypes(include=['object', 'bool']).columns.tolist()\n",
            "num_cols = X.select_dtypes(exclude=['object', 'bool']).columns.tolist()\n",
            "print(f'Initial cat_cols: {len(cat_cols)}, num_cols: {len(num_cols)}')\n",
            "\n",
            "important_combos = [\n",
            "    ('merchant_category', 'card_type', 'txn_count_last_24h'),\n",
            "    ('merchant_category', 'customer_age'),\n",
            "]\n",
            "\n",
            "category_map = {}\n",
            "\n",
            "def feature_engineering(df, fit=False):\n",
            "    # 1. Decimal / Fractional extraction\n",
            "    for col in ['amount_usd', 'hours_since_last_txn', 'velocity_score', 'distance_from_home_km']:\n",
            "        if col in df.columns:\n",
            "            df[f'_{col}_decimal'] = (df[col] % 1).round(2).astype('float32')\n",
            "\n",
            "    # 2. Benford First Digit of amount\n",
            "    if 'amount_usd' in df.columns:\n",
            "        df['_amount_first_digit'] = df['amount_usd'].astype(str).str.extract(r'([1-9])')[0].fillna('0').astype(int).astype('category')\n",
            "        df['_is_round_dollar'] = ((df['amount_usd'] % 1).round(2) == 0).astype(int).astype('category')\n",
            "        df['_is_99_cents'] = ((df['amount_usd'] % 1).round(2) == 0.99).astype(int).astype('category')\n",
            "\n",
            "    # 3. Discretize numerical distributions into categories\n",
            "    for col in num_cols + ['_amount_usd_decimal', '_hours_since_last_txn_decimal', '_velocity_score_decimal', '_distance_from_home_km_decimal']:\n",
            "        if col in df.columns:\n",
            "            cat_name = f'{col}_cat_' if col in num_cols else f'{col[1:]}_cat_'\n",
            "            values = np.floor(df[col]) if col in num_cols else df[col]\n",
            "            if fit:\n",
            "                codes, uniques = values.factorize()\n",
            "                category_map[col] = uniques\n",
            "            else:\n",
            "                uniques = category_map[col]\n",
            "                code_map = {cat: i for i, cat in enumerate(uniques)}\n",
            "                codes = values.map(code_map).fillna(-1).astype('int32')\n",
            "            df[cat_name] = codes\n",
            "            df[cat_name] = df[cat_name].astype('category')\n",
            "\n",
            "    # 4. Quantile Binning\n",
            "    bin_config = {'merchant_risk_score': [7], 'hours_since_last_txn': [500, 1000]}\n",
            "    for col, bins_list in bin_config.items():\n",
            "        if col in df.columns:\n",
            "            for n_bins in bins_list:\n",
            "                bin_name = f'{col}_{n_bins}_quantile_bin_'\n",
            "                if fit:\n",
            "                    kb = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile', subsample=None)\n",
            "                    binned = kb.fit_transform(df[[col]]).ravel().astype('int32')\n",
            "                    category_map[bin_name] = kb\n",
            "                else:\n",
            "                    kb = category_map[bin_name]\n",
            "                    binned = kb.transform(df[[col]]).ravel().astype('int32')\n",
            "                df[bin_name] = binned\n",
            "                df[bin_name] = df[bin_name].astype('category')\n",
            "\n",
            "    # 5. High-order interaction categories\n",
            "    combo_names = []\n",
            "    for cols in important_combos:\n",
            "        combo_name = '_'.join(cols) + '_'\n",
            "        combo_names.append(combo_name)\n",
            "        combo_series = df[cols[0]].astype(str)\n",
            "        for col in cols[1:]: \n",
            "            combo_series = combo_series + '_' + df[col].astype(str)\n",
            "        if fit:\n",
            "            codes, uniques = pd.factorize(combo_series, sort=False)\n",
            "            category_map[combo_name] = uniques\n",
            "        else:\n",
            "            uniques = category_map[combo_name]\n",
            "            code_map = {cat: i for i, cat in enumerate(uniques)}\n",
            "            codes = combo_series.map(code_map).fillna(-1).astype('int32')\n",
            "        df[combo_name] = codes\n",
            "        df[combo_name] = df[combo_name].astype('category')\n",
            "\n",
            "    new_cat_cols = [col for col in df.columns if col.endswith('_')]\n",
            "    new_num_cols = [col for col in df.columns if col.startswith('_')]\n",
            "    return df, new_cat_cols, new_num_cols, combo_names\n",
            "\n",
            "X, new_cat_cols, new_num_cols, combo_names = feature_engineering(X, fit=True)\n",
            "X_test, _, _, _ = feature_engineering(X_test, fit=False)\n",
            "cat_cols += new_cat_cols\n",
            "num_cols += new_num_cols\n",
            "print(f'Feature engineering complete: X shape={X.shape}, X_test={X_test.shape}')\n"
        ]
    })

    # Cell 5: Stratified 5-Fold Training Loop
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "%%time\n",
            "FOLDS = 5\n",
            "SEED = 42\n",
            "\n",
            "skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)\n",
            "oof_preds = np.zeros(len(X), dtype=np.float64)\n",
            "test_preds = np.zeros(len(X_test), dtype=np.float64)\n",
            "\n",
            "tabm_params = {\n",
            "    'device': 'cuda' if torch.cuda.is_available() else 'cpu',\n",
            "    'random_state': SEED,\n",
            "    'val_metric_name': '1-auc_ovr',\n",
            "    'tabm_k': 16,\n",
            "    'batch_size': 512,\n",
            "    'lr': 0.002,\n",
            "    'weight_decay': 0.01,\n",
            "    'n_epochs': 12,\n",
            "    'patience': 4,\n",
            "    'allow_amp': True,\n",
            "    'verbosity': 2,\n",
            "}\n",
            "\n",
            "te_cols = combo_names + ['hours_since_last_txn']\n",
            "\n",
            "for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):\n",
            "    t0 = time.time()\n",
            "    print('=' * 60)\n",
            "    print(f'>>> Fold {fold}/{FOLDS} TabM Training')\n",
            "    print('=' * 60)\n",
            "\n",
            "    X_tr = X.iloc[tr_idx].copy()\n",
            "    X_val = X.iloc[val_idx].copy()\n",
            "    y_tr = y[tr_idx]\n",
            "    y_val = y[val_idx]\n",
            "    X_tst = X_test.copy()\n",
            "\n",
            "    # Out-of-fold Target Encoding on key interactions\n",
            "    TE = TargetEncoder(cv=FOLDS, smooth='auto', shuffle=True, random_state=SEED + fold)\n",
            "    tr_enc = TE.fit_transform(X_tr[te_cols], y_tr)\n",
            "    val_enc = TE.transform(X_val[te_cols])\n",
            "    tst_enc = TE.transform(X_tst[te_cols])\n",
            "\n",
            "    te_names = [f'_{col}_TE' for col in te_cols]\n",
            "    X_tr[te_names] = tr_enc.astype('float32')\n",
            "    X_val[te_names] = val_enc.astype('float32')\n",
            "    X_tst[te_names] = tst_enc.astype('float32')\n",
            "\n",
            "    print(f'Train features: {X_tr.shape[1]} columns')\n",
            "\n",
            "    # Instantiate and fit TabM\n",
            "    fold_params = dict(tabm_params)\n",
            "    fold_params['random_state'] = SEED + fold\n",
            "\n",
            "    model = TabM_D_Classifier(**fold_params)\n",
            "    model.fit(X_tr, y_tr, X_val, y_val)\n",
            "\n",
            "    val_preds = model.predict_proba(X_val)[:, 1]\n",
            "    fold_test_preds = model.predict_proba(X_tst)[:, 1]\n",
            "\n",
            "    oof_preds[val_idx] = val_preds\n",
            "    test_preds += fold_test_preds / FOLDS\n",
            "\n",
            "    fold_auc = roc_auc_score(y_val, val_preds)\n",
            "    fold_prauc = average_precision_score(y_val, val_preds)\n",
            "    elapsed = time.time() - t0\n",
            "    print(f'Fold {fold} finished in {elapsed:.1f}s | ROC-AUC: {fold_auc:.6f} | PR-AUC: {fold_prauc:.6f}')\n",
            "\n",
            "    if torch.cuda.is_available():\n",
            "        torch.cuda.empty_cache()\n"
        ]
    })

    # Cell 6: Overall Metrics & Export
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Compute Overall Performance & Export\n",
            "overall_auc = roc_auc_score(y, oof_preds)\n",
            "overall_prauc = average_precision_score(y, oof_preds)\n",
            "\n",
            "print('=' * 60)\n",
            "print(f'>>> Overall TabM 5-Fold OOF ROC-AUC: {overall_auc:.6f}')\n",
            "print(f'>>> Overall TabM 5-Fold OOF PR-AUC:  {overall_prauc:.6f}')\n",
            "print('=' * 60)\n",
            "\n",
            "np.save('oof_tabm.npy', oof_preds.astype(np.float32))\n",
            "np.save('test_tabm.npy', test_preds.astype(np.float32))\n",
            "\n",
            "# Generate rank-scaled submission\n",
            "r_test = (rankdata(test_preds) - 0.5) / len(test_preds)\n",
            "sub = pd.DataFrame({'id': test_id, 'is_fraud': r_test})\n",
            "sub.to_csv('tabm_submission.csv', index=False)\n",
            "\n",
            "print('Saved files:')\n",
            "print('  oof_tabm.npy:       ', os.path.getsize('oof_tabm.npy') / (1024 * 1024), 'MB')\n",
            "print('  test_tabm.npy:      ', os.path.getsize('test_tabm.npy') / (1024 * 1024), 'MB')\n",
            "print('  tabm_submission.csv:', os.path.getsize('tabm_submission.csv') / (1024 * 1024), 'MB')\n",
            "print('Done!')\n"
        ]
    })

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    out_file = Path(__file__).resolve().parent / "model.ipynb"
    with open(out_file, "w") as f:
        json.dump(notebook, f, indent=2)
    print(f"Generated notebook at: {out_file}")

if __name__ == "__main__":
    create_notebook()
