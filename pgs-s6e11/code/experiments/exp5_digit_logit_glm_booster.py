"""
================================================================================
Experiment 5: Megayak 84+ Key Target-Encoded Logit GLM + Shallow Tree Booster
================================================================================
Stage 1:
  - 84 1D digit keys (cents, int, ten, dec, leading digits, token keys)
  - 8 joint interaction keys (screened with Likelihood-Ratio G-test)
  - 5-fold inner Bayesian target encoding to logits
  - Generative decimal & token likelihood encoding
  - L2-Regularized Logistic Regression on logits (C=0.3) -> Stage 1 Margin z_glm (AUC ~0.9505)

Stage 2:
  - Shallow LightGBM (num_leaves=4, lr=0.03, colsample=0.5, early_stopping=300)
  - Boosted directly on residual from init_score = z_glm
  - Target Standalone CV: 0.95150 - 0.95170+
  - Calibrated with 5-Zone Zero-Fraud Manifold (ZALL_5)
================================================================================
"""

import os, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from tokenizers import Tokenizer

warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "dataset"
OUT_DIR = ROOT_DIR / "code" / "experiments" / "exp5_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TOKENIZER_PATH = ROOT_DIR / "external_models" / "artifacts" / "gpt2-tokenizer.json"

T0 = time.time()
def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)

def main():
    log("🚀 Starting Exp 5: Megayak Digit-Logit GLM + Shallow Tree Booster")

    # 1. Load Data
    tr = pd.read_csv(DATA_DIR / "train.csv")
    te = pd.read_csv(DATA_DIR / "test.csv")
    sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")

    TARGET = "is_fraud"
    CATS = ["merchant_category", "card_type", "auth_method", "channel", "device_type"]
    FLAGS = ["is_foreign_transaction", "is_new_merchant", "used_vpn",
             "ip_country_mismatch", "billing_shipping_mismatch", "is_ai_generated_scam_attempt"]
    NUMS = ["amount_usd", "hours_since_last_txn", "txn_count_last_24h", "distance_from_home_km",
            "card_age_months", "customer_age", "account_balance_usd", "cvv_retry_count",
            "velocity_score", "time_of_day_hour", "day_of_week", "merchant_risk_score", "prior_disputes"]
    FEATS = NUMS + FLAGS + CATS
    FLOATS = ["amount_usd", "hours_since_last_txn", "distance_from_home_km", "account_balance_usd",
              "velocity_score", "merchant_risk_score"]

    for df in (tr, te):
        df[FLAGS] = df[FLAGS].astype("int8")
    y = tr[TARGET].astype(int).values
    NTR, NTE = len(tr), len(te)
    full = pd.concat([tr[FEATS], te[FEATS]], ignore_index=True)
    FOLDS = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(NTR), y))

    # 2. Extract 1D Digit Keys
    log("Building 1D digit keys...")
    K = {}
    for c in FEATS:
        K[f"v_{c}"] = full[c].astype(str).values if c in CATS else full[c].values
    for c in FLOATS:
        x = full[c].values
        K[f"cents_{c}"] = (np.round(x * 100) % 100).astype(np.int64)
        K[f"int_{c}"] = np.floor(x).astype(np.int64)
        K[f"ten_{c}"] = np.floor(x / 10).astype(np.int64)
        K[f"dec_{c}"] = np.floor(x * 10 + 1e-6).astype(np.int64)
    for c in FLOATS:
        i = pd.Series(np.floor(full[c].values).astype(np.int64)).astype(str)
        K[f"l1_{c}"] = (i.str.len() * 10 + i.str[0].astype(int)).values
        K[f"l2_{c}"] = (i.str.len() * 100 + i.str[:2].astype(int)).values

    # 3. Extract Token Keys
    log("Extracting GPT-2 token keys...")
    gpt2 = Tokenizer.from_file(str(TOKENIZER_PATH))
    STR = pd.concat([pd.read_csv(DATA_DIR / "train.csv", usecols=FLOATS, dtype=str),
                     pd.read_csv(DATA_DIR / "test.csv", usecols=FLOATS, dtype=str)], ignore_index=True)
    TOK = {}
    for c in FLOATS:
        p = TOK[c] = STR[c].map({v: gpt2.encode(" " + v).tokens for v in STR[c].unique()})
        K[f"t1_{c}"] = p.str[0].values
        K[f"t2_{c}"] = p.map(lambda t: t[1] if len(t) > 1 else "").values
        K[f"tl_{c}"] = p.str[-1].values
        K[f"tn_{c}"] = p.str.len().values

    KNAMES = list(K)
    codes = [pd.factorize(pd.Series(K[n]))[0].astype(np.int32) for n in KNAMES]
    KMAT = np.column_stack(codes)
    NCAT = np.array([c.max() + 1 for c in codes])
    del K, codes

    # 4. Raw View Matrix
    RAW = full[NUMS + FLAGS].copy()
    for c in CATS:
        RAW[c] = pd.factorize(full[c], sort=True)[0].astype(np.int16)
    for c in FLOATS:
        RAW[f"cents_{c}"] = (np.round(full[c].values * 100) % 100).astype(np.int16)
    RAWV = RAW.values.astype(np.float32)
    CAT_IDX = [list(RAW.columns).index(c) for c in CATS]
    log(f"Base setup: {len(KNAMES)} digit/token keys, {RAW.shape[1]} raw columns")

    # 5. Joint Keys Screened with G-Test
    log("Evaluating joint interaction keys...")
    JOINT = [("j_category_risk", ("merchant_category", "merchant_risk_score")),
             ("j_foreign_ip", ("is_foreign_transaction", "ip_country_mismatch")),
             ("j_category_amount", ("merchant_category", "q:amount_usd")),
             ("j_txn_velocity", ("txn_count_last_24h", "q:velocity_score")),
             ("j_category_age", ("merchant_category", "customer_age")),
             ("j_amount_risk", ("q:amount_usd", "q:merchant_risk_score")),
             ("j_hours_velocity", ("q:hours_since_last_txn", "q:velocity_score")),
             ("j_age_risk", ("customer_age", "q:merchant_risk_score")),
             ("j_newmerchant_weekday", ("is_new_merchant", "day_of_week")),
             ("j_scam_velocity", ("is_ai_generated_scam_attempt", "q:velocity_score")),
             ("j_age_cardage", ("customer_age", "card_age_months")),
             ("j_hour_weekday", ("time_of_day_hour", "day_of_week")),
             ("j_txn_age", ("txn_count_last_24h", "customer_age"))]

    def joint_col(df, c):
        return pd.qcut(df[c[2:]], 10, labels=False, duplicates="drop") if c.startswith("q:") else df[c]

    def joint_code(df, cols):
        k = np.zeros(len(df), np.int64)
        for c in cols:
            u = pd.factorize(joint_col(df, c))[0]
            k = k * (u.max() + 1) + u
        return pd.factorize(k)[0]

    lgt1 = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    rows, JCODES = [], {}
    for name, cols in JOINT:
        JCODES[name] = joint_code(full, cols).astype(np.int32)
        k = JCODES[name][:NTR]
        add = np.full(NTR, -(len(cols) - 1) * lgt1(y.mean()))
        dof = 0
        for c in cols:
            u = pd.factorize(joint_col(full, c))[0][:NTR]
            add += lgt1(np.bincount(u, weights=y) / np.bincount(u))[u]
            dof += u.max()
        s, n = np.bincount(k, weights=y), np.bincount(k)
        r = np.bincount(k, weights=1 / (1 + np.exp(-add))) / n
        with np.errstate(divide="ignore", invalid="ignore"):
            g = np.where(s > 0, s * np.log(s / (n * r)), 0) + np.where(n > s, (n - s) * np.log((n - s) / (n * (1 - r))), 0)
        rows.append(dict(key=name, cells=(n > 0).sum(), G=2 * g.sum(), df=(n > 0).sum() - 1 - dof))

    GT = pd.DataFrame(rows).set_index("key")
    GT["kept"] = GT["G"] > 1.2 * GT["df"]
    USED = list(GT.index[GT["kept"]])
    KMAT = np.column_stack([KMAT] + [JCODES[n] for n in USED])
    NCAT = np.append(NCAT, [JCODES[n].max() + 1 for n in USED])
    KNAMES += USED
    log(f"Kept {len(USED)} joint keys -> Total {len(KNAMES)} target-encoded keys")

    # 6. Target Encoding & Generative Encoding Functions
    def te_rate(k, t, ncat, prior, smooth):
        s = np.bincount(k, weights=t, minlength=ncat)
        n = np.bincount(k, minlength=ncat)
        return (s + prior * smooth) / (n + smooth)

    def te_fold(a, b, smooth=20.0, inner=5, seed=0):
        ya = y[a].astype(np.float64)
        Ka, Kb, Kt = KMAT[a], KMAT[b], KMAT[NTR:]
        Ea = np.zeros(Ka.shape, np.float32)
        Eb = np.zeros(Kb.shape, np.float32)
        Et = np.zeros(Kt.shape, np.float32)
        inner_f = list(StratifiedKFold(inner, shuffle=True, random_state=seed).split(np.zeros(len(a)), ya))
        for j in range(KMAT.shape[1]):
            for i, o in inner_f:
                Ea[o, j] = te_rate(Ka[i, j], ya[i], NCAT[j], ya[i].mean(), smooth)[Ka[o, j]]
            r = te_rate(Ka[:, j], ya, NCAT[j], ya.mean(), smooth)
            Eb[:, j], Et[:, j] = r[Kb[:, j]], r[Kt[:, j]]
        return Ea, Eb, Et

    POOL2 = ["amount_usd", "hours_since_last_txn", "distance_from_home_km", "account_balance_usd"]
    POOL1 = ["velocity_score", "merchant_risk_score"]
    dec = lambda c: STR[c].str.split(".").str[1].astype(str)
    lit = lambda c: TOK[c].map(lambda t: t[t.index(".") - 1] if "." in t else t[-1])
    GROUPS = []
    for mem, fn in [(POOL2, dec), (POOL2, lambda c: dec(c).str[0]), (POOL1, dec), (POOL1, lit)]:
        v = pd.concat([fn(c) for c in mem], ignore_index=True)
        GROUPS.append(pd.factorize(v)[0].astype(np.int32).reshape(len(mem), -1).T.copy())
    MARKOV = [pd.factorize(lit(c) + "|" + dec(c))[0].astype(np.int32) for c in POOL2]

    def gen_encode(i, o, m0=50.0, m1=2000.0, m2=200.0):
        yi = y[i].astype(np.float64)
        prior, cols, field0 = yi.mean(), [], []
        for g, C in enumerate(GROUPS):
            ncat = int(C.max()) + 1
            s = [np.bincount(C[i, j], weights=yi, minlength=ncat) for j in range(C.shape[1])]
            n = [np.bincount(C[i, j], minlength=ncat) for j in range(C.shape[1])]
            pool = (sum(s) + prior * m0) / (sum(n) + m0)
            for j in range(C.shape[1]):
                h = ((s[j] + m1 * pool) / (n[j] + m1))[C[o, j]]
                cols.append(h)
                if g == 0:
                    field0.append(h)
        for j, M in enumerate(MARKOV):
            ncat = int(M.max()) + 1
            s, n = np.bincount(M[i], weights=yi, minlength=ncat), np.bincount(M[i], minlength=ncat)
            cols.append((s[M[o]] + m2 * field0[j]) / (n[M[o]] + m2))
        return np.column_stack(cols).astype(np.float32)

    def gen_fold(a, b, inner=5, seed=0):
        Ga = None
        for i, o in StratifiedKFold(inner, shuffle=True, random_state=seed).split(np.zeros(len(a)), y[a]):
            E = gen_encode(a[i], a[o])
            Ga = np.zeros((len(a), E.shape[1]), np.float32) if Ga is None else Ga
            Ga[o] = E
        E = gen_encode(a, np.concatenate([b, np.arange(NTR, NTR + NTE)]))
        return Ga, E[:len(b)], E[len(b):]

    ONEHOT = pd.get_dummies(full[CATS].astype(str), dtype=np.float32).values
    BASE = np.hstack([full[FLAGS].values.astype(np.float32), ONEHOT])
    KCNT = np.column_stack([np.bincount(KMAT[:, j])[KMAT[:, j]] for j in range(KMAT.shape[1])]).astype(np.float32)

    def logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def view(idx, E):
        return np.hstack([RAWV[idx], E, KCNT[idx]])

    LGB_PARAMS = dict(n_estimators=20000, learning_rate=0.03, num_leaves=4, min_child_samples=100, subsample=0.8,
                      subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0, max_bin=255, verbose=-1,
                      n_jobs=os.cpu_count(), random_state=42)

    # 7. 5-Fold Training
    log("--- Starting 5-Fold Stratified Training ---")
    oof_glm = np.zeros(NTR, dtype=np.float32)
    oof_boost = np.zeros(NTR, dtype=np.float32)
    test_boost = np.zeros(NTE, dtype=np.float32)
    its = []

    for f, (a, b) in enumerate(FOLDS):
        f_t0 = time.time()
        # Encodings
        Ea, Eb, Et = te_fold(a, b, smooth=20.0, seed=0)
        Ga, Gb, Gt = gen_fold(a, b, seed=0)
        Ga_, Gb_, Gt_ = np.hstack([Ea, Ga]), np.hstack([Eb, Gb]), np.hstack([Et, Gt])

        # Stage 1: GLM
        sc = StandardScaler().fit(logit(Ga_))
        Da = np.hstack([sc.transform(logit(Ga_)), BASE[a]])
        Db = np.hstack([sc.transform(logit(Gb_)), BASE[b]])
        Dt = np.hstack([sc.transform(logit(Gt_)), BASE[NTR:]])

        glm = LogisticRegression(C=0.3, max_iter=200, tol=1e-7, solver="newton-cholesky").fit(Da, y[a])
        za = glm.decision_function(Da)
        zb = glm.decision_function(Db)
        zt = glm.decision_function(Dt)
        oof_glm[b] = zb
        glm_auc = roc_auc_score(y[b], zb)

        # Stage 2: LightGBM Boosted on Margin
        Xa = view(a, Ea)
        Xb = view(b, Eb)
        Xt = view(np.arange(NTR, NTR + NTE), Et)

        m = lgb.LGBMClassifier(**LGB_PARAMS)
        m.fit(Xa, y[a], init_score=za, eval_set=[(Xb, y[b])], eval_init_score=[zb], eval_metric="auc",
              categorical_feature=CAT_IDX, callbacks=[lgb.early_stopping(300, verbose=False)])

        it = m.best_iteration_
        its.append(it)

        # Predictions
        oof_boost[b] = zb + m.predict(Xb, raw_score=True)
        test_boost += (zt + m.predict(Xt, raw_score=True)) / len(FOLDS)
        boost_auc = roc_auc_score(y[b], oof_boost[b])
        log(f"Fold {f}: GLM AUC {glm_auc:.5f} -> Boosted AUC {boost_auc:.5f} (+{(boost_auc - glm_auc)*1e5:.1f} pts, {it} trees, {time.time() - f_t0:.1f}s)")

    # 8. Calibration & Metrics
    # Compute 5-zone manifold
    def compute_manifold(df):
        clean = ((df["cvv_retry_count"] == 0) & (df["prior_disputes"] == 0) & 
                 (~df["is_new_merchant"].astype(bool)) & (~df["is_foreign_transaction"].astype(bool)) & 
                 (~df["used_vpn"].astype(bool)) & (~df["ip_country_mismatch"].astype(bool)) & 
                 (~df["billing_shipping_mismatch"].astype(bool)) & (~df["is_ai_generated_scam_attempt"].astype(bool))).to_numpy()
        z1 = df["merchant_risk_score"].to_numpy() < 13.1
        z2 = (df["velocity_score"].to_numpy() <= 4.1) & (df["merchant_risk_score"].to_numpy() <= 20.0)
        z34 = clean & (((df["velocity_score"].to_numpy() <= 4.1) & (df["merchant_risk_score"].to_numpy() <= 33.8)) |
                       ((df["velocity_score"].to_numpy() <= 6.0) & (df["merchant_risk_score"].to_numpy() <= 25.0)))
        z5 = clean & (df["velocity_score"].to_numpy() <= 5.0) & (df["merchant_risk_score"].to_numpy() <= 35.0) & (df["amount_usd"].to_numpy() <= 30.0)
        return z1 | z2 | z34 | z5

    m_tr = compute_manifold(tr)
    m_te = compute_manifold(te)

    def calibrate(pred, mask):
        cal = pred.copy()
        fmin = np.min(cal[~mask])
        cal[mask] = np.minimum(cal[mask], fmin - 1e-7)
        return cal

    glm_overall = roc_auc_score(y, oof_glm)
    boost_raw = roc_auc_score(y, oof_boost)
    oof_boost_cal = calibrate(oof_boost, m_tr)
    boost_cal = roc_auc_score(y, oof_boost_cal)

    log("=" * 80)
    log(f"🏆 FINAL 5-FOLD CV RESULTS:")
    log(f"  Stage 1 GLM Standalone OOF AUC:        {glm_overall:.6f}")
    log(f"  Stage 2 Boosted Raw OOF AUC:           {boost_raw:.6f}")
    log(f"  Calibrated (+ 5-Zone Manifold) AUC:    {boost_cal:.6f} (+{(boost_cal - boost_raw)*1e5:.1f} pts)")
    log("=" * 80)

    # 9. Save Artifacts
    np.save(OUT_DIR / "oof_digit_glm_boost.npy", oof_boost.astype(np.float32))
    np.save(OUT_DIR / "oof_digit_glm_boost_cal.npy", oof_boost_cal.astype(np.float32))
    np.save(OUT_DIR / "test_digit_glm_boost.npy", test_boost.astype(np.float32))
    log(f"Saved OOF and Test predictions to {OUT_DIR}")

if __name__ == "__main__":
    main()
