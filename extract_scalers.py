# -*- coding: utf-8 -*-
"""
Extract exact StandardScaler parameters via linear regression and save model metadata.
Also copy all model files to the models/ directory.

Run once to prepare everything needed for the web prediction app.
"""
import pandas as pd
import numpy as np
import pickle
import os
import shutil
from sklearn.linear_model import LinearRegression

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
SCALER_DIR = os.path.join(BASE, 'scalers')
MODEL_DIR = os.path.join(BASE, 'models')
os.makedirs(SCALER_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

DATA_8 = os.path.join(ROOT, '数据', '8_合并的数据.xlsx')

def recover_scaler_params(df_raw, df_enc, target_raw_idx, target_enc_idx, cat_prefixes, sheet_label):
    """Recover exact StandardScaler (mu, sigma) from raw+encoded training data using linear regression."""
    raw_cols = list(df_raw.columns)
    enc_cols = list(df_enc.columns)

    # Identify numerical feature columns in raw data (exclude target and categorical)
    num_raw_cols = []
    num_raw_indices = []
    for i, c in enumerate(raw_cols):
        if i == target_raw_idx:
            continue
        if df_raw[c].dtype in ['float64', 'int64']:
            num_raw_cols.append(c)
            num_raw_indices.append(i)

    # Identify numerical columns in encoded data (same order as raw)
    enc_num_cols = []
    for c in enc_cols:
        if c == enc_cols[target_enc_idx]:  # skip target
            continue
        is_cat = any(c.startswith(p) for p in cat_prefixes)
        if not is_cat:
            enc_num_cols.append(c)

    # Recover (mu, sigma) via linear regression
    means, stds = [], []
    for i in range(len(num_raw_cols)):
        raw_vals = df_raw[num_raw_cols[i]].values.reshape(-1, 1)
        enc_vals = df_enc[enc_num_cols[i]].values
        lr = LinearRegression().fit(raw_vals, enc_vals)
        sigma = 1.0 / lr.coef_[0]
        mu = -lr.intercept_ / lr.coef_[0]
        means.append(mu)
        stds.append(sigma)

    # Verify
    num_data = df_raw[num_raw_cols].values.astype(np.float64)
    scaled = (num_data - np.array(means)) / np.array(stds)
    enc_data_verify = df_enc[enc_num_cols].values.astype(np.float64)
    max_diff = np.abs(scaled - enc_data_verify).max()
    print(f'  [{sheet_label}] Scaler recovered: {len(num_raw_cols)} num features, max diff = {max_diff:.8f}')

    # Categorical feature info
    cat_cols_raw = []
    cat_values = {}
    for i, c in enumerate(raw_cols):
        if i == target_raw_idx:
            continue
        if df_raw[c].dtype not in ['float64', 'int64']:
            cat_cols_raw.append(c)
            cat_values[c] = sorted(df_raw[c].dropna().unique().tolist())

    # Target scaler (for NN inverse transform)
    target_raw = df_raw[raw_cols[target_raw_idx]].values.astype(np.float64)
    target_mean = target_raw.mean()
    target_std = target_raw.std(ddof=0)

    # One-hot encoded feature order (as the NN model expects)
    enc_feature_order = [c for i, c in enumerate(enc_cols) if i != target_enc_idx]

    # CatBoost feature order = raw column order, excluding target
    cb_feature_order = [c for i, c in enumerate(raw_cols) if i != target_raw_idx]

    # CatBoost categorical feature names (for reference, .cbm stores this internally)
    cb_cat_features = [c for c in cb_feature_order
                       if df_raw[c].dtype not in ['float64', 'int64']]

    return {
        'scaler_mean': np.array(means),
        'scaler_std': np.array(stds),
        'numerical_features': num_raw_cols,
        'categorical_features': cat_cols_raw,
        'categorical_values': cat_values,
        'target_mean': target_mean,
        'target_std': target_std,
        'encoded_feature_order': enc_feature_order,
        'cb_feature_order': cb_feature_order,
        'target_raw_idx': target_raw_idx,
        'target_encoded_idx': target_enc_idx,
        'n_features': len(enc_feature_order)
    }

# ==========================================================
# 1. AL dosage scaler
# ==========================================================
print('=== AL Dosage Scaler ===')
df_al_raw = pd.read_excel(DATA_8, sheet_name='AL')
df_al_enc = pd.read_excel(DATA_8, sheet_name='AL_编码归一化')
al_pkg = recover_scaler_params(
    df_al_raw, df_al_enc,
    target_raw_idx=8, target_enc_idx=7,
    cat_prefixes=['水厂类型', '水厂消毒类型'],
    sheet_label='AL'
)
with open(os.path.join(SCALER_DIR, 'al_scaler.pkl'), 'wb') as f:
    pickle.dump(al_pkg, f)
print(f'  Saved al_scaler.pkl')

# ==========================================================
# 2. FE dosage scaler
# ==========================================================
print('\n=== FE Dosage Scaler ===')
df_fe_raw = pd.read_excel(DATA_8, sheet_name='FE')
df_fe_enc = pd.read_excel(DATA_8, sheet_name='FE_编码归一化')
fe_pkg = recover_scaler_params(
    df_fe_raw, df_fe_enc,
    target_raw_idx=8, target_enc_idx=7,
    cat_prefixes=['水厂类型', '水厂消毒类型', '水源类别'],
    sheet_label='FE'
)
with open(os.path.join(SCALER_DIR, 'fe_scaler.pkl'), 'wb') as f:
    pickle.dump(fe_pkg, f)
print(f'  Saved fe_scaler.pkl')

# ==========================================================
# 3. Turbidity classification metadata
# ==========================================================
print('\n=== Turbidity Classification Metadata ===')
TURB_DIR = os.path.join(ROOT, '2-出水浊度')
turb_meta = {}
for ct in ['AL', 'FE']:
    df_t = pd.read_excel(os.path.join(TURB_DIR, '出水浊度占比8.xlsx'), sheet_name=ct)
    cols_t = list(df_t.columns)
    num_f = [c for i, c in enumerate(cols_t)
             if i != len(cols_t)-1 and df_t[c].dtype in ['float64', 'int64']]
    cat_f = [c for i, c in enumerate(cols_t)
             if i != len(cols_t)-1 and df_t[c].dtype not in ['float64', 'int64']]
    turb_meta[ct] = {
        'numerical_features': num_f,
        'categorical_features': cat_f,
        'target_col': cols_t[-1],
        'threshold': 0.3,
        'all_columns': cols_t
    }
    print(f'  [{ct}] {len(num_f)} num + {len(cat_f)} cat features')

with open(os.path.join(SCALER_DIR, 'turbidity_meta.pkl'), 'wb') as f:
    pickle.dump(turb_meta, f)
print(f'  Saved turbidity_meta.pkl')

# ==========================================================
# 4. Copy model files
# ==========================================================
print('\n=== Copying Model Files ===')

# AL CatBoost dosage
src = os.path.join(ROOT, 'AL混凝剂/AL混凝剂-cat boost/训练结果/模型保存/best_model.cbm')
dst = os.path.join(MODEL_DIR, 'al_dosage_catboost.cbm')
shutil.copy2(src, dst)
print(f'  Copied: al_dosage_catboost.cbm')

# FE CatBoost dosage
src = os.path.join(ROOT, 'FE混凝剂/FE混凝剂-cat boost/单次训练测试/best_model.cbm')
dst = os.path.join(MODEL_DIR, 'fe_dosage_catboost.cbm')
shutil.copy2(src, dst)
print(f'  Copied: fe_dosage_catboost.cbm')

# AL NN models (3 seeds)
for seed in [42, 123, 1024]:
    src = os.path.join(ROOT, f'AL混凝剂/AL混凝剂-NN/第三次/模型保存/best_model_{seed}.pt')
    dst = os.path.join(MODEL_DIR, f'al_nn_seed{seed}.pt')
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f'  Copied: al_nn_seed{seed}.pt')
    else:
        print(f'  WARNING: not found: {src}')

# AL RF
src = os.path.join(ROOT, 'AL混凝剂/AL混凝剂-RF/第二次/模型保存/best_model.pkl')
dst = os.path.join(MODEL_DIR, 'al_dosage_rf.pkl')
shutil.copy2(src, dst)
print(f'  Copied: al_dosage_rf.pkl')

# AL SVR
src = os.path.join(ROOT, 'AL混凝剂/AL混凝剂-SVR/第一次/模型保存/best_model.pkl')
dst = os.path.join(MODEL_DIR, 'al_dosage_svr.pkl')
shutil.copy2(src, dst)
print(f'  Copied: al_dosage_svr.pkl')

# FE RF
src = os.path.join(ROOT, 'FE混凝剂/FE混凝剂-RF/第二次/模型保存/best_model.pkl')
dst = os.path.join(MODEL_DIR, 'fe_dosage_rf.pkl')
shutil.copy2(src, dst)
print(f'  Copied: fe_dosage_rf.pkl')

# FE SVR
src = os.path.join(ROOT, 'FE混凝剂/FE混凝剂-SVR/第一次/模型保存/best_model.pkl')
dst = os.path.join(MODEL_DIR, 'fe_dosage_svr.pkl')
shutil.copy2(src, dst)
print(f'  Copied: fe_dosage_svr.pkl')

# AL turbidity classification CatBoost
src = os.path.join(ROOT, '2-出水浊度/AL-二分类全训/模型保存/al_fulltrain_model.cbm')
dst = os.path.join(MODEL_DIR, 'al_turbidity_catboost.cbm')
shutil.copy2(src, dst)
print(f'  Copied: al_turbidity_catboost.cbm')

# FE turbidity classification CatBoost
src = os.path.join(ROOT, '2-出水浊度/FE-二分类/模型保存/fe_best_model.cbm')
dst = os.path.join(MODEL_DIR, 'fe_turbidity_catboost.cbm')
shutil.copy2(src, dst)
print(f'  Copied: fe_turbidity_catboost.cbm')

print('\n=== Done! All scalers and models prepared ===')
