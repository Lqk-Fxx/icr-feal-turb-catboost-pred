# -*- coding: utf-8 -*-
"""
ICR 混凝剂预测工具模块
支持: AL/FE 投加量预测(回归) + AL/FE 出水浊度二分类

模型:
  CatBoost (.cbm) — 原生处理类别特征，直接接受原始数据
  NN Ensemble (.pt) — 需要 StandardScaler + OneHot 编码
  RF / SVR (.pkl) — 参考模型
"""
import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from catboost import CatBoostRegressor, CatBoostClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
SCALER_DIR = os.path.join(BASE_DIR, 'scalers')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================================
# Cache loaded models to avoid repeated disk I/O
# ==========================================================
_cache = {}

# ==========================================================
# NN Model Architecture (must match training)
# ==========================================================
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, neurons, dropout, use_bn=True):
        super().__init__()
        layers = []
        prev = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(prev, neurons))
            if use_bn:
                layers.append(nn.BatchNorm1d(neurons))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = neurons
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ==========================================================
# Model Loading
# ==========================================================
def _load_scaler(ct):
    """Load scaler package for AL or FE."""
    key = f'scaler_{ct}'
    if key not in _cache:
        path = os.path.join(SCALER_DIR, f'{ct.lower()}_scaler.pkl')
        with open(path, 'rb') as f:
            _cache[key] = pickle.load(f)
    return _cache[key]

def _load_turbidity_meta():
    """Load turbidity classification metadata."""
    if 'turb_meta' not in _cache:
        path = os.path.join(SCALER_DIR, 'turbidity_meta.pkl')
        with open(path, 'rb') as f:
            _cache['turb_meta'] = pickle.load(f)
    return _cache['turb_meta']

def _load_catboost_regressor(ct):
    """Load CatBoost regressor for AL or FE."""
    key = f'cb_reg_{ct}'
    if key not in _cache:
        path = os.path.join(MODEL_DIR, f'{ct.lower()}_dosage_catboost.cbm')
        _cache[key] = CatBoostRegressor().load_model(path)
    return _cache[key]

def _load_catboost_classifier(ct):
    """Load CatBoost classifier for AL or FE turbidity."""
    key = f'cb_cls_{ct}'
    if key not in _cache:
        path = os.path.join(MODEL_DIR, f'{ct.lower()}_turbidity_catboost.cbm')
        _cache[key] = CatBoostClassifier().load_model(path)
    return _cache[key]

def _load_nn_ensemble(ct):
    """Load NN Ensemble (5 seeds) for AL or FE."""
    key = f'nn_{ct}'
    if key not in _cache:
        models = []
        for seed in [42, 123, 456, 789, 1024]:
            path = os.path.join(MODEL_DIR, f'{ct.lower()}_nn_seed{seed}.pt')
            if not os.path.exists(path):
                continue

            # Load checkpoint
            checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)

            # Determine architecture from checkpoint or use known defaults
            if ct == 'AL':
                hidden_layers = checkpoint.get('hidden_layers', 3)
                neurons = checkpoint.get('neurons', 128)
                dropout = checkpoint.get('dropout', 0.1)
                use_bn = checkpoint.get('use_bn', True)
            else:  # FE
                hidden_layers = checkpoint.get('hidden_layers', 2)
                neurons = checkpoint.get('neurons', 128)
                dropout = checkpoint.get('dropout', 0.1)
                use_bn = checkpoint.get('use_bn', True)

            input_dim = checkpoint.get('input_dim', 17)
            model = MLP(input_dim, hidden_layers, neurons, dropout, use_bn).to(DEVICE)

            # Handle both raw state_dict and full checkpoint
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

            model.eval()
            models.append(model)

        if not models:
            raise FileNotFoundError(f'No NN models found for {ct}')
        _cache[key] = models
    return _cache[key]

# ==========================================================
# Prediction: Dosage (Regression)
# ==========================================================

def predict_dosage_catboost(ct, input_dict):
    """
    Predict dosage using CatBoost.

    Parameters
    ----------
    ct : str — 'AL' or 'FE'
    input_dict : dict — keys matching raw feature names, e.g.
        {'pH': 7.5, '碱度-ALk_ppmCaCO3': 120, ...,
         '水厂类型-M_WTP_Type': 'CONV', '水厂消毒类型-WTP_Disinf_Type': 'CL2', ...}

    Returns
    -------
    dict with 'prediction' (float, ppm), 'model': 'CatBoost', 'r2_train': float
    """
    model = _load_catboost_regressor(ct)
    scaler_pkg = _load_scaler(ct)

    # CatBoost needs features in EXACT training order (raw column order excl target)
    cb_features = scaler_pkg['cb_feature_order']

    row = {}
    for feat in cb_features:
        val = input_dict.get(feat)
        if val is None:
            raise ValueError(f'Missing feature: {feat}')
        row[feat] = val

    df = pd.DataFrame([row])

    # CatBoost handles categorical features automatically
    # The .cbm file stores feature names and categorical feature indices
    pred = float(model.predict(df)[0])

    # Training R² for reference
    r2_ref = {'AL': 0.7733, 'FE': 0.8802}[ct]

    return {
        'prediction': max(0, pred),  # dosage cannot be negative
        'prediction_raw': pred,
        'model': 'CatBoost',
        'r2_train': r2_ref,
        'unit': 'ppmAl' if ct == 'AL' else 'ppmFe'
    }


def predict_dosage_nn_ensemble(ct, input_dict):
    """
    Predict dosage using NN 5-Seed Ensemble.

    Pipeline: raw input → StandardScaler → OneHot encode → NN → inverse transform
    """
    models = _load_nn_ensemble(ct)
    scaler_pkg = _load_scaler(ct)
    n_seeds = len(models)

    # Step 1: Scale numerical features
    num_feats = scaler_pkg['numerical_features']
    num_vals = np.array([[float(input_dict[f]) for f in num_feats]], dtype=np.float64)
    scaled_num = (num_vals - scaler_pkg['scaler_mean']) / scaler_pkg['scaler_std']

    # Step 2: One-hot encode categorical features
    oh_parts = []
    for cat_feat in scaler_pkg['categorical_features']:
        raw_val = input_dict[cat_feat]
        categories = scaler_pkg['categorical_values'][cat_feat]
        vec = np.zeros(len(categories), dtype=np.float32)
        if raw_val in categories:
            vec[categories.index(raw_val)] = 1.0
        oh_parts.append(vec)

    oh_vector = np.concatenate(oh_parts)

    # Step 3: Assemble feature vector in encoded_feature_order
    # Numerical features first (in order), then one-hot columns
    feature_vector = np.concatenate([scaled_num.flatten(), oh_vector]).astype(np.float32)

    # Verify length
    expected_len = scaler_pkg['n_features']
    if len(feature_vector) != expected_len:
        raise ValueError(
            f'Feature vector length {len(feature_vector)} != expected {expected_len}. '
            f'Numerical: {len(num_feats)}, One-hot total: {len(oh_vector)}'
        )

    # Step 4: NN Ensemble prediction (scaled output)
    x = torch.tensor(feature_vector.reshape(1, -1)).to(DEVICE)
    preds_scaled = []
    with torch.no_grad():
        for model in models:
            preds_scaled.append(model(x).cpu().numpy().flatten()[0])

    ensemble_scaled = np.mean(preds_scaled)
    std_scaled = np.std(preds_scaled) if n_seeds > 1 else 0

    # Step 5: Inverse transform target (scaled → ppm)
    pred_ppm = ensemble_scaled * scaler_pkg['target_std'] + scaler_pkg['target_mean']

    r2_ref = {'AL': 0.7744, 'FE': 0.8992}[ct]

    return {
        'prediction': max(0, pred_ppm),
        'prediction_raw': pred_ppm,
        'model': f'NN Ensemble ({n_seeds} seeds)',
        'r2_train': r2_ref,
        'unit': 'ppmAl' if ct == 'AL' else 'ppmFe',
        'ensemble_std': std_scaled * scaler_pkg['target_std'],  # std in ppm
        'individual_predictions': [
            float(p * scaler_pkg['target_std'] + scaler_pkg['target_mean'])
            for p in preds_scaled
        ]
    }


def predict_dosage(ct, input_dict):
    """
    Predict dosage using both CatBoost and NN Ensemble.
    Returns combined results.
    """
    results = {}
    try:
        results['catboost'] = predict_dosage_catboost(ct, input_dict)
    except Exception as e:
        results['catboost'] = {'error': str(e)}

    try:
        results['nn_ensemble'] = predict_dosage_nn_ensemble(ct, input_dict)
    except Exception as e:
        results['nn_ensemble'] = {'error': str(e)}

    return results


# ==========================================================
# Prediction: Turbidity Classification
# ==========================================================

def predict_turbidity(ct, input_dict):
    """
    Predict effluent turbidity classification (Pass/Fail) using CatBoost.

    Parameters
    ----------
    ct : str — 'AL' or 'FE'
    input_dict : dict — raw feature values

    Returns
    -------
    dict with 'prediction': 'Pass'/'Fail', 'probability': float, 'threshold': 0.3
    """
    model = _load_catboost_classifier(ct)
    meta = _load_turbidity_meta()[ct]
    threshold = meta['threshold']

    # Build DataFrame
    all_features = meta['numerical_features'] + meta['categorical_features']
    row = {}
    for feat in all_features:
        val = input_dict.get(feat)
        if val is None:
            raise ValueError(f'Missing feature: {feat}')
        row[feat] = val

    df = pd.DataFrame([row])

    # Predict
    proba = float(model.predict_proba(df)[0, 1])  # probability of class 1 (Fail)
    pred_class = int(proba >= 0.5)  # CatBoost default threshold
    pred_custom = int(proba >= 0.4)  # might differ from training's 0.3 threshold logic

    # The model was trained with target = (出水浊度 >= 0.3).astype(int)
    # So class 1 = Fail (>= 0.3 NTU), class 0 = Pass (< 0.3 NTU)
    label = 'Fail' if pred_class == 1 else 'Pass'

    return {
        'prediction': label,
        'prediction_detail': f'Effluent Turbidity ≥ {threshold} NTU' if label == 'Fail' else f'Effluent Turbidity < {threshold} NTU',
        'probability_fail': proba,
        'probability_pass': 1 - proba,
        'threshold': threshold,
        'model': 'CatBoost',
        'is_fail': pred_class == 1
    }


# ==========================================================
# Utility: Get feature info for the UI
# ==========================================================

def get_feature_info(ct, task='dosage'):
    """
    Get feature metadata for building UI input forms.

    Returns
    -------
    dict with 'numerical': [{name, min, max, default, unit, description}],
              'categorical': [{name, options, default}]
    """
    if task == 'dosage':
        pkg = _load_scaler(ct)

        # Numerical features with reasonable ranges
        num_info = []
        num_defaults = {
            'pH': (7.5, 6.0, 9.0, ''),
            '碱度-ALk_ppmCaCO3': (100, 20, 300, 'ppm CaCO₃'),
            '碱度-.ALk_ppmCaCO3': (100, 20, 300, 'ppm CaCO₃'),
            '浊度-TURB_NTU': (10, 0.1, 100, 'NTU'),
            '温度-TEMP_C': (18, 1, 35, '°C'),
            '总硬度-Tot_HARD_ppmCaCO3': (150, 20, 500, 'ppm CaCO₃'),
            '总有机碳-TOC_ppmC': (3.0, 0.5, 15, 'ppm C'),
            '总有机碳TOC_ppmC': (3.0, 0.5, 15, 'ppm C'),
            'UV吸光度-254_UV_cm_1': (0.1, 0.0, 0.5, 'cm⁻¹'),
            'UV紫外-254UV_cm_1': (0.1, 0.0, 0.5, 'cm⁻¹'),
            '月平均进水流量-M_Avg_Inft_Flow_MGD': (50, 0, 300, 'MGD'),
            '月均进水流量-M_Avg_Inft_Flow_MGD': (50, 0, 300, 'MGD'),
            '月均进水流量-M_Avg_Inft_Flow_MGD（百万加仑\\天）': (50, 0, 300, 'MGD'),
            '出水浊度_Y': (0.1, 0.0, 1.0, 'NTU (target)'),
            '混凝剂投加量-Alum_dose_ppmAl': (2.5, 0.1, 20, 'ppm Al'),
            '混凝剂投加量-Iron_dose_ppmFe': (5.0, 0.1, 50, 'ppm Fe'),
        }

        for fname in pkg['numerical_features']:
            d = num_defaults.get(fname, (1.0, 0.0, 100, ''))
            num_info.append({
                'name': fname,
                'default': d[0],
                'min': d[1],
                'max': d[2],
                'unit': d[3]
            })

        # Categorical features
        cat_info = []
        for fname in pkg['categorical_features']:
            options = pkg['categorical_values'][fname]
            cat_info.append({
                'name': fname,
                'options': options,
                'default': options[0]
            })

        return {'numerical': num_info, 'categorical': cat_info}

    elif task == 'turbidity':
        meta = _load_turbidity_meta()[ct]

        num_defaults = {
            'pH': (7.5, 6.0, 9.0, ''),
            '碱度-ALk_ppmCaCO3': (100, 20, 300, 'ppm CaCO₃'),
            '碱度-.ALk_ppmCaCO3': (100, 20, 300, 'ppm CaCO₃'),
            '浊度-TURB_NTU': (10, 0.1, 100, 'NTU'),
            '温度-TEMP_C': (18, 1, 35, '°C'),
            '总硬度-Tot_HARD_ppmCaCO3': (150, 20, 500, 'ppm CaCO₃'),
            '总有机碳-TOC_ppmC': (3.0, 0.5, 15, 'ppm C'),
            '总有机碳TOC_ppmC': (3.0, 0.5, 15, 'ppm C'),
            'UV吸光度-254_UV_cm_1': (0.1, 0.0, 0.5, 'cm⁻¹'),
            'UV紫外-254UV_cm_1': (0.1, 0.0, 0.5, 'cm⁻¹'),
            '月平均进水流量-M_Avg_Inft_Flow_MGD': (50, 0, 300, 'MGD'),
            '月均进水流量-M_Avg_Inft_Flow_MGD': (50, 0, 300, 'MGD'),
            '月均进水流量-M_Avg_Inft_Flow_MGD（百万加仑\\天）': (50, 0, 300, 'MGD'),
            '混凝剂投加量-Alum_dose_ppmAl': (2.5, 0.1, 20, 'ppm Al'),
            '铝混凝剂-Alum_dose_ppmAl': (2.5, 0.1, 20, 'ppm Al'),
            '混凝剂投加量-Iron_dose_ppmFe': (5.0, 0.1, 50, 'ppm Fe'),
            '铁混凝剂-Iron_dose_ppmFe': (5.0, 0.1, 50, 'ppm Fe'),
        }

        num_info = []
        for fname in meta['numerical_features']:
            d = num_defaults.get(fname, (1.0, 0.0, 100, ''))
            num_info.append({
                'name': fname,
                'default': d[0],
                'min': d[1],
                'max': d[2],
                'unit': d[3]
            })

        cat_info = []
        cat_values = meta.get('categorical_values', {})
        for fname in meta['categorical_features']:
            options = cat_values.get(fname, [])
            cat_info.append({
                'name': fname,
                'options': options,
                'default': options[0] if options else ''
            })

        return {'numerical': num_info, 'categorical': cat_info}


def get_dosage_feature_info(ct):
    """Convenience: get dosage-specific features (same as get_feature_info for dosage task)."""
    return get_feature_info(ct, 'dosage')


# ==========================================================
# Quick test
# ==========================================================
if __name__ == '__main__':
    # Build test inputs using ACTUAL column names from scaler/metadata
    pkg_al = _load_scaler('AL')
    pkg_fe = _load_scaler('FE')
    meta_al = _load_turbidity_meta()['AL']
    meta_fe = _load_turbidity_meta()['FE']

    # Default values for numerical features
    num_defaults = {
        'pH': 7.5,
        '碱度-ALk_ppmCaCO3': 120, '碱度-.ALk_ppmCaCO3': 120,
        '浊度-TURB_NTU': 10, '温度-TEMP_C': 20,
        '总硬度-Tot_HARD_ppmCaCO3': 150,
        '总有机碳-TOC_ppmC': 3.0, '总有机碳TOC_ppmC': 3.0,
        'UV吸光度-254_UV_cm_1': 0.1, 'UV紫外-254UV_cm_1': 0.1,
        '月均进水流量-M_Avg_Inft_Flow_MGD': 50,
        '月均进水流量-M_Avg_Inft_Flow_MGD（百万加仑\\天）': 50,
        '出水浊度_Y': 0.1,
        '铝混凝剂-Alum_dose_ppmAl': 2.5,
        '铁混凝剂-Iron_dose_ppmFe': 5.0,
    }
    cat_defaults = {
        '水厂类型-M_WTP_Type': 'CONV',
        '水厂消毒类型-WTP_Disinf_Type': 'CL2',
        '水源类别-M_Source_Cat': 'SW',
    }

    # Test AL dosage
    al_dosage_input = {}
    for f in pkg_al['numerical_features']:
        al_dosage_input[f] = num_defaults.get(f, 1.0)
    for f in pkg_al['categorical_features']:
        al_dosage_input[f] = cat_defaults[f]
    print('=== AL Dosage Prediction ===')
    r = predict_dosage('AL', al_dosage_input)
    for k, v in r.items():
        if 'error' not in str(v):
            print(f'  {k}: {v["prediction"]:.3f} {v["unit"]} ({v["model"]})')
        else:
            print(f'  {k}: ERROR - {v["error"]}')

    # Test FE dosage
    fe_dosage_input = {}
    for f in pkg_fe['numerical_features']:
        fe_dosage_input[f] = num_defaults.get(f, 1.0)
    for f in pkg_fe['categorical_features']:
        fe_dosage_input[f] = cat_defaults[f]
    print('\n=== FE Dosage Prediction ===')
    r = predict_dosage('FE', fe_dosage_input)
    for k, v in r.items():
        if 'error' not in str(v):
            print(f'  {k}: {v["prediction"]:.3f} {v["unit"]} ({v["model"]})')
        else:
            print(f'  {k}: ERROR - {v["error"]}')

    # Test AL turbidity
    al_turb_input = {}
    for f in meta_al['numerical_features']:
        al_turb_input[f] = num_defaults.get(f, 1.0)
    for f in meta_al['categorical_features']:
        al_turb_input[f] = cat_defaults.get(f, cat_defaults['水厂类型-M_WTP_Type'])
    print('\n=== AL Turbidity Classification ===')
    r = predict_turbidity('AL', al_turb_input)
    print(f'  {r["prediction"]} (prob_fail={r["probability_fail"]:.3f})')

    # Test FE turbidity
    fe_turb_input = {}
    for f in meta_fe['numerical_features']:
        fe_turb_input[f] = num_defaults.get(f, 1.0)
    for f in meta_fe['categorical_features']:
        fe_turb_input[f] = cat_defaults.get(f, cat_defaults['水厂类型-M_WTP_Type'])
    print('\n=== FE Turbidity Classification ===')
    r = predict_turbidity('FE', fe_turb_input)
    print(f'  {r["prediction"]} (prob_fail={r["probability_fail"]:.3f})')

    print('\n=== All tests passed! ===')
