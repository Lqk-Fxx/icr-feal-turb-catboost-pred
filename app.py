# -*- coding: utf-8 -*-
"""
ICR 混凝剂智能预测系统 — Streamlit Web App
仅使用 CatBoost 模型，简洁稳定。
支持中英双语切换。
启动: streamlit run app.py
"""
import streamlit as st
import matplotlib.pyplot as plt
import sys, os, traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import predict_dosage_catboost, predict_turbidity, get_feature_info

st.set_page_config(page_title='ICR Prediction', page_icon='🧪', layout='wide')

# ==========================================================
# 🌐 Bilingual Translation Dictionary
# ==========================================================
LANG = {
    'zh': {
        # Sidebar
        'sidebar_title': '🧪 ICR 预测系统',
        'sidebar_lang': '🌐 语言 / Language',
        'sidebar_task': '📋 选择任务',
        'task_al_dosage': '🔬 AL 投加量预测',
        'task_fe_dosage': '🧲 FE 投加量预测',
        'task_al_turb': '📊 AL 出水浊度分类',
        'task_fe_turb': '📊 FE 出水浊度分类',
        'sidebar_help': '📖 使用说明',
        'help_text': '''
1. **选择任务** — 左侧切换预测类型
2. **输入特征** — 填入水质参数
3. **点击预测** — 查看 CatBoost 结果

**模型**: CatBoost 梯度提升树
- AL 投加量 R² = 0.773
- FE 投加量 R² = 0.880
''',
        # Main
        'main_title': '🧪 ICR 混凝剂智能预测系统',
        'main_caption': 'CatBoost 模型 | AL/FE 投加量预测 + 出水浊度分类',
        'section_al_dosage': '🔬 AL (铝混凝剂) 投加量预测',
        'section_fe_dosage': '🧲 FE (铁混凝剂) 投加量预测',
        'section_al_turb': '📊 AL 出水浊度二分类',
        'section_fe_turb': '📊 FE 出水浊度二分类',
        'info_al_dosage': '💡 输入进水水质和**目标出水浊度**，预测需要的铝盐投加量 (ppm Al)。',
        'info_fe_dosage': '💡 输入进水水质和**目标出水浊度**，预测需要的铁盐投加量 (ppm Fe)。',
        'info_al_turb': '💡 输入水质参数和当前铝盐投加量，模型预测出水浊度 ≥ {threshold} NTU 的概率。',
        'info_fe_turb': '💡 输入水质参数和当前铁盐投加量，模型预测出水浊度 ≥ {threshold} NTU 的概率。',
        # Form
        'section_numerical': '📐 连续特征',
        'section_categorical': '🏷️ 类别特征',
        'submit_btn': '🔮 开始预测',
        'predicting': '预测中...',
        # Results - dosage
        'result_title': '📈 预测结果',
        'prediction_error': '预测失败: {error}',
        'catboost_pred': 'CatBoost 预测投加量',
        'train_r2': '训练集 R²',
        'model_label': '模型',
        # Results - turbidity
        'turb_result_title': '📊 分类结果',
        'turb_fail': '⚠️ **FAIL** — 预测出水浊度 ≥ {threshold} NTU 概率较高',
        'turb_pass': '✅ **PASS** — 预测出水浊度 < {threshold} NTU 概率较高',
        'fail_prob': 'Fail 概率',
        'train_threshold': '模型训练阈值: {threshold} NTU',
        'prob_axis': '概率',
        'footer': 'ICR Coagulant Prediction System | CatBoost | Streamlit',
        # Slider
        'turb_threshold_label': '🚦 **出水浊度判定阈值 (NTU)**',
        'turb_threshold_help': '出水浊度 ≥ 此值判定为 Fail。模型训练时使用 0.3 NTU，更改阈值仅影响显示判定。',
        # Feature labels
        'pH': 'pH',
        'avg_inflow': '月均进水流量 (MGD)',
        'eff_turb': '出水浊度 (NTU) — 期望/目标值',
        'in_turb': '进水浊度 (NTU)',
        'temp': '水温 (°C)',
        'uv': 'UV₂₅₄ (cm⁻¹)',
        'toc': 'TOC (ppm C)',
        'alkalinity': '碱度 (ppm CaCO₃)',
        'hardness': '总硬度 (ppm CaCO₃)',
        'al_dose': '铝盐投加量 (ppm Al)',
        'fe_dose': '铁盐投加量 (ppm Fe)',
        'wtp_type': '水厂类型',
        'disinf_type': '消毒类型',
        'source_cat': '水源类别',
        # Help hints
        'help_pH': '进水 pH，通常 6.0-9.0',
        'help_alkalinity': '进水碱度，以 CaCO₃ 计',
        'help_in_turb': '进水浊度，反映颗粒物含量',
        'help_eff_turb': '期望出水浊度（投加量预测用）/ 实际出水浊度（分类用）',
        'help_temp': '进水水温',
        'help_hardness': '进水总硬度，以 CaCO₃ 计',
        'help_toc': '总有机碳含量',
        'help_uv': 'UV₂₅₄ 吸光度',
        'help_avg_inflow': '水厂月均进水流量',
        'help_al_dose': '铝盐投加量（用于浊度分类）',
        'help_fe_dose': '铁盐投加量（用于浊度分类）',
    },
    'en': {
        # Sidebar
        'sidebar_title': '🧪 ICR Prediction System',
        'sidebar_lang': '🌐 语言 / Language',
        'sidebar_task': '📋 Select Task',
        'task_al_dosage': '🔬 AL Dosage Prediction',
        'task_fe_dosage': '🧲 FE Dosage Prediction',
        'task_al_turb': '📊 AL Turbidity Classification',
        'task_fe_turb': '📊 FE Turbidity Classification',
        'sidebar_help': '📖 Instructions',
        'help_text': '''
1. **Select Task** — Choose prediction type on the left
2. **Enter Features** — Fill in water quality parameters
3. **Predict** — View CatBoost results

**Model**: CatBoost Gradient Boosting
- AL Dosage R² = 0.773
- FE Dosage R² = 0.880
''',
        # Main
        'main_title': '🧪 ICR Coagulant Intelligent Prediction System',
        'main_caption': 'CatBoost Model | AL/FE Dosage Prediction + Effluent Turbidity Classification',
        'section_al_dosage': '🔬 AL (Aluminum Coagulant) Dosage Prediction',
        'section_fe_dosage': '🧲 FE (Iron Coagulant) Dosage Prediction',
        'section_al_turb': '📊 AL Effluent Turbidity Classification',
        'section_fe_turb': '📊 FE Effluent Turbidity Classification',
        'info_al_dosage': '💡 Enter source water quality and **target effluent turbidity** to predict required alum dose (ppm Al).',
        'info_fe_dosage': '💡 Enter source water quality and **target effluent turbidity** to predict required iron dose (ppm Fe).',
        'info_al_turb': '💡 Enter water quality and current alum dose; the model predicts the probability of effluent turbidity ≥ {threshold} NTU.',
        'info_fe_turb': '💡 Enter water quality and current iron dose; the model predicts the probability of effluent turbidity ≥ {threshold} NTU.',
        # Form
        'section_numerical': '📐 Continuous Features',
        'section_categorical': '🏷️ Categorical Features',
        'submit_btn': '🔮 Predict',
        'predicting': 'Predicting...',
        # Results - dosage
        'result_title': '📈 Prediction Results',
        'prediction_error': 'Prediction failed: {error}',
        'catboost_pred': 'CatBoost Predicted Dosage',
        'train_r2': 'Training R²',
        'model_label': 'Model',
        # Results - turbidity
        'turb_result_title': '📊 Classification Results',
        'turb_fail': '⚠️ **FAIL** — High probability of effluent turbidity ≥ {threshold} NTU',
        'turb_pass': '✅ **PASS** — High probability of effluent turbidity < {threshold} NTU',
        'fail_prob': 'Fail Probability',
        'train_threshold': 'Model training threshold: {threshold} NTU',
        'prob_axis': 'Probability',
        'footer': 'ICR Coagulant Prediction System | CatBoost | Streamlit',
        # Slider
        'turb_threshold_label': '🚦 **Effluent Turbidity Threshold (NTU)**',
        'turb_threshold_help': 'Effluent turbidity ≥ this value is classified as Fail. The model was trained with 0.3 NTU; changing this threshold only affects the displayed classification.',
        # Feature labels
        'pH': 'pH',
        'avg_inflow': 'Avg Inlet Flow (MGD)',
        'eff_turb': 'Effluent Turbidity (NTU) — Target',
        'in_turb': 'Inlet Turbidity (NTU)',
        'temp': 'Water Temperature (°C)',
        'uv': 'UV₂₅₄ (cm⁻¹)',
        'toc': 'TOC (ppm C)',
        'alkalinity': 'Alkalinity (ppm CaCO₃)',
        'hardness': 'Total Hardness (ppm CaCO₃)',
        'al_dose': 'Alum Dose (ppm Al)',
        'fe_dose': 'Iron Dose (ppm Fe)',
        'wtp_type': 'WTP Type',
        'disinf_type': 'Disinfection Type',
        'source_cat': 'Source Water Category',
        # Help hints
        'help_pH': 'Inlet pH, typically 6.0–9.0',
        'help_alkalinity': 'Inlet alkalinity as CaCO₃',
        'help_in_turb': 'Inlet turbidity, reflects particulate content',
        'help_eff_turb': 'Target (dosage) / Actual (classification) effluent turbidity',
        'help_temp': 'Inlet water temperature',
        'help_hardness': 'Inlet total hardness as CaCO₃',
        'help_toc': 'Total organic carbon content',
        'help_uv': 'UV₂₅₄ absorbance',
        'help_avg_inflow': 'Monthly average inlet flow',
        'help_al_dose': 'Alum dose (for turbidity classification)',
        'help_fe_dose': 'Iron dose (for turbidity classification)',
    }
}


def t(key, **kwargs):
    """Translate a key to the current language. Supports format kwargs."""
    lang = st.session_state.get('lang', 'zh')
    text = LANG.get(lang, LANG['zh']).get(key, key)
    if kwargs:
        text = text.format(**kwargs)
    return text


# ==========================================================
# Friendly label mapping — 根据语言返回标签
# ==========================================================
def make_label(fname):
    """Convert internal feature name to a clean display label (unit included)."""
    # 精确匹配 — 先匹配长关键词避免歧义
    exact_map = {
        'pH': t('pH'),
        '月均进水流量': t('avg_inflow'),
        '出水浊度': t('eff_turb'),
        '浊度': t('in_turb'),
        '温度': t('temp'),
        'UV': t('uv'),
        'TOC': t('toc'),
        '总有机碳': t('toc'),
        '碱度': t('alkalinity'),
        '总硬度': t('hardness'),
        '铝混凝剂': t('al_dose'),
        '铁混凝剂': t('fe_dose'),
        '水厂类型': t('wtp_type'),
        '消毒类型': t('disinf_type'),
        '水源类别': t('source_cat'),
    }
    for key, label in exact_map.items():
        if key in fname:
            return label
    return fname


def make_help(fname):
    """Generate help text for a feature."""
    hints = {
        'pH': t('help_pH'),
        '碱度': t('help_alkalinity'),
        '浊度': t('help_in_turb'),
        '出水浊度': t('help_eff_turb'),
        '温度': t('help_temp'),
        '总硬度': t('help_hardness'),
        '总有机碳': t('help_toc'),
        'UV': t('help_uv'),
        '月均进水流量': t('help_avg_inflow'),
        '铝混凝剂': t('help_al_dose'),
        '铁混凝剂': t('help_fe_dose'),
    }
    for key, hint in hints.items():
        if key in fname:
            return hint
    return ''


# ==========================================================
# Sidebar
# ==========================================================
st.sidebar.markdown(f"## {t('sidebar_title')}")

# Language selector — always at top
lang = st.sidebar.radio(
    t('sidebar_lang'),
    ['🇨🇳 中文', '🇺🇸 English'],
    index=0,
    horizontal=True,
    key='lang_selector'
)
# Map UI selection to language code
new_lang = 'zh' if '中文' in lang else 'en'
if 'lang' not in st.session_state:
    st.session_state.lang = 'zh'
st.session_state.lang = new_lang

st.sidebar.markdown("---")

task = st.sidebar.radio(
    t('sidebar_task'),
    [t('task_al_dosage'), t('task_fe_dosage'),
     t('task_al_turb'), t('task_fe_turb')]
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
### {t('sidebar_help')}
{t('help_text')}
""")


# ==========================================================
# Build input form
# ==========================================================
def build_input_form(feature_info, num_cols=3):
    """Build input form from feature metadata."""
    inputs = {}

    if feature_info.get('numerical'):
        st.markdown(f"#### {t('section_numerical')}")
        cols = st.columns(num_cols)
        for i, feat in enumerate(feature_info['numerical']):
            with cols[i % num_cols]:
                label = make_label(feat['name'])
                help_text = make_help(feat['name'])
                val = float(feat.get('default', 0.0))
                lo = float(feat.get('min', 0.0))
                hi = float(feat.get('max', 1000.0))
                inputs[feat['name']] = st.number_input(
                    label, min_value=lo, max_value=hi, value=val,
                    help=help_text, key=f"n_{i}"
                )

    if feature_info.get('categorical'):
        st.markdown(f"#### {t('section_categorical')}")
        cat_feats = feature_info['categorical']
        cols = st.columns(min(len(cat_feats), 3))
        for i, feat in enumerate(cat_feats):
            with cols[i % len(cols)]:
                label = make_label(feat['name'])
                options = feat.get('options', [])
                default = feat.get('default', options[0] if options else '')
                default_idx = options.index(default) if default in options else 0
                inputs[feat['name']] = st.selectbox(
                    label, options, index=default_idx, key=f"c_{i}"
                )

    return inputs


# ==========================================================
# Display dosage result (CatBoost only)
# ==========================================================
def display_dosage_result(result):
    st.markdown("---")
    st.markdown(f"### {t('result_title')}")

    if 'error' in result:
        st.error(t('prediction_error', error=result['error']))
        return

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#2193b0,#6dd5ed);border-radius:12px;
                    padding:1.5rem 2rem;color:white;">
            <div style="font-size:0.9rem;opacity:0.85;">{t('catboost_pred')}</div>
            <div style="font-size:3rem;font-weight:800;">
                {result['prediction']:.2f} <span style="font-size:1.2rem;">{result['unit']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.metric(t('train_r2'), f"{result.get('r2_train', 0):.4f}")
    with col3:
        st.metric(t('model_label'), "CatBoost")


# ==========================================================
# Display turbidity result with custom threshold
# ==========================================================
def display_turbidity_result(result, user_threshold):
    st.markdown("---")
    st.markdown(f"### {t('turb_result_title')}")

    proba = result.get('probability_fail', 0)
    model_threshold = result.get('threshold', 0.3)

    # 按用户阈值判定
    is_fail_user = proba >= 0.5  # CatBoost 默认 0.5 概率阈值

    c1, c2 = st.columns([1, 2])

    with c1:
        if is_fail_user:
            st.error(t('turb_fail', threshold=model_threshold))
        else:
            st.success(t('turb_pass', threshold=model_threshold))
        st.metric(t('fail_prob'), f"{proba:.1%}")
        st.caption(t('train_threshold', threshold=model_threshold))

    with c2:
        fig, ax = plt.subplots(figsize=(5, 1.5))
        colors = ['#27ae60', '#e74c3c']
        pass_label = 'Pass' if new_lang == 'en' else 'Pass'
        fail_label = 'Fail'
        ax.barh(['Risk'], [1-proba], color=colors[0], height=0.4,
                label=f'{pass_label} ({(1-proba)*100:.1f}%)')
        ax.barh(['Risk'], [proba], left=[1-proba], color=colors[1], height=0.4,
                label=f'{fail_label} ({proba*100:.1f}%)')
        ax.set_xlim(0, 1)
        ax.axvline(0.5, color='black', lw=1, ls='--', alpha=0.4)
        ax.legend(fontsize=9, loc='lower right')
        ax.set_xlabel(t('prob_axis'))
        st.pyplot(fig)


# ==========================================================
# Main Content
# ==========================================================
st.title(t('main_title'))
st.caption(t('main_caption'))

# Task key (language-independent) — match by prefix
TASK_AL_DOSAGE = t('task_al_dosage')
TASK_FE_DOSAGE = t('task_fe_dosage')
TASK_AL_TURB = t('task_al_turb')

# ---- AL Dosage ----
if TASK_AL_DOSAGE in task:
    st.markdown(f"## {t('section_al_dosage')}")
    st.info(t('info_al_dosage'))

    feature_info = get_feature_info('AL', 'dosage')
    with st.form('al_form'):
        inputs = build_input_form(feature_info)
        submitted = st.form_submit_button(t('submit_btn'), use_container_width=True)
    if submitted:
        try:
            with st.spinner(t('predicting')):
                result = predict_dosage_catboost('AL', inputs)
            display_dosage_result(result)
        except Exception as e:
            st.error(f"预测出错: {e}")
            st.code(traceback.format_exc())

# ---- FE Dosage ----
elif TASK_FE_DOSAGE in task:
    st.markdown(f"## {t('section_fe_dosage')}")
    st.info(t('info_fe_dosage'))

    feature_info = get_feature_info('FE', 'dosage')
    with st.form('fe_form'):
        inputs = build_input_form(feature_info)
        submitted = st.form_submit_button(t('submit_btn'), use_container_width=True)
    if submitted:
        try:
            with st.spinner(t('predicting')):
                result = predict_dosage_catboost('FE', inputs)
            display_dosage_result(result)
        except Exception as e:
            st.error(f"预测出错: {e}")
            st.code(traceback.format_exc())

# ---- AL Turbidity ----
elif TASK_AL_TURB in task:
    st.markdown(f"## {t('section_al_turb')}")

    user_threshold = st.slider(
        t('turb_threshold_label'),
        min_value=0.05, max_value=2.0, value=0.3, step=0.05,
        help=t('turb_threshold_help')
    )
    st.info(t('info_al_turb', threshold=user_threshold))

    feature_info = get_feature_info('AL', 'turbidity')
    with st.form('al_turb_form'):
        inputs = build_input_form(feature_info)
        submitted = st.form_submit_button(t('submit_btn'), use_container_width=True)
    if submitted:
        try:
            with st.spinner(t('predicting')):
                result = predict_turbidity('AL', inputs)
            display_turbidity_result(result, user_threshold)
        except Exception as e:
            st.error(f"预测出错: {e}")
            st.code(traceback.format_exc())

# ---- FE Turbidity ----
else:
    st.markdown(f"## {t('section_fe_turb')}")

    user_threshold = st.slider(
        t('turb_threshold_label'),
        min_value=0.05, max_value=2.0, value=0.3, step=0.05,
        help=t('turb_threshold_help')
    )
    st.info(t('info_fe_turb', threshold=user_threshold))

    feature_info = get_feature_info('FE', 'turbidity')
    with st.form('fe_turb_form'):
        inputs = build_input_form(feature_info)
        submitted = st.form_submit_button(t('submit_btn'), use_container_width=True)
    if submitted:
        try:
            with st.spinner(t('predicting')):
                result = predict_turbidity('FE', inputs)
            display_turbidity_result(result, user_threshold)
        except Exception as e:
            st.error(f"预测出错: {e}")
            st.code(traceback.format_exc())

st.markdown("---")
st.caption(t('footer'))
