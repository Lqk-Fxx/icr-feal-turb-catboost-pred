# -*- coding: utf-8 -*-
"""
ICR 混凝剂智能预测系统 — Streamlit Web App
仅使用 CatBoost 模型，简洁稳定。
启动: streamlit run app.py
"""
import streamlit as st
import matplotlib.pyplot as plt
import sys, os, traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import predict_dosage_catboost, predict_turbidity, get_feature_info

st.set_page_config(page_title='ICR 预测系统', page_icon='🧪', layout='wide')

# ==========================================================
# Friendly label mapping — 精确匹配，标签自带单位
# ==========================================================
def make_label(fname):
    """Convert internal feature name to a clean display label (unit included)."""
    # 精确匹配 — 先匹配长关键词避免歧义
    exact_map = {
        'pH': 'pH',
        '月均进水流量': '月均进水流量 (MGD)',
        '出水浊度': '出水浊度 (NTU) — 期望/目标值',
        '浊度': '进水浊度 (NTU)',
        '温度': '水温 (°C)',
        'UV': 'UV₂₅₄ (cm⁻¹)',
        'TOC': 'TOC (ppm C)',
        '总有机碳': 'TOC (ppm C)',
        '碱度': '碱度 (ppm CaCO₃)',
        '总硬度': '总硬度 (ppm CaCO₃)',
        '铝混凝剂': '铝盐投加量 (ppm Al)',
        '铁混凝剂': '铁盐投加量 (ppm Fe)',
        '水厂类型': '水厂类型',
        '消毒类型': '消毒类型',
        '水源类别': '水源类别',
    }
    for key, label in exact_map.items():
        if key in fname:
            return label
    return fname

def make_help(fname):
    """Generate help text for a feature."""
    hints = {
        'pH': '进水 pH，通常 6.0-9.0',
        '碱度': '进水碱度，以 CaCO₃ 计',
        '浊度': '进水浊度，反映颗粒物含量',
        '出水浊度': '期望出水浊度（投加量预测用）/ 实际出水浊度（分类用）',
        '温度': '进水水温',
        '总硬度': '进水总硬度，以 CaCO₃ 计',
        '总有机碳': '总有机碳含量',
        'UV': 'UV₂₅₄ 吸光度',
        '月均进水流量': '水厂月均进水流量',
        '铝混凝剂': '铝盐投加量（用于浊度分类）',
        '铁混凝剂': '铁盐投加量（用于浊度分类）',
    }
    for key, hint in hints.items():
        if key in fname:
            return hint
    return ''

# ==========================================================
# Sidebar
# ==========================================================
st.sidebar.markdown("## 🧪 ICR 预测系统")
st.sidebar.markdown("---")

task = st.sidebar.radio(
    '### 📋 选择任务',
    ['🔬 AL 投加量预测', '🧲 FE 投加量预测',
     '📊 AL 出水浊度分类', '📊 FE 出水浊度分类']
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📖 使用说明

1. **选择任务** — 左侧切换预测类型
2. **输入特征** — 填入水质参数
3. **点击预测** — 查看 CatBoost 结果

**模型**: CatBoost 梯度提升树
- AL 投加量 R² = 0.773
- FE 投加量 R² = 0.880
""")

# ==========================================================
# Build input form — 不再追加额外单位
# ==========================================================
def build_input_form(feature_info, num_cols=3):
    """Build input form from feature metadata."""
    inputs = {}

    if feature_info.get('numerical'):
        st.markdown("#### 📐 连续特征")
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
        st.markdown("#### 🏷️ 类别特征")
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
    st.markdown("### 📈 预测结果")

    if 'error' in result:
        st.error(f"预测失败: {result['error']}")
        return

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#2193b0,#6dd5ed);border-radius:12px;
                    padding:1.5rem 2rem;color:white;">
            <div style="font-size:0.9rem;opacity:0.85;">CatBoost 预测投加量</div>
            <div style="font-size:3rem;font-weight:800;">
                {result['prediction']:.2f} <span style="font-size:1.2rem;">{result['unit']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.metric("训练集 R²", f"{result.get('r2_train', 0):.4f}")
    with col3:
        st.metric("模型", "CatBoost")


# ==========================================================
# Display turbidity result with custom threshold
# ==========================================================
def display_turbidity_result(result, user_threshold):
    st.markdown("---")
    st.markdown("### 📊 分类结果")

    proba = result.get('probability_fail', 0)
    model_threshold = result.get('threshold', 0.3)

    # 按用户阈值判定
    is_fail_user = proba >= 0.5  # CatBoost 默认 0.5 概率阈值
    label_user = 'Fail' if is_fail_user else 'Pass'

    c1, c2 = st.columns([1, 2])

    with c1:
        if is_fail_user:
            st.error(f"⚠️ **FAIL** — 预测出水浊度 ≥ {model_threshold} NTU 概率较高")
        else:
            st.success(f"✅ **PASS** — 预测出水浊度 < {model_threshold} NTU 概率较高")
        st.metric("Fail 概率", f"{proba:.1%}")
        st.caption(f"模型训练阈值: {model_threshold} NTU")

    with c2:
        fig, ax = plt.subplots(figsize=(5, 1.5))
        colors = ['#27ae60', '#e74c3c']
        ax.barh(['风险'], [1-proba], color=colors[0], height=0.4, label=f'Pass ({(1-proba)*100:.1f}%)')
        ax.barh(['风险'], [proba], left=[1-proba], color=colors[1], height=0.4, label=f'Fail ({proba*100:.1f}%)')
        ax.set_xlim(0, 1)
        ax.axvline(0.5, color='black', lw=1, ls='--', alpha=0.4)
        ax.legend(fontsize=9, loc='lower right')
        ax.set_xlabel('概率')
        st.pyplot(fig)


# ==========================================================
# Main Content
# ==========================================================
st.title('🧪 ICR 混凝剂智能预测系统')
st.caption('CatBoost 模型 | AL/FE 投加量预测 + 出水浊度分类')

# ---- AL Dosage ----
if 'AL 投加量' in task:
    st.markdown("## 🔬 AL (铝混凝剂) 投加量预测")
    st.info("💡 输入进水水质和**目标出水浊度**，预测需要的铝盐投加量 (ppm Al)。")

    feature_info = get_feature_info('AL', 'dosage')
    with st.form('al_form'):
        inputs = build_input_form(feature_info)
        submitted = st.form_submit_button('🔮 开始预测', use_container_width=True)
    if submitted:
        try:
            with st.spinner('预测中...'):
                result = predict_dosage_catboost('AL', inputs)
            display_dosage_result(result)
        except Exception as e:
            st.error(f"预测出错: {e}")
            st.code(traceback.format_exc())

# ---- FE Dosage ----
elif 'FE 投加量' in task:
    st.markdown("## 🧲 FE (铁混凝剂) 投加量预测")
    st.info("💡 输入进水水质和**目标出水浊度**，预测需要的铁盐投加量 (ppm Fe)。")

    feature_info = get_feature_info('FE', 'dosage')
    with st.form('fe_form'):
        inputs = build_input_form(feature_info)
        submitted = st.form_submit_button('🔮 开始预测', use_container_width=True)
    if submitted:
        try:
            with st.spinner('预测中...'):
                result = predict_dosage_catboost('FE', inputs)
            display_dosage_result(result)
        except Exception as e:
            st.error(f"预测出错: {e}")
            st.code(traceback.format_exc())

# ---- AL Turbidity ----
elif 'AL 出水浊度' in task:
    st.markdown("## 📊 AL 出水浊度二分类")

    # 阈值滑块 — 放在 form 外面
    user_threshold = st.slider(
        '🚦 **出水浊度判定阈值 (NTU)**',
        min_value=0.05, max_value=2.0, value=0.3, step=0.05,
        help='出水浊度 ≥ 此值判定为 Fail。模型训练时使用 0.3 NTU，更改阈值仅影响显示判定。'
    )
    st.info(f"💡 输入水质参数和当前铝盐投加量，模型预测出水浊度 ≥ {user_threshold} NTU 的概率。")

    feature_info = get_feature_info('AL', 'turbidity')
    with st.form('al_turb_form'):
        inputs = build_input_form(feature_info)
        submitted = st.form_submit_button('🔮 开始预测', use_container_width=True)
    if submitted:
        try:
            with st.spinner('预测中...'):
                result = predict_turbidity('AL', inputs)
            display_turbidity_result(result, user_threshold)
        except Exception as e:
            st.error(f"预测出错: {e}")
            st.code(traceback.format_exc())

# ---- FE Turbidity ----
else:
    st.markdown("## 📊 FE 出水浊度二分类")

    user_threshold = st.slider(
        '🚦 **出水浊度判定阈值 (NTU)**',
        min_value=0.05, max_value=2.0, value=0.3, step=0.05,
        help='出水浊度 ≥ 此值判定为 Fail。模型训练时使用 0.3 NTU，更改阈值仅影响显示判定。'
    )
    st.info(f"💡 输入水质参数和当前铁盐投加量，模型预测出水浊度 ≥ {user_threshold} NTU 的概率。")

    feature_info = get_feature_info('FE', 'turbidity')
    with st.form('fe_turb_form'):
        inputs = build_input_form(feature_info)
        submitted = st.form_submit_button('🔮 开始预测', use_container_width=True)
    if submitted:
        try:
            with st.spinner('预测中...'):
                result = predict_turbidity('FE', inputs)
            display_turbidity_result(result, user_threshold)
        except Exception as e:
            st.error(f"预测出错: {e}")
            st.code(traceback.format_exc())

st.markdown("---")
st.caption("ICR Coagulant Prediction System | CatBoost | Streamlit")
