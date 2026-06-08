import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
font_list = [f.name for f in fm.fontManager.ttflist]

if 'Malgun Gothic' in font_list:
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif 'NanumGothic' in font_list:
    plt.rcParams['font.family'] = 'NanumGothic'

plt.rcParams['axes.unicode_minus'] = False
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ── 페이지 설정 ──────────────────────────────────────────────
st.set_page_config(
    page_title="폐암 환자 군집 분석 시스템",
    page_icon="🫁",
    layout="centered",
)

# ── CSS 스타일 ────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Noto Sans KR', sans-serif;
    }
    .main { background-color: #fdf6f0; }

    .title-box {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }
    .title-box h1 {
        font-size: 2rem;
        font-weight: 700;
        color: #2d2d2d;
    }
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 0.9rem;
        margin-bottom: 2rem;
    }
    hr { border: none; border-top: 1px solid #e8ddd4; margin: 1.5rem 0; }

    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #2d2d2d;
        margin-bottom: 1rem;
    }

    /* 결과 박스 */
    .result-healthy {
        background: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 0.8rem 1.2rem;
        border-radius: 6px;
        color: #2e7d32;
        font-weight: 600;
        margin: 1rem 0;
    }
    .result-caution {
        background: #fff8e1;
        border-left: 4px solid #ffc107;
        padding: 0.8rem 1.2rem;
        border-radius: 6px;
        color: #f57f17;
        font-weight: 600;
        margin: 1rem 0;
    }
    .result-danger {
        background: #fce4ec;
        border-left: 4px solid #e91e63;
        padding: 0.8rem 1.2rem;
        border-radius: 6px;
        color: #880e4f;
        font-weight: 600;
        margin: 1rem 0;
    }
    .info-text {
        font-size: 0.85rem;
        color: #666;
        margin-bottom: 0.5rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #ff7043, #f4511e);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
        cursor: pointer;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.88; }
</style>
""", unsafe_allow_html=True)

# ── 샘플 데이터 & 모델 학습 ────────────────────────────────────
@st.cache_resource
def train_model():
    np.random.seed(42)
    n = 120

    # 군집 0: 매우 건강군 (흡연↓, 음주↓, 나이 다양)
    c0 = pd.DataFrame({
        'age':     np.random.normal(45, 10, n//3).clip(20, 80),
        'smoking': np.random.uniform(0, 8,   n//3),
        'alcohol': np.random.uniform(0, 3,   n//3),
        'label':   0
    })
    # 군집 1: 위험군 (흡연↑, 음주↑)
    c1 = pd.DataFrame({
        'age':     np.random.normal(60, 8, n//3).clip(30, 85),
        'smoking': np.random.uniform(15, 35, n//3),
        'alcohol': np.random.uniform(4, 9,  n//3),
        'label':   1
    })
    # 군집 2: 건강군 (흡연 중간, 음주 중간)
    c2 = pd.DataFrame({
        'age':     np.random.normal(50, 12, n//3).clip(25, 80),
        'smoking': np.random.uniform(5, 18, n//3),
        'alcohol': np.random.uniform(2, 6,  n//3),
        'label':   2
    })

    df = pd.concat([c0, c1, c2], ignore_index=True)
    features = df[['age', 'smoking', 'alcohol']].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X_scaled)

    # 센터 기준으로 군집 의미 재정렬 (흡연량 기준)
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    smoking_order = np.argsort(centers[:, 1])  # 흡연 낮은→높은
    label_map = {smoking_order[0]: 0, smoking_order[2]: 1, smoking_order[1]: 2}

    return kmeans, scaler, df, label_map

kmeans, scaler, df, label_map = train_model()

CLUSTER_INFO = {
    0: {"name": "매우 건강군", "color": "#4caf50", "css": "result-healthy",  "emoji": "✅"},
    1: {"name": "위험군",     "color": "#e91e63", "css": "result-danger",   "emoji": "⚠️"},
    2: {"name": "건강군",     "color": "#ffc107", "css": "result-caution",  "emoji": "🟡"},
}

# ── 헤더 ──────────────────────────────────────────────────────
st.markdown('<div class="title-box"><h1>🫁 폐암 환자 군집 분석 시스템</h1></div>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI가 환자의 특성을 분석하여<br>어떤 군집(유형)에 속하는지 예측합니다.</p>', unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# ── 입력 섹션 ─────────────────────────────────────────────────
st.markdown('<p class="section-header">📋 환자 정보 입력</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    age     = st.number_input("나이",   min_value=1.0,  max_value=120.0, value=50.0, step=1.0, format="%.2f")
with col2:
    smoking = st.number_input("흡연량", min_value=0.0,  max_value=40.0,  value=10.0, step=0.5, format="%.2f")
with col3:
    alcohol = st.number_input("음주량", min_value=0.0,  max_value=20.0,  value=5.0,  step=0.5, format="%.2f")

st.markdown("<br>", unsafe_allow_html=True)

# ── 분석 버튼 ─────────────────────────────────────────────────
if st.button("🔍 군집 분석하기"):
    patient = np.array([[age, smoking, alcohol]])
    patient_scaled = scaler.transform(patient)
    raw_label = kmeans.predict(patient_scaled)[0]
    cluster_id = label_map.get(raw_label, raw_label)
    info = CLUSTER_INFO[cluster_id]

    # 결과 표시
    st.markdown(
        f'<div class="{info["css"]}">{info["emoji"]} 이 환자는 <strong>{cluster_id}번 군집</strong>에 속합니다. ({info["name"]})</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="info-text">0번은 매우 건강군, 1번은 위험군, 2번은 건강군입니다.</p>',
        unsafe_allow_html=True
    )

    # ── 산점도 시각화 ──────────────────────────────────────────
    st.markdown('<p class="section-header" style="margin-top:1.5rem;">📊 군집 시각화</p>', unsafe_allow_html=True)

    # 전체 데이터에 예측 라벨 부여
    X_all = df[['age', 'smoking', 'alcohol']].values
    X_all_scaled = scaler.transform(X_all)
    raw_labels = kmeans.predict(X_all_scaled)
    df['cluster'] = [label_map.get(l, l) for l in raw_labels]

    color_map = {0: "#4caf50", 1: "#e91e63", 2: "#ffc107"}
    cluster_colors = df['cluster'].map(color_map)

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor('#fdf6f0')
    ax.set_facecolor('#fdf6f0')

    scatter = ax.scatter(
        df['smoking'], df['alcohol'],
        c=cluster_colors, alpha=0.7, s=55, edgecolors='white', linewidths=0.5
    )

    # 현재 환자 표시
    ax.scatter(smoking, alcohol, marker='*', s=280, color='#1565c0',
               zorder=5, edgecolors='white', linewidths=0.8, label='현재 환자')

    # 범례
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4caf50', label='0: 매우 건강군'),
        Patch(facecolor='#ffc107', label='2: 건강군'),
        Patch(facecolor='#e91e63', label='1: 위험군'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=8,
              framealpha=0.6, facecolor='white')

    ax.set_xlabel("흡연량", fontsize=10)
    ax.set_ylabel("음주량", fontsize=10)
    ax.set_title("군집 시각화", fontsize=12, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)

    # ── 상세 분석 카드 ─────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<p class="section-header">📈 상세 분석</p>', unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("나이",   f"{age:.0f}세")
    with m2:
        st.metric("흡연량", f"{smoking:.1f}")
    with m3:
        st.metric("음주량", f"{alcohol:.1f}")

    advice = {
        0: "현재 매우 건강한 상태입니다. 꾸준한 생활 습관을 유지하세요! 🎉",
        1: "흡연·음주량이 높아 폐암 위험군입니다. 즉시 전문의 상담을 권장합니다. 🏥",
        2: "비교적 건강하지만 생활 습관 개선이 도움이 됩니다. 정기 검진을 권장합니다. 🩺",
    }
    st.info(advice[cluster_id])
