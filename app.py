import streamlit as st
# ── 산점도 시각화 ──────────────────────────────────────────
st.markdown(
    '<p class="section-header" style="margin-top:1.5rem;">📊 군집 시각화</p>',
    unsafe_allow_html=True
)



color_map = {
    0: "#4caf50",  # 매우 건강군
    1: "#e91e63",  # 위험군
    2: "#ffc107"   # 건강군
}

cluster_colors = df['cluster'].map(color_map)

fig, ax = plt.subplots(figsize=(8, 6))

# 전체 환자 데이터
ax.scatter(
    df['smoking'],
    df['alcohol'],
    c=cluster_colors,
    alpha=0.7,
    s=60,
    edgecolors='white',
    linewidths=0.5
)

# 현재 입력 환자
ax.scatter(
    smoking,
    alcohol,
    marker='*',
    s=300,
    color='blue',
    edgecolors='white',
    linewidths=1,
    label='현재 환자'
)

# 범례
from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor='#4caf50', label='0번 : 매우 건강군'),
    Patch(facecolor='#ffc107', label='2번 : 건강군'),
    Patch(facecolor='#e91e63', label='1번 : 위험군')
]

ax.legend(
    handles=legend_elements,
    loc='upper left'
)

ax.set_xlabel('흡연량')
ax.set_ylabel('음주량')
ax.set_title('폐암 환자 군집 분석')

ax.grid(True, linestyle='--', alpha=0.3)

st.pyplot(fig)
kmeans, scaler, df, label_map = train_model()
