import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ==========================
# 한글 깨짐 방지
# ==========================
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ==========================
# 모델 불러오기
# ==========================
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# ==========================
# 화면 제목
# ==========================
st.title("폐암 환자 군집 분류 시스템")
st.write("환자 정보를 입력하면 군집을 예측합니다.")

# ==========================
# 사용자 입력
# ==========================
patient_age = st.number_input(
    "나이",
    min_value=0.0,
    max_value=120.0,
    value=50.0
)

patient_air_quality = st.number_input(
    "공기질",
    min_value=0.0,
    value=50.0
)

patient_smoking = st.number_input(
    "흡연량",
    min_value=0.0,
    value=10.0
)

# ==========================
# 예측 버튼
# ==========================
if st.button("군집 예측"):

    new_patient = pd.DataFrame(
        [[patient_age, patient_air_quality, patient_smoking]],
        columns=["나이", "공기질", "흡연량"]
    )

    # 스케일링
    new_patient_scaled = scaler.transform(new_patient)

    # 군집 예측
    pred_cluster = model.predict(new_patient_scaled)

    cluster_num = int(pred_cluster[0])

    st.success(
        f"이 환자는 {cluster_num}번 군집에 속합니다."
    )

    # ==========================
    # 군집별 색상
    # ==========================
    cluster_colors = {
        0: "red",
        1: "blue",
        2: "green",
        3: "pink"
    }

    # ==========================
    # 그래프 생성
    # ==========================
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        patient_age,
        patient_smoking,
        color=cluster_colors.get(cluster_num, "black"),
        s=300
    )

    ax.set_xlabel("나이")
    ax.set_ylabel("흡연량")
    ax.set_title("폐암 환자 군집 분석")

    st.pyplot(fig)
