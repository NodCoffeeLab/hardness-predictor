# Import necessary libraries
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Load Model and Tools ---
# This part loads the saved files. The files must be in the same folder as this app.py file.
# 저장된 파일들을 불러옵니다. 이 파일들은 app.py와 같은 폴더에 있어야 합니다.
try:
    model = joblib.load('final_ordinal_model.pkl')
    scaler = joblib.load('scaler.pkl')
    encoder = joblib.load('encoder.pkl')
    train_mean = joblib.load('train_mean.pkl')
except FileNotFoundError:
    st.error("Model files not found. Please make sure all .pkl files are in the same folder as app.py.")
    st.stop()


# --- Bilingual Labels ---
# 한/영 라벨 (나중에 결과를 표시할 때 사용)
labels_ko = ['1단계(낮음)', '2단계(조금 낮음)', '3단계(중간)', '4단계(조금 높음)', '5단계(높음)']
labels_en = ['Level 1 (Low)', 'Level 2 (Slightly Low)', 'Level 3 (Medium)', 'Level 4 (Slightly High)', 'Level 5 (High)']


# --- Web App Interface ---
st.set_page_config(page_title="Green Bean Hardness Predictor", layout="wide")

st.title("경도 예측 모델 / Green Bean Hardness Predictor")
st.markdown("Enter the green bean's data to predict its hardness level. / 생두의 데이터를 입력하여 경도 단계를 예측하세요.")

# --- Input Fields ---
col1, col2, col3 = st.columns(3)

with col1:
    st.header("Omix Data")
    density = st.number_input("Omix bulk Density (g/L)", min_value=600.0, max_value=900.0, value=750.0, step=0.1, format="%.1f")
    humidity = st.number_input("Omix Humidity (%)", min_value=8.0, max_value=13.0, value=11.0, step=0.1, format="%.1f")
    volume = st.number_input("Omix Volume", min_value=100.0, max_value=300.0, value=180.0, step=0.1, format="%.1f")

with col2:
    st.header("Cultivation & Context")
    altitude = st.number_input("Altitude (m)", min_value=0, max_value=3000, value=1650, step=10)
    country = st.text_input("Country (e.g., Brazil)", "Brazil")
    variety = st.text_input("Variety (e.g., Robusta)", "Catuai")

with col3:
    st.header("Details")
    details = st.text_input("Details (e.g., Decaf)", "")


# --- Prediction Logic ---
if st.button("Predict Hardness / 경도 예측하기", use_container_width=True):
    
    # --- Feature Engineering (same logic as in Colab) ---
    # Colab에서 했던 것과 동일한 로직으로 특성을 생성합니다.
    HIGH_ALTITUDE_THRESHOLD = 1800
    
    density_x_volume = density * volume
    altitude_x_volume = altitude * volume
    is_high_altitude = 1 if altitude >= HIGH_ALTITUDE_THRESHOLD else 0
    
    mean_density_val = train_mean['Omix bulk Density']
    std_density_val = 36.6 # This is a calculated value, hardcoding for simplicity. 실제 학습 시 계산된 값.
    is_extreme_density = 1 if (density < (mean_density_val - std_density_val) or density > (mean_density_val + std_density_val)) else 0
    
    is_brazil = 1 if (country and 'brazil' in country.lower()) else 0
    is_robusta = 1 if (variety and 'robusta' in variety.lower()) else 0
    is_decaf = 1 if (details and 'decaf' in details.lower()) else 0
    
    # Create a DataFrame for the input
    # 입력값을 데이터프레임으로 만듭니다.
    input_data = {
        'Omix Humidity': humidity,
        'Omix bulk Density': density,
        'Omix Volume': volume,
        'Altitude': altitude,
        'Density_x_Volume': density_x_volume,
        'Altitude_x_Volume': altitude_x_volume,
        'is_high_altitude': is_high_altitude,
        'is_extreme_density': is_extreme_density,
        'is_brazil': is_brazil,
        'is_robusta': is_robusta,
        'is_decaf': is_decaf
    }
    input_df = pd.DataFrame([input_data])
    
    # Fill missing values and scale
    # 누락값을 채우고 스케일링합니다.
    input_df_filled = input_df.fillna(train_mean)
    input_df_scaled = scaler.transform(input_df_filled)
    
    # --- Prediction ---
    # 예측 실행
    prediction_numeric = model.predict(input_df_scaled)[0]
    probabilities = model.predict_proba(input_df_scaled)[0]
    
    # --- Display Results ---
    # 결과 표시
    st.subheader("Prediction Results / 예측 결과")
    
    ko_label = labels_ko[prediction_numeric]
    en_label = labels_en[prediction_numeric]
    
    st.success(f"**Predicted Hardness Level: {en_label}**")
    st.success(f"**예측된 경도 단계: {ko_label}**")
    
    st.markdown("---")
    st.write("Prediction Confidence / 예측 확신도:")
    
    prob_series = pd.Series(probabilities, index=encoder.classes_).sort_values(ascending=False)
    
    for label_numeric, prob in prob_series.head(3).items():
        ko_label_prob = labels_ko[label_numeric]
        en_label_prob = labels_en[label_numeric]
        st.write(f"{ko_label_prob} / {en_label_prob}: **{prob:.1%}**")