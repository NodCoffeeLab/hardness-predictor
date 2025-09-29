# =================================================================
#                 Green Bean Hardness Prediction Model (beta)
# =================================================================
import streamlit as st
import pandas as pd
import joblib

# --- 1. Load Model and Tools ---
@st.cache_data
def load_resources():
    try:
        model = joblib.load('final_ordinal_model.pkl')
        scaler = joblib.load('scaler.pkl')
        encoder = joblib.load('encoder.pkl')
        train_mean = joblib.load('train_mean.pkl')
        df = pd.read_excel('HardnessData.xlsx') 
        countries = sorted(df['Country'].dropna().unique().tolist())
    except FileNotFoundError:
        return None, None, None, None, None
    return model, scaler, encoder, train_mean, countries

model, scaler, encoder, train_mean, countries = load_resources()

if model is None:
    st.error("Model files not found. Please make sure all .pkl files and HardnessData.xlsx are in the same folder as app.py.")
    st.stop()

# --- Bilingual Labels ---
labels_ko = ['1단계(낮음)', '2단계(조금 낮음)', '3단계(중간)', '4단계(조금 높음)', '5단계(높음)']
labels_en = ['Level 1 (Low)', 'Level 2 (Slightly Low)', 'Level 3 (Medium)', 'Level 4 (Slightly High)', 'Level 5 (High)']

# --- 2. Web App Interface ---
st.set_page_config(page_title="Green Bean Hardness Prediction Model", layout="wide")
st.title("생두 경도 예측 모델 (베타) / Green Bean Hardness Prediction Model (beta)")

# --- Introduction Section using Tabs ---
intro_ko, intro_en = st.tabs(["소개 (Korean)", "Introduction (English)"])
with intro_ko:
    st.header("소개")
    st.markdown("노드 커피랩에서 개발 중인 **'디플루이드 오믹스 측정 데이터 기반 생두 경도 예측 모델'**입니다. 현재는 생두 경도 예측 모델의 기본 기능을 테스트하는 단계이며, 추후 미구현된 기능을 업데이트할 예정입니다.")
    st.subheader("모델의 목표와 배경")
    st.markdown("생두의 경도는 로스팅 중 발열 반응, 1차 크랙, 세포벽 분해와 관련되는 매우 중요한 요소입니다. 하지만 이를 직접 측정하는 장비가 없어 객관적인 데이터로 활용하기 어려웠습니다. 본 예측 모델은 생두의 측정 데이터와 이력 정보를 토대로 경도를 예측하여, 이를 로스팅 변수로 활용할 수 있도록 하는 것을 목표로 합니다.")
    st.warning("**주의사항**: 경도는 밀도, 수분, 재배 고도 등 여타 모든 요소와 구분되는 독립된 변수이므로 예측 정확성은 100%가 될 수 없습니다. 예측된 경도 정보는 **보조적인 참고 자료**로만 활용해 주시기를 바랍니다.")
    with st.expander("데이터 측정 방법 보기"):
        st.subheader("계량 방법")
        st.markdown("예측 모델의 기반 데이터는 디플루이드 코리아에서 개발/판매하는 분배기를 사용하여 측정된 값입니다. 오믹스에 기본으로 포함된 자를 사용하여 측정하는 경우에는, 밀도값에 **1.065**를 곱하여 입력해 주세요.\n1. 샘플 컨테이너로부터 약 10cm 높이에서 생두가 소복하게 쌓이도록 붓습니다.\n2. 분배기를 올린 다음 화살표에 손가락을 걸고 살짝 눌러가며 5회 이상, 생두가 움직이지 않을 때까지 돌립니다.\n3. 자를 비스듬하게 세워 동-서-남-북으로 밀어 평탄하게 만듭니다. 이때 자는 컨테이너에 닿은 상태를 유지해야 합니다.\n4. 컨테이너를 오믹스에 체결하여 측정합니다. (수분활성도는 약 30초 경과 후 측정)")
with intro_en:
    st.header("Introduction")
    st.markdown("This is the **'Deefluid Omix Data-based Green Bean Hardness Prediction Model (beta)'** under development at Node Coffee Lab. This is the initial testing phase for the model's basic functions. Additional features will be implemented in future updates.")
    st.subheader("Model Goal and Background")
    st.markdown("Green bean hardness is a critical factor related to exothermic reactions, first crack, and cell wall degradation during roasting. However, the lack of direct measurement equipment has made it difficult to use hardness as an objective variable. This model aims to estimate green bean hardness based on physical measurements and historical data, making it a usable variable for roasting.")
    st.warning("**Disclaimer**: Since hardness is an independent variable distinct from density, moisture, altitude, and other factors, the prediction accuracy cannot be 100%. Please use the predicted hardness information as **supplementary reference material only.**")
    with st.expander("See Measurement Guide"):
        st.subheader("Measurement Method")
        st.markdown("The model's base data was measured using a distributor from Deefluid Korea. If using the default Omix ruler, please multiply the measured density value by **1.065** before input.\n1. Pour beans from a height of about 10 cm into the sample container to form a slight mound.\n2. Place the distributor on top and turn it at least 5 times at a moderate speed until the beans settle.\n3. Use a ruler, held diagonally, to level the beans by pushing from four cardinal directions. The ruler must remain in contact with the container.\n4. Place the container in the Omix device to take the measurement. (Measure water activity after about 30 seconds.)")

st.markdown("---")

# --- 3. Input Fields ---
col1, col2 = st.columns(2)
with col1:
    st.header("오믹스 측정값 / Omix Measured Value")
    density = st.number_input("부피 밀도 / Bulk Density (g/L)", value=None, placeholder="800.0", format="%.1f")
    humidity = st.number_input("수분 / Moisture Content (%)", value=None, placeholder="11.0", format="%.1f")
    # --- 수정된 부분 1 ---
    volume = st.number_input("부피 / Volume (mm³)", value=None, placeholder="170", format="%.0f")
    
    st.header("부가 정보 / Extra Information")
    # --- 수정된 부분 2 ---
    altitude = st.number_input("재배 고도 / Cultivation Altitude (m)", value=None, placeholder="1650", format="%.0f")
    
    country_options = [""] + countries + ["직접 입력 / Other"]
    selected_country = st.selectbox("생산 국가 / Country", options=country_options)
    if selected_country == "직접 입력 / Other":
        country = st.text_input("생산 국가 직접 입력 / Enter Country Manually", placeholder="e.g., Ethiopia")
    else:
        country = selected_country

    variety = st.selectbox("종 / Variety", ["", "Arabica", "Robusta", "Liberica"])
    processing = st.selectbox("가공 / Processing", ["", "Water Process Decaffeination", "Future feature / 추후 지원 예정"])
    cultivar = st.text_input("재배종 / Cultivar", placeholder="Future feature / 추후 지원 예정", disabled=True)

with col2:
    st.header("오믹스 데이터 보정 / Omix data correction")
    st.info("This is a future feature and is currently disabled. / 추후 연구하여 추가할 기능입니다.")
    st.number_input("측정 데이터 중 부피 밀도 최대값 / Max Bulk Density", disabled=True, value=None)
    st.number_input("측정 데이터 중 부피 밀도 최소값 / Min Bulk Density", disabled=True, value=None)
    st.text_input("측정 데이터 중 부피 밀도 중간값 범위 / Median Bulk Density", disabled=True)
    st.number_input("측정 데이터 중 수분 최대값 / Max Moisture Content", disabled=True, value=None)
    st.number_input("측정 데이터 중 수분 최소값 / Min Moisture Content", disabled=True, value=None)
    st.text_input("측정 데이터 중 수분 중간값 범위 / Median Moisture Content", disabled=True)

# --- 4. Prediction Logic ---
if st.button("Predict Hardness / 경도 예측하기", use_container_width=True, type="primary"):
    if any(v is None for v in [density, humidity, volume, altitude]):
        st.warning("Please fill in all essential fields. / 필수 항목(밀도, 수분, 부피, 고도)을 모두 입력해주세요.")
    else:
        # ... (Prediction logic is the same)
        HIGH_ALTITUDE_THRESHOLD = 1800
        mean_density_val = train_mean['Omix bulk Density']
        std_density_val = 36.6
        density_x_volume = density * volume
        altitude_x_volume = altitude * volume
        is_high_altitude = 1 if altitude >= HIGH_ALTITUDE_THRESHOLD else 0
        is_extreme_density = 1 if (density < (mean_density_val - std_density_val) or density > (mean_density_val + std_density_val)) else 0
        is_brazil = 1 if (country and 'brazil' in country.lower()) else 0
        is_robusta = 1 if (variety and 'robusta' in variety.lower()) else 0
        is_decaf = 1 if (processing and 'decaf' in processing.lower()) else 0
        input_data = {'Omix Humidity': humidity, 'Omix bulk Density': density, 'Omix Volume': volume, 'Altitude': altitude,
                      'Density_x_Volume': density_x_volume, 'Altitude_x_Volume': altitude_x_volume,
                      'is_high_altitude': is_high_altitude, 'is_extreme_density': is_extreme_density,
                      'is_brazil': is_brazil, 'is_robusta': is_robusta, 'is_decaf': is_decaf}
        input_df = pd.DataFrame([input_data])
        input_df_filled = input_df.fillna(train_mean)
        input_df_scaled = scaler.transform(input_df_filled)
        prediction_numeric = model.predict(input_df_scaled)[0]
        probabilities = model.predict_proba(input_df_scaled)[0]
        
        st.subheader("Prediction Results / 예측 결과")
        ko_label = labels_ko[prediction_numeric]
        en_label = labels_en[prediction_numeric]
        st.success(f"**예측된 경도 단계: {ko_label}**")
        st.success(f"**Predicted Hardness Level: {en_label}**")
        
        st.markdown("---")
        st.write("Prediction Confidence / 예측 확신도:")
        prob_series = pd.Series(probabilities, index=encoder.classes_).sort_values(ascending=False)
        for label_numeric, prob in prob_series.head(3).items():
            ko_label_prob = labels_ko[label_numeric]
            en_label_prob = labels_en[label_numeric]
            st.write(f"{ko_label_prob} / {en_label_prob}: **{prob:.1%}**")

