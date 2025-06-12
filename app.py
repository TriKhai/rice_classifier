import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from machine_learning.config import MODEL_PATH, FEATURES_PATH, DESCRIPTION_PATH, TARGET_COLUMN, SCALER_PATH

model_path = os.path.join("machine_learning", MODEL_PATH)
features_path = os.path.join("machine_learning", FEATURES_PATH)
description_path = os.path.join("machine_learning", DESCRIPTION_PATH)
scaler_path = os.path.join("machine_learning", SCALER_PATH)

# print(model_path)

# Tiêu đề ứng dụng
st.title("Ứng dụng Phân loại Gạo bằng Random Forest")

# Tải mô hình, scaler và thông tin
st.header("1. Tải Mô hình và Thông tin")
try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_columns = joblib.load(features_path)
    data_description = joblib.load(description_path)
    st.success("Tải mô hình, scaler và thông tin thành công!")
    st.session_state['model'] = model
    st.session_state['scaler'] = scaler
    st.session_state['feature_columns'] = feature_columns
    st.session_state['data_description'] = data_description
except FileNotFoundError:
    st.error("Không tìm thấy mô hình, scaler hoặc thông tin. Vui lòng chạy 'preprocess.py' và 'train_model.py' trước.")
    st.stop()

# Giao diện nhập liệu và dự đoán
st.header("2. Dự đoán Giống Gạo")
with st.form(key="prediction_form"):
    st.subheader("Nhập thông tin đặc trưng")
    input_data = {}
    for col in st.session_state['feature_columns']:
        min_val = data_description[col]['min']
        max_val = data_description[col]['max']
        default_val = data_description[col]['mean']
        input_data[col] = st.number_input(
            f"{col} (Phạm vi gợi ý: {min_val:.2f} - {max_val:.2f})",
            min_value=float(min_val),
            max_value=float(max_val),
            value=float(default_val),
            step=0.01
        )
    submit_button = st.form_submit_button(label="Dự đoán")

if submit_button:
    input_df = pd.DataFrame([input_data])
    # Chuẩn hóa dữ liệu đầu vào
    input_scaled = pd.DataFrame(st.session_state['scaler'].transform(input_df), columns=input_df.columns)
    prediction = st.session_state['model'].predict(input_scaled)[0]
    probabilities = st.session_state['model'].predict_proba(input_scaled)[0]
    prob_dict = {st.session_state['model'].classes_[i]: prob for i, prob in enumerate(probabilities)}

    # Hiển thị kết quả
    st.success(f"**Kết quả dự đoán:** {prediction}")
    st.write("**Xác suất dự đoán:**")
    for cls, prob in prob_dict.items():
        st.write(f"{cls}: {prob:.2%}")

    # Lưu kết quả dự đoán vào CSV
    result = pd.DataFrame({
        'Prediction': [prediction],
        **input_data,
        **{f"Prob_{cls}": [prob] for cls, prob in prob_dict.items()}
    })
    result.to_csv('predictions.csv', mode='a', index=False, header=not pd.io.common.file_exists('predictions.csv'))
    st.write("Kết quả đã được lưu vào 'predictions.csv'")

# Hiển thị độ quan trọng của đặc trưng
if st.checkbox("Hiển thị độ quan trọng của đặc trưng"):
    model = st.session_state['model']
    feature_importance = model.feature_importances_
    st.write("**Độ quan trọng của đặc trưng:**")

    # Tạo biểu đồ bằng matplotlib
    fig, ax = plt.subplots()
    ax.bar(feature_columns, feature_importance, color=['#4CAF50', '#2196F3', '#FF9800', '#F44336'])
    ax.set_xlabel("Đặc trưng")
    ax.set_ylabel("Độ quan trọng")
    ax.set_title("Độ quan trọng của các đặc trưng")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

# Hiển thị phân bố lớp
if st.checkbox("Hiển thị phân bố lớp"):
    st.image("class_distribution.png", caption="Phân bố các lớp trong tập dữ liệu")

# # Hướng dẫn sử dụng
# st.sidebar.header("Hướng dẫn")
# st.sidebar.write("""
# 1. Chạy 'preprocess.py' để tiền xử lý, tách và chuẩn hóa dữ liệu.
# 2. Chạy 'train_model.py' để huấn luyện và lưu mô hình.
# 3. Nhập các giá trị đặc trưng vào form.
# 4. Nhấn "Dự đoán" để xem kết quả và xác suất.
# 5. Tích vào ô để xem độ quan trọng của đặc trưng hoặc phân bố lớp.
# """)