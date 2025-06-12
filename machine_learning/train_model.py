import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler

from config import DATA_OUTPUT, MODEL_PATH, FEATURES_PATH, TARGET_COLUMN, SCALER_PATH

# Tải dữ liệu đã xử lý
print("Đang tải dữ liệu đã xử lý...")
try:
    data = joblib.load(DATA_OUTPUT)
    FEATURE_COLUMNS = joblib.load(FEATURES_PATH)
except FileNotFoundError:
    print("Lỗi: Không tìm thấy tệp dữ liệu hoặc cột đặc trưng. Vui lòng chạy 'preprocess.py' trước.")
    exit()

# Chuẩn bị dữ liệu
X = data[FEATURE_COLUMNS]
y = data[TARGET_COLUMN]

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#chuẩn hóa
scaler = MinMaxScaler()
X_train= pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# Huấn luyện mô hình
print("Đang huấn luyện mô hình...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Đánh giá mô hình
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Độ chính xác trên tập kiểm tra: {accuracy:.2f}")
print("\nBáo cáo phân loại:")
print(classification_report(y_test, y_pred))

# Lưu mô hình
print("Đang lưu mô hình...")
joblib.dump(rf_model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
print(f"Mô hình đã được lưu vào '{MODEL_PATH}'")