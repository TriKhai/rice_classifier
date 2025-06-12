import pandas as pd
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_excel("https://github.com/ltdaovn/dataset/raw/master/Rice2024.xlsx")
print(data.head())

# Tải dữ liệu Iris
# print("Đang tải dữ liệu Iris...")
# iris = load_iris()
# data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
# data['species'] = iris.target_names[iris.target]

print(data['Class'].value_counts())
data['Class']=data['Class'].str.upper()
print(data['Class'].value_counts())

class_counts = data['Class'].value_counts()
plt.figure(figsize=(8, 5))
sns.barplot(x=class_counts.index, y=class_counts.values, palette='viridis')
plt.title('Số lượng mẫu theo lớp')
plt.xlabel('Lớp')
plt.ylabel('Số lượng mẫu')
plt.show()

# # Cấu hình
# TARGET_COLUMN = "species"
# FEATURE_COLUMNS = iris.feature_names  # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
# MODEL_PATH = "random_forest_iris_model.pkl"
# FEATURES_PATH = "iris_feature_columns.pkl"
# DESCRIPTION_PATH = "iris_data_description.pkl"
#
# # Kiểm tra cột
# if TARGET_COLUMN not in data.columns or not all(col in data.columns for col in FEATURE_COLUMNS):
#     print("Lỗi: Cột mục tiêu hoặc đặc trưng không tồn tại trong dữ liệu!")
#     print("Cột có sẵn:", list(data.columns))
#     exit()
#
# X = data[FEATURE_COLUMNS]
# y = data[TARGET_COLUMN]
#
# # Chia dữ liệu
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Huấn luyện mô hình
# print("Đang huấn luyện mô hình...")
# rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
# rf_model.fit(X_train, y_train)
#
# # Đánh giá mô hình
# y_pred = rf_model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Độ chính xác trên tập kiểm tra: {accuracy:.2f}")
# print("\nBáo cáo phân loại:")
# print(classification_report(y_test, y_pred))
#
# # Lưu mô hình, cột đặc trưng và thống kê mô tả
# print("Đang lưu mô hình và thông tin...")
# joblib.dump(rf_model, MODEL_PATH)
# joblib.dump(FEATURE_COLUMNS, FEATURES_PATH)
# joblib.dump(data[FEATURE_COLUMNS].describe(), DESCRIPTION_PATH)
# print(f"Mô hình đã được lưu vào '{MODEL_PATH}'")
# print(f"Danh sách cột đặc trưng đã được lưu vào '{FEATURES_PATH}'")
# print(f"Thống kê mô tả đã được lưu vào '{DESCRIPTION_PATH}'")