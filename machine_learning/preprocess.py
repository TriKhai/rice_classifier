import pandas as pd
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from config import DATA_PATH, TARGET_COLUMN, FEATURE_COLUMNS, DATA_OUTPUT, FEATURES_PATH, DESCRIPTION_PATH
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# data = pd.read_excel(DATA_URL)
# print(data.head())

# Tải dữ liệu
print(f"Đang tải dữ liệu từ {DATA_PATH}")
try:
    data = pd.read_excel(DATA_PATH)
    print("Tải dữ liệu thành công!")
    print("\n5 dòng đầu tiên của dữ liệu:")
    print(data.head())
except Exception as e:
    print(f"Lỗi khi tải dữ liệu: {e}")
    exit()

# handle class
print(data['Class'].value_counts())
data['Class']=data['Class'].str.upper()
print(data['Class'].value_counts())

# class_counts = data['Class'].value_counts()
# plt.figure(figsize=(8, 5))
# sns.barplot(x=class_counts.index, y=class_counts.values, hue=class_counts.index, palette='viridis', legend=False)
# plt.title('Số lượng mẫu theo lớp')
# plt.xlabel('Lớp')
# plt.ylabel('Số lượng mẫu')
# plt.show()

# handle
print(data.isnull().sum())
# # Kiểm tra dữ liệu thiếu
# print("\nKiểm tra dữ liệu thiếu...")
# if data.isnull().sum().sum() > 0:
#     print("Cảnh báo: Dữ liệu có giá trị thiếu!")
#     data = data.fillna(data.mean(numeric_only=True))
#
# # Kiểm tra cột
# if TARGET_COLUMN not in data.columns or not all(col in data.columns for col in FEATURE_COLUMNS):
#     print("Lỗi: Cột mục tiêu hoặc đặc trưng không tồn tại trong dữ liệu!")
#     print("Cột có sẵn:", list(data.columns))
#     exit()

data = data.drop(['Id', 'Nickname'], axis=1)
X = data.drop('Class', axis=1)
y = data['Class']

def clean_number(value):
    if isinstance(value, str):
        value = value.replace(',', '')
    try:
        return float(value)
    except:
        return None  # hoặc np.nan

for col in X.columns:
    X[col] = X[col].apply(clean_number)

# Missing data
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Encoding
le = LabelEncoder()
y = le.fit_transform(y)

class_counts = data[TARGET_COLUMN].value_counts()
print("\nPhân bố lớp:")
print(class_counts)

# Vẽ biểu đồ phân bố lớp
plt.figure(figsize=(8, 5))
sns.barplot(x=class_counts.index, y=class_counts.values, hue=class_counts.index, palette='viridis', legend=False)
plt.title('Số lượng mẫu theo lớp')
plt.xlabel('Lớp')
plt.ylabel('Số lượng mẫu')
plt.savefig('class_distribution.png')
plt.close()
print("Đã lưu biểu đồ phân bố lớp vào 'class_distribution.png'")

# Lưu dữ liệu đã xử lý
y = pd.Series(y, name='Class')
data_processed = pd.concat([X, y], axis=1)

print("\nĐang lưu dữ liệu đã xử lý...")
joblib.dump(data_processed, DATA_OUTPUT)
joblib.dump(FEATURE_COLUMNS, FEATURES_PATH)
joblib.dump(data_processed[FEATURE_COLUMNS].describe(), DESCRIPTION_PATH)
print(f"Dữ liệu đã xử lý đã được lưu vào '{DATA_OUTPUT}'")
print(f"Danh sách cột đặc trưng đã được lưu vào '{FEATURES_PATH}'")
print(f"Thống kê mô tả đã được lưu vào '{DESCRIPTION_PATH}'")