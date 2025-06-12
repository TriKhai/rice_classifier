## Member List 

1. Nguyễn Hoàng Tiến Danh - 0869635xxx - B2205858 - danhb2205858@student.ctu.edu.vn
2. Thân Quốc Tuấn - 0584020xxx - B2207574 - tuanb220574@student.ctu.edu.vn
3. Lý Trí Khải - 0848041xxx - B2207530 - khaib2207530@student.ctu.edu.vn

## Technologies Used:

- Python 3.10
- Streamlit

## Task

Phân tích tập dữ liệu Rice2024
(https://github.com/ltdaovn/dataset/raw/master/Rice2024.xlsx)

- Tìm hiểu dữ liệu
- Tiền xử lý dữ liệu
- Phân tích trực quan
- Xây dựng mô hình máy học (hồi quy, phân lớp, gom nhóm, …)
- Đánh giá mô hình
- Xây dựng website để dự đoán loại gạo

## Application

### Mô tả
undating...

### Yêu cầu

- Python 3.10

### Cài đặt

1. Clone repo frontend (nếu chưa)

```bash
git clone <url-repo-github>
cd <thư-mục-rice_classifier>
```

2. Cài đặt thư viện
```bash
pip install -r requirements.txt
```

3. Chạy mô hình
```bash
python machine_learning/preprocess.py
python machine_learning/train_model.py
```

4. Chạy app
```bash
streamlit run app.py
```
