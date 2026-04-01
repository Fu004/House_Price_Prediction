# 🏠 House Price Prediction - Ames Housing Dataset

Đồ án môn học: Xây dựng hệ thống dự báo giá nhà sử dụng Machine Learning và Streamlit.

## 📁 Cấu trúc dự án

```
house_price_project/
│
├── data/
│   └── AmesHousing.csv          # Dataset gốc (copy vào đây)
│
├── models/                       # Tự động tạo sau khi chạy model_training.py
│   ├── linear_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   ├── feature_names.pkl
│   └── categorical_options.pkl
│
├── model_training.py             # Phần 1: Tiền xử lý + Huấn luyện mô hình
├── app.py                        # Phần 2: Giao diện Streamlit
├── requirements.txt              # Danh sách thư viện
└── README.md                     # File này
```

## 🚀 Hướng dẫn chạy

### Bước 1: Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### Bước 2: Copy dataset
Đặt file `AmesHousing.csv` vào thư mục `data/`.

### Bước 3: Huấn luyện mô hình (Sinh viên 1 & 2 phụ trách)
```bash
python model_training.py
```

### Bước 4: Chạy giao diện web (Sinh viên 3 phụ trách)
```bash
streamlit run app.py
```

Mở trình duyệt tại: http://localhost:8501

## 👥 Phân công nhóm gợi ý

| Thành viên | Phụ trách |
|---|---|
| Sinh viên 1 | `model_training.py` - Phần EDA, tiền xử lý, đánh giá mô hình |
| Sinh viên 2 | `model_training.py` - Phần huấn luyện 3 mô hình, lưu artifacts |
| Sinh viên 3 | `app.py` - Toàn bộ giao diện Streamlit |

## 📊 Mô hình sử dụng

- **Linear Regression**: Baseline, hồi quy tuyến tính cơ bản
- **Random Forest**: Ensemble learning, 200 cây quyết định
- **XGBoost**: Gradient Boosting, hiệu suất cao nhất
