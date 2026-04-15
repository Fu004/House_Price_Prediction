import pandas as pd
import numpy as np
import joblib
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# CÁC THAM SỐ CẤU HÌNH CHUNG
DATA_PATH = "data/AmesHousing.csv"   # Đường dẫn đến file dữ liệu
MODEL_DIR = "models"                   # Thư mục lưu model
RANDOM_STATE = 42                      # Seed để tái lập kết quả
TEST_SIZE = 0.2                        # Tỷ lệ dữ liệu test

# Các cột đặc trưng được chọn (quan trọng nhất, dễ hiểu, đủ đại diện)
NUMERIC_FEATURES = [
    "Gr Liv Area",      # Diện tích sống (trên mặt đất), sq ft
    "Total Bsmt SF",    # Diện tích tầng hầm
    "1st Flr SF",       # Diện tích tầng 1
    "Garage Area",      # Diện tích garage
    "Lot Area",         # Diện tích lô đất
    "Year Built",       # Năm xây dựng
    "Year Remod/Add",   # Năm cải tạo
    "Overall Qual",     # Chất lượng tổng thể (1-10)
    "Overall Cond",     # Tình trạng tổng thể (1-10)
    "TotRms AbvGrd",    # Tổng số phòng (không tính phòng tắm)
    "Full Bath",        # Số phòng tắm đầy đủ
    "Bedroom AbvGr",    # Số phòng ngủ
    "Fireplaces",       # Số lò sưởi
    "Garage Cars",      # Sức chứa garage (số xe)
]

CATEGORICAL_FEATURES = [
    "Neighborhood",     # Khu vực / Vị trí
    "House Style",      # Kiểu nhà (1 tầng, 2 tầng, v.v.)
    "Bldg Type",        # Loại tòa nhà (nhà riêng, song lập, v.v.)
    "Central Air",      # Có điều hòa trung tâm không (Y/N)
    "Kitchen Qual",     # Chất lượng bếp (Ex, Gd, TA, Fa, Po)
    "Exter Qual",       # Chất lượng ngoại thất
]

TARGET = "SalePrice"    # Biến mục tiêu: Giá bán nhà (USD)

# MAPPING KHOẢNG PHỐ TỪ MÃ VIẾT TẮT ĐẾN TÊN ĐẦY ĐỦ
NEIGHBORHOOD_MAP = {
    "Blmngtn": "Bloomington Heights",
    "Blueste": "Bluestem",
    "BrDale": "Briardale",
    "BrkSide": "Brookside",
    "ClearCr": "Clear Creek",
    "CollgCr": "College Creek",
    "Crawfor": "Crawford",
    "Edwards": "Edwards",
    "Gilbert": "Gilbert",
    "IDOTRR": "Iowa DOT and Rail Road",
    "Greens": "Greenshire",
    "GrnHill": "Green Hills",
    "Landmrk": "Landmark",
    "MeadowV": "Meadow Village",
    "Mitchel": "Mitchell",
    "Names": "North Ames",
    "NAmes": "North Ames",
    "NoRidge": "Northridge",
    "NPkVill": "Northpark Villa",
    "NridgHt": "Northridge Heights",
    "NWAmes": "Northwest Ames",
    "OldTown": "Old Town",
    "SWISU": "South & West of Iowa State University",
    "Sawyer": "Sawyer",
    "SawyerW": "Sawyer West",
    "Somerst": "Somerset",
    "StoneBr": "Stone Brook",
    "Timber": "Timberland",
    "Veenker": "Veenker"
}


# BƯỚC 1: ĐỌC VÀ KHÁM PHÁ DỮ LIỆU (EDA NHANH)
def load_and_explore(path: str) -> pd.DataFrame:
    """
    Đọc file CSV và in thông tin cơ bản về dataset.
    
    Args:
        path: Đường dẫn đến file CSV.
    Returns:
        DataFrame gốc chưa qua xử lý.
    """
    print("=" * 60)
    print("BƯỚC 1: ĐỌC DỮ LIỆU")
    print("=" * 60)

    df = pd.read_csv(path)
    
    # Thay thế mã viết tắt khoảng phố bằng tên đầy đủ
    df["Neighborhood"] = df["Neighborhood"].map(NEIGHBORHOOD_MAP)
    
    print(f"  Số dòng: {df.shape[0]:,} | Số cột: {df.shape[1]}")
    print(f"  Biến mục tiêu '{TARGET}': min={df[TARGET].min():,} | max={df[TARGET].max():,} | mean={df[TARGET].mean():,.0f}")

    # Kiểm tra missing values trong các cột sẽ dùng
    all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    missing = df[all_features + [TARGET]].isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        print("\n  Missing values phát hiện:")
        print(missing.to_string())
    else:
        print("  Không có missing values trong các cột được chọn.")

    return df

# BƯỚC 2: TIỀN XỬ LÝ DỮ LIỆU
def preprocess(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("BƯỚC 2: TIỀN XỬ LÝ DỮ LIỆU")
    print("=" * 60)

    # --- 2.1 Chọn cột cần dùng ---
    all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    df_selected = df[all_features + [TARGET]].copy()

    # --- 2.2 Xử lý Missing Values ---
    # Biến số: điền bằng trung vị (median) - ít bị ảnh hưởng bởi outlier
    for col in NUMERIC_FEATURES:
        if df_selected[col].isnull().any():
            median_val = df_selected[col].median()
            df_selected[col] = df_selected[col].fillna(median_val)
            print(f"  [Numeric] '{col}': điền missing bằng median = {median_val:.1f}")

    # Biến phân loại: điền bằng giá trị xuất hiện nhiều nhất (mode)
    for col in CATEGORICAL_FEATURES:
        if df_selected[col].isnull().any():
            mode_val = df_selected[col].mode()[0]
            df_selected[col] = df_selected[col].fillna(mode_val)
            print(f"  [Categorical] '{col}': điền missing bằng mode = '{mode_val}'")

    print("  Xử lý missing values hoàn tất.")

    # --- 2.3 Mã hóa biến phân loại (Label Encoding) ---
    # Lưu các LabelEncoder để dùng lại trong app.py
    label_encoders = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        df_selected[col] = le.fit_transform(df_selected[col].astype(str))
        label_encoders[col] = le
        print(f"  [Encode] '{col}': {len(le.classes_)} nhãn -> {list(le.classes_[:5])}...")

    # --- 2.4 Đảm bảo không còn NaN nào sót lại (an toàn) ---
    for col in NUMERIC_FEATURES:
        if df_selected[col].isnull().any():
            df_selected[col] = df_selected[col].fillna(df_selected[col].median())

    # --- 2.5 Tách X và y ---
    X = df_selected[all_features]
    y = df_selected[TARGET]
    feature_names = list(X.columns)

    # --- 2.6 Chia tập Train / Test ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"\n  Tập Train: {X_train.shape[0]} mẫu | Tập Test: {X_test.shape[0]} mẫu")

    # --- 2.7 Chuẩn hóa dữ liệu (StandardScaler) ---
    # Chỉ chuẩn hóa biến số, biến phân loại đã được mã hóa thành số
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[NUMERIC_FEATURES] = scaler.fit_transform(X_train[NUMERIC_FEATURES])
    X_test_scaled[NUMERIC_FEATURES] = scaler.transform(X_test[NUMERIC_FEATURES])

    print("  Chuẩn hóa StandardScaler hoàn tất.")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoders, feature_names


# =============================================================================
# BƯỚC 3: HUẤN LUYỆN 3 MÔ HÌNH
# =============================================================================

def train_models(X_train, y_train) -> dict:
    """
    Huấn luyện 3 mô hình: Linear Regression, Random Forest, XGBoost.

    Args:
        X_train: Tập đặc trưng huấn luyện đã qua tiền xử lý.
        y_train: Nhãn (giá nhà) của tập huấn luyện.
    Returns:
        Dictionary chứa các mô hình đã huấn luyện.
    """
    print("\n" + "=" * 60)
    print("BƯỚC 3: HUẤN LUYỆN MÔ HÌNH")
    print("=" * 60)

    models = {
        # Mô hình 1: Hồi quy tuyến tính - baseline đơn giản nhất
        "Linear Regression": LinearRegression(),

        # Mô hình 2: Random Forest - ensemble nhiều cây quyết định
        "Random Forest": RandomForestRegressor(
            n_estimators=200,       # Số cây
            max_depth=15,           # Độ sâu tối đa
            min_samples_split=5,    # Số mẫu tối thiểu để chia nhánh
            random_state=RANDOM_STATE,
            n_jobs=-1               # Dùng toàn bộ CPU
        ),

        # Mô hình 3: XGBoost - gradient boosting mạnh, thường thắng competition
        "XGBoost": XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,     # Tốc độ học (nhỏ = chậm nhưng chính xác hơn)
            max_depth=6,
            subsample=0.8,          # Lấy 80% dữ liệu mỗi lần boosting
            colsample_bytree=0.8,   # Lấy 80% cột mỗi lần xây cây
            random_state=RANDOM_STATE,
            verbosity=0
        ),
    }

    trained = {}
    for name, model in models.items():
        print(f"  Đang huấn luyện: {name}...", end=" ", flush=True)
        model.fit(X_train, y_train)
        trained[name] = model
        print("Xong!")

    return trained


# =============================================================================
# BƯỚC 4: ĐÁNH GIÁ MÔ HÌNH
# =============================================================================

def evaluate_models(trained_models: dict, X_test, y_test):
    """
    Đánh giá các mô hình trên tập test bằng MAE, RMSE, R².

    Args:
        trained_models: Dict các mô hình đã huấn luyện.
        X_test: Tập test đặc trưng.
        y_test: Nhãn thực tế của tập test.
    """
    print("\n" + "=" * 60)
    print("BƯỚC 4: ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST")
    print("=" * 60)
    print(f"  {'Mô hình':<22} {'MAE':>12} {'RMSE':>12} {'R²':>8}")
    print("  " + "-" * 58)

    results = {}
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}
        print(f"  {name:<22} ${mae:>10,.0f} ${rmse:>10,.0f} {r2:>7.4f}")

    # Xác định mô hình tốt nhất dựa trên R²
    best_name = max(results, key=lambda k: results[k]["R2"])
    print(f"\n  => Mô hình tốt nhất (R²): {best_name} (R² = {results[best_name]['R2']:.4f})")

    return results

# BƯỚC 5: LƯU MÔ HÌNH VÀ CÁC TIỀN XỬ LÝ
def save_artifacts(trained_models: dict, scaler, label_encoders: dict, feature_names: list):
    print("\n" + "=" * 60)
    print("BƯỚC 5: LƯU MÔ HÌNH VÀ TIỀN XỬ LÝ")
    print("=" * 60)

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Tên file tương ứng với từng mô hình
    model_filenames = {
        "Linear Regression": "linear_regression.pkl",
        "Random Forest":     "random_forest.pkl",
        "XGBoost":           "xgboost.pkl",
    }

    for name, model in trained_models.items():
        path = os.path.join(MODEL_DIR, model_filenames[name])
        joblib.dump(model, path)
        print(f"  Đã lưu: {path}")

    # Lưu scaler (dùng để chuẩn hóa dữ liệu nhập từ người dùng)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    print(f"  Đã lưu: {MODEL_DIR}/scaler.pkl")

    # Lưu label encoders (dict)
    joblib.dump(label_encoders, os.path.join(MODEL_DIR, "label_encoders.pkl"))
    print(f"  Đã lưu: {MODEL_DIR}/label_encoders.pkl")

    # Lưu danh sách feature names (đảm bảo đúng thứ tự khi dự báo)
    joblib.dump(feature_names, os.path.join(MODEL_DIR, "feature_names.pkl"))
    print(f"  Đã lưu: {MODEL_DIR}/feature_names.pkl")

    # Lưu danh sách các nhãn gốc của từng biến phân loại (dùng cho selectbox trong app)
    cat_options = {col: list(le.classes_) for col, le in label_encoders.items()}
    joblib.dump(cat_options, os.path.join(MODEL_DIR, "categorical_options.pkl"))
    print(f"  Đã lưu: {MODEL_DIR}/categorical_options.pkl")

    print("\n  Hoàn tất! Tất cả artifacts đã được lưu.")


# CHẠY TOÀN BỘ PIPELINE
if __name__ == "__main__":
    # Bước 1: Đọc dữ liệu
    df = load_and_explore(DATA_PATH)

    # Bước 2: Tiền xử lý
    X_train, X_test, y_train, y_test, scaler, label_encoders, feature_names = preprocess(df)

    # Bước 3: Huấn luyện
    trained_models = train_models(X_train, y_train)

    # Bước 4: Đánh giá
    evaluate_models(trained_models, X_test, y_test)

    # Bước 5: Lưu
    save_artifacts(trained_models, scaler, label_encoders, feature_names)
