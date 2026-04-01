# =============================================================================
# app.py
# Phần 2: Giao diện Web Dự báo Giá Nhà - Ames Housing Dataset
# Thư viện: Streamlit
# Tác giả: Nhóm sinh viên - Đồ án Môn Học
# Chạy: streamlit run app.py
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# =============================================================================
# CẤU HÌNH TRANG VÀ ĐƯỜNG DẪN
# =============================================================================

st.set_page_config(
    page_title="🏠 Dự Báo Giá Nhà - Ames Housing",
    page_icon="🏠",
    layout="wide",
)

MODEL_DIR = "models"

# Ánh xạ tên mô hình hiển thị -> tên file .pkl
MODEL_FILES = {
    "Linear Regression (Hồi quy tuyến tính)": "linear_regression.pkl",
    "Random Forest (Rừng ngẫu nhiên)":         "random_forest.pkl",
    "XGBoost (Gradient Boosting)":              "xgboost.pkl",
}

# Các cột đặc trưng (phải khớp với model_training.py)
NUMERIC_FEATURES = [
    "Gr Liv Area", "Total Bsmt SF", "1st Flr SF", "Garage Area",
    "Lot Area", "Year Built", "Year Remod/Add", "Overall Qual",
    "Overall Cond", "TotRms AbvGrd", "Full Bath", "Bedroom AbvGr",
    "Fireplaces", "Garage Cars",
]
CATEGORICAL_FEATURES = [
    "Neighborhood", "House Style", "Bldg Type", "Central Air",
    "Kitchen Qual", "Exter Qual",
]


# =============================================================================
# HÀM TẢI ARTIFACTS (CÓ CACHE ĐỂ TRÁNH TẢI LẠI MỖI LẦN TƯƠNG TÁC)
# =============================================================================

@st.cache_resource
def load_artifacts():
    """
    Tải scaler, label_encoders, feature_names và categorical_options từ thư mục models/.
    Dùng @st.cache_resource để chỉ tải 1 lần duy nhất khi app khởi động.
    
    Returns:
        Tuple: (scaler, label_encoders, feature_names, cat_options)
    Raises:
        FileNotFoundError: Nếu chưa chạy model_training.py.
    """
    required_files = ["scaler.pkl", "label_encoders.pkl", "feature_names.pkl", "categorical_options.pkl"]
    for f in required_files:
        if not os.path.exists(os.path.join(MODEL_DIR, f)):
            raise FileNotFoundError(
                f"Không tìm thấy '{f}'. Vui lòng chạy 'python model_training.py' trước!"
            )

    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    label_encoders = joblib.load(os.path.join(MODEL_DIR, "label_encoders.pkl"))
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
    cat_options = joblib.load(os.path.join(MODEL_DIR, "categorical_options.pkl"))

    return scaler, label_encoders, feature_names, cat_options


@st.cache_resource
def load_model(model_filename: str):
    """
    Tải mô hình ML từ file .pkl.
    Dùng @st.cache_resource để không load lại khi người dùng tương tác.

    Args:
        model_filename: Tên file pkl của mô hình.
    Returns:
        Mô hình sklearn/xgboost đã huấn luyện.
    """
    path = os.path.join(MODEL_DIR, model_filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Không tìm thấy mô hình '{path}'. Vui lòng chạy 'python model_training.py' trước!"
        )
    return joblib.load(path)


# =============================================================================
# HÀM TIỀN XỬ LÝ ĐẦU VÀO CỦA NGƯỜI DÙNG
# =============================================================================

def preprocess_input(user_input: dict, scaler, label_encoders, feature_names) -> pd.DataFrame:
    """
    Biến đổi dữ liệu nhập từ người dùng thành dạng phù hợp để đưa vào mô hình.
    Quy trình: Label Encode các biến phân loại -> StandardScale các biến số.

    Args:
        user_input: Dictionary {tên_cột: giá_trị}.
        scaler: StandardScaler đã fit từ bước training.
        label_encoders: Dict các LabelEncoder theo từng cột phân loại.
        feature_names: Danh sách tên cột theo đúng thứ tự.
    Returns:
        DataFrame 1 dòng đã được mã hóa và chuẩn hóa, sẵn sàng để dự báo.
    """
    # Tạo DataFrame 1 dòng từ dữ liệu người dùng
    df_input = pd.DataFrame([user_input])

    # Mã hóa các biến phân loại bằng LabelEncoder đã lưu
    for col in CATEGORICAL_FEATURES:
        le = label_encoders[col]
        val = str(df_input[col].iloc[0])
        # Xử lý trường hợp giá trị mới không có trong tập huấn luyện
        if val not in le.classes_:
            val = le.classes_[0]  # Dùng giá trị đầu tiên làm fallback
        df_input[col] = le.transform([val])

    # Đảm bảo đúng thứ tự cột
    df_input = df_input[feature_names]

    # Chuẩn hóa các biến số
    df_input[NUMERIC_FEATURES] = scaler.transform(df_input[NUMERIC_FEATURES])

    return df_input


# =============================================================================
# GIAO DIỆN CHÍNH
# =============================================================================

def main():
    # --- Tiêu đề trang ---
    st.title("🏠 Hệ Thống Dự Báo Giá Nhà")
    st.markdown("**Dataset: Ames Housing** | Môn học: Machine Learning | Nhóm sinh viên")
    st.markdown("---")

    # --- Tải artifacts (scaler, encoder, v.v.) ---
    try:
        scaler, label_encoders, feature_names, cat_options = load_artifacts()
    except FileNotFoundError as e:
        st.error(f"⚠️ {e}")
        st.info("👉 Mở terminal và chạy: `python model_training.py`")
        st.stop()

    # =========================================================================
    # SIDEBAR: Chọn mô hình và nhập thông số nhà
    # =========================================================================
    with st.sidebar:
        st.header("⚙️ Cài đặt & Nhập liệu")
        st.markdown("---")

        # --- Chọn mô hình ---
        st.subheader("1️⃣ Chọn mô hình dự báo")
        selected_model_name = st.selectbox(
            "Mô hình:",
            options=list(MODEL_FILES.keys()),
            help="Chọn thuật toán Machine Learning để dự báo giá nhà."
        )

        st.markdown("---")

        # --- Nhập thông số nhà ---
        st.subheader("2️⃣ Thông số ngôi nhà")

        # == Phần Diện tích & Kết cấu ==
        st.markdown("**📐 Diện tích & Kết cấu**")
        gr_liv_area = st.number_input(
            "Diện tích sống (sq ft)", min_value=300, max_value=6000,
            value=1500, step=50,
            help="Tổng diện tích sống trên mặt đất (không tính tầng hầm)"
        )
        total_bsmt_sf = st.number_input(
            "Diện tích tầng hầm (sq ft)", min_value=0, max_value=3000,
            value=800, step=50
        )
        first_flr_sf = st.number_input(
            "Diện tích tầng 1 (sq ft)", min_value=300, max_value=4000,
            value=1000, step=50
        )
        lot_area = st.number_input(
            "Diện tích lô đất (sq ft)", min_value=1000, max_value=100000,
            value=10000, step=500
        )
        garage_area = st.number_input(
            "Diện tích Garage (sq ft)", min_value=0, max_value=1500,
            value=400, step=50
        )

        st.markdown("**🛏️ Phòng & Tiện nghi**")
        bedroom = st.slider("Số phòng ngủ", min_value=0, max_value=8, value=3)
        full_bath = st.slider("Số phòng tắm đầy đủ", min_value=0, max_value=4, value=2)
        tot_rms = st.slider("Tổng số phòng (không tính phòng tắm)", min_value=2, max_value=14, value=7)
        fireplaces = st.slider("Số lò sưởi", min_value=0, max_value=4, value=1)
        garage_cars = st.slider("Sức chứa Garage (số xe)", min_value=0, max_value=4, value=2)

        st.markdown("**📅 Thời gian**")
        year_built = st.number_input("Năm xây dựng", min_value=1872, max_value=2010, value=1980, step=1)
        year_remod = st.number_input("Năm cải tạo gần nhất", min_value=1950, max_value=2010, value=2000, step=1)

        st.markdown("**⭐ Đánh giá chất lượng**")
        overall_qual = st.slider(
            "Chất lượng tổng thể (1=Rất kém, 10=Xuất sắc)",
            min_value=1, max_value=10, value=6
        )
        overall_cond = st.slider(
            "Tình trạng tổng thể (1=Rất kém, 10=Xuất sắc)",
            min_value=1, max_value=10, value=5
        )

        st.markdown("**📍 Vị trí & Loại nhà**")
        neighborhood = st.selectbox(
            "Khu vực (Neighborhood)",
            options=cat_options.get("Neighborhood", ["NAmes"]),
        )
        house_style = st.selectbox(
            "Kiểu nhà (House Style)",
            options=cat_options.get("House Style", ["1Story"]),
        )
        bldg_type = st.selectbox(
            "Loại tòa nhà (Building Type)",
            options=cat_options.get("Bldg Type", ["1Fam"]),
        )
        central_air = st.selectbox(
            "Điều hòa trung tâm",
            options=cat_options.get("Central Air", ["Y", "N"]),
        )
        kitchen_qual = st.selectbox(
            "Chất lượng bếp (Ex=Xuất sắc, Gd=Tốt, TA=Trung bình, Fa=Kém)",
            options=cat_options.get("Kitchen Qual", ["TA"]),
        )
        exter_qual = st.selectbox(
            "Chất lượng ngoại thất",
            options=cat_options.get("Exter Qual", ["TA"]),
        )

        st.markdown("---")
        # Nút dự báo
        predict_button = st.button("🔮 Dự Báo Giá Nhà", type="primary", use_container_width=True)

    # =========================================================================
    # KHU VỰC NỘI DUNG CHÍNH: Hiển thị kết quả
    # =========================================================================

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📊 Kết quả dự báo")

        if not predict_button:
            st.info("👈 Nhập thông số ngôi nhà ở thanh bên trái và nhấn **Dự Báo Giá Nhà**.")
        else:
            # --- Tải mô hình được chọn ---
            try:
                model = load_model(MODEL_FILES[selected_model_name])
            except FileNotFoundError as e:
                st.error(f"⚠️ {e}")
                st.stop()

            # --- Tổng hợp dữ liệu nhập của người dùng ---
            user_input = {
                "Gr Liv Area":    gr_liv_area,
                "Total Bsmt SF":  total_bsmt_sf,
                "1st Flr SF":     first_flr_sf,
                "Garage Area":    garage_area,
                "Lot Area":       lot_area,
                "Year Built":     year_built,
                "Year Remod/Add": year_remod,
                "Overall Qual":   overall_qual,
                "Overall Cond":   overall_cond,
                "TotRms AbvGrd":  tot_rms,
                "Full Bath":      full_bath,
                "Bedroom AbvGr":  bedroom,
                "Fireplaces":     fireplaces,
                "Garage Cars":    garage_cars,
                "Neighborhood":   neighborhood,
                "House Style":    house_style,
                "Bldg Type":      bldg_type,
                "Central Air":    central_air,
                "Kitchen Qual":   kitchen_qual,
                "Exter Qual":     exter_qual,
            }

            # --- Kiểm tra dữ liệu hợp lệ ---
            if year_remod < year_built:
                st.warning("⚠️ Năm cải tạo không thể nhỏ hơn năm xây dựng! Vui lòng kiểm tra lại.")
                st.stop()

            # --- Tiền xử lý và dự báo ---
            try:
                X_pred = preprocess_input(user_input, scaler, label_encoders, feature_names)
                predicted_price = model.predict(X_pred)[0]
                predicted_price = max(0, predicted_price)  # Giá không thể âm
            except Exception as e:
                st.error(f"❌ Lỗi khi dự báo: {e}")
                st.stop()

            # --- Hiển thị kết quả ---
            st.success("✅ Dự báo thành công!")

            # Giá bằng USD
            st.metric(
                label=f"💰 Giá nhà dự báo ({selected_model_name.split('(')[0].strip()})",
                value=f"${predicted_price:,.0f} USD",
            )

            # Khoảng dự báo ước tính (+/- 10% để tham khảo)
            lower = predicted_price * 0.90
            upper = predicted_price * 1.10
            st.markdown(
                f"📏 **Khoảng ước tính (±10%):** `${lower:,.0f}` — `${upper:,.0f}` USD"
            )

            st.markdown("---")

            # Bảng tóm tắt thông số đã nhập
            st.markdown("**📋 Thông số đã nhập:**")
            summary_data = {
                "Thông số": [
                    "Diện tích sống", "Diện tích tầng hầm", "Diện tích tầng 1",
                    "Diện tích lô đất", "Diện tích Garage", "Số phòng ngủ",
                    "Số phòng tắm", "Chất lượng tổng thể", "Năm xây dựng",
                    "Khu vực", "Kiểu nhà", "Điều hòa trung tâm"
                ],
                "Giá trị": [
                    f"{gr_liv_area:,} sq ft", f"{total_bsmt_sf:,} sq ft", f"{first_flr_sf:,} sq ft",
                    f"{lot_area:,} sq ft", f"{garage_area:,} sq ft", f"{bedroom} phòng",
                    f"{full_bath} phòng", f"{overall_qual}/10", str(year_built),
                    neighborhood, house_style, central_air
                ]
            }
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

    with col2:
        st.subheader("ℹ️ Về các mô hình")

        model_info = {
            "Linear Regression (Hồi quy tuyến tính)": {
                "icon": "📈",
                "desc": "Mô hình cơ bản nhất. Giả định quan hệ tuyến tính giữa đặc trưng và giá. Nhanh, dễ giải thích nhưng độ chính xác thấp hơn.",
                "pros": "Nhanh, minh bạch",
                "cons": "Không nắm bắt phi tuyến"
            },
            "Random Forest (Rừng ngẫu nhiên)": {
                "icon": "🌲",
                "desc": "Kết hợp nhiều cây quyết định (ensemble). Xử lý tốt dữ liệu phi tuyến và ít bị overfitting.",
                "pros": "Chính xác, ổn định",
                "cons": "Chậm hơn, khó giải thích"
            },
            "XGBoost (Gradient Boosting)": {
                "icon": "⚡",
                "desc": "Thuật toán boosting mạnh nhất hiện nay. Xây cây tuần tự, mỗi cây sửa lỗi cây trước. Thường đạt kết quả tốt nhất.",
                "pros": "Rất chính xác",
                "cons": "Cần chỉnh nhiều tham số"
            },
        }

        for name, info in model_info.items():
            is_selected = name == selected_model_name
            border_color = "#4CAF50" if is_selected else "#e0e0e0"
            with st.container(border=True):
                st.markdown(f"**{info['icon']} {name.split('(')[0].strip()}**" + (" ✅" if is_selected else ""))
                st.caption(info["desc"])
                st.markdown(f"✔️ *{info['pros']}* &nbsp;&nbsp; ✖️ *{info['cons']}*")


# =============================================================================
# ĐIỂM CHẠY CHƯƠNG TRÌNH
# =============================================================================

if __name__ == "__main__":
    main()
