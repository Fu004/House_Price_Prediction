import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os


# CẤU HÌNH TRANG VÀ ĐƯỜNG DẪN
st.set_page_config(
    page_title="House Price Prediction - Ames Housing",
    page_icon="",
    layout="wide",
)

MODEL_DIR = "models"

MODEL_FILES = {
    "Linear Regression (Linear Regression)": "linear_regression.pkl",
    "Random Forest (Random Forest)":         "random_forest.pkl",
    "XGBoost (Gradient Boosting)":              "xgboost.pkl",
}

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

HOUSE_STYLE_DISPLAY = {
    "1Story": "1 Story",
    "1.5Fin": "1.5 Story Finished",
    "1.5Unf": "1.5 Story Unfinished",
    "2Story": "2 Story",
    "2.5Fin": "2.5 Story Finished",
    "2.5Unf": "2.5 Story Unfinished",
    "SFoyer": "Split Foyer",
    "SLvl": "Split Level",
}

BLDG_TYPE_DISPLAY = {
    "1Fam": "Single-family Detached",
    "2fmCon": "Two-family Conversion",
    "Duplx": "Duplex",
    "TwnhsE": "Townhouse End Unit",
    "Twnhs": "Townhouse Inside Unit",
}

CENTRAL_AIR_DISPLAY = {
    "Y": "Yes",
    "N": "No",
}

QUALITY_DISPLAY = {
    "Ex": "Excellent",
    "Gd": "Good",
    "TA": "Average/Typical",
    "Fa": "Fair",
    "Po": "Poor",
}

# HÀM TẢI ARTIFACTS 
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
                f"File '{f}' not found. Please run 'python model_training.py' first!"
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
            f"Model '{path}' not found. Please run 'python model_training.py' first!"
        )
    return joblib.load(path)

def get_display_value(code: str, display_map: dict) -> str:
    return display_map.get(code, code)


def selectbox_with_display(label: str, values: list, display_map: dict, default: str = None):
    options = [display_map.get(value, value) for value in values]
    if default is not None and default in values:
        default_index = values.index(default)
    else:
        default_index = 0
    selected_label = st.selectbox(label, options=options, index=default_index)
    selected_code = values[options.index(selected_label)]
    return selected_code, selected_label

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
    df_input = pd.DataFrame([user_input])


    for col in CATEGORICAL_FEATURES:
        le = label_encoders[col]
        val = str(df_input[col].iloc[0])

        if val not in le.classes_:
            val = le.classes_[0]  # Use first class as fallback
        df_input[col] = le.transform([val])


    df_input = df_input[feature_names]

    df_input[NUMERIC_FEATURES] = scaler.transform(df_input[NUMERIC_FEATURES])

    return df_input



# GIAO DIỆN CHÍNH
def main():

    st.title("House Price Prediction System")
    st.markdown("**Dataset: Ames Housing** | Course: Machine Learning | Student Group")
    st.markdown("---")


    try:
        scaler, label_encoders, feature_names, cat_options = load_artifacts()
    except FileNotFoundError as e:
        st.error(f"{e}")
        st.info("Open terminal and run: `python model_training.py`")
        st.stop()

    with st.sidebar:
        st.header("Settings & Input")
        st.markdown("---")

        st.subheader("1️⃣ Choose Prediction Model")
        selected_model_name = st.selectbox(
            "Model:",
            options=list(MODEL_FILES.keys()),
            help="Select the Machine Learning algorithm to predict house prices."
        )

        st.markdown("---")

        st.subheader("2️⃣ House Specifications")

        st.markdown("**Area & Structure**")
        gr_liv_area = st.number_input(
            "Living Area (sq ft)", min_value=300, max_value=6000,
            value=1500, step=50,
            help="Total above grade (ground) living area square feet"
        )
        total_bsmt_sf = st.number_input(
            "Basement Area (sq ft)", min_value=0, max_value=3000,
            value=800, step=50
        )
        first_flr_sf = st.number_input(
            "First Floor Area (sq ft)", min_value=300, max_value=4000,
            value=1000, step=50
        )
        lot_area = st.number_input(
            "Lot Area (sq ft)", min_value=1000, max_value=100000,
            value=10000, step=500
        )
        garage_area = st.number_input(
            "Garage Area (sq ft)", min_value=0, max_value=1500,
            value=400, step=50
        )

        st.markdown("**🛏️ Rooms & Amenities**")
        bedroom = st.slider("Bedrooms", min_value=0, max_value=8, value=3)
        full_bath = st.slider("Full Bathrooms", min_value=0, max_value=4, value=2)
        tot_rms = st.slider("Total Rooms (excluding bathrooms)", min_value=2, max_value=14, value=7)
        fireplaces = st.slider("Fireplaces", min_value=0, max_value=4, value=1)
        garage_cars = st.slider("Garage Capacity (cars)", min_value=0, max_value=4, value=2)

        st.markdown("**Time**")
        year_built = st.number_input("Year Built", min_value=1872, max_value=2010, value=1980, step=1)
        year_remod = st.number_input("Year Remodeled", min_value=1950, max_value=2010, value=2000, step=1)

        st.markdown("**Quality Ratings**")
        overall_qual = st.slider(
            "Overall Quality (1=Very Poor, 10=Very Excellent)",
            min_value=1, max_value=10, value=6
        )
        overall_cond = st.slider(
            "Overall Condition (1=Very Poor, 10=Very Excellent)",
            min_value=1, max_value=10, value=5
        )

        st.markdown("**Location & House Type**")
        neighborhood = st.selectbox(
            "Neighborhood",
            options=cat_options.get("Neighborhood", ["NAmes"]),
        )
        house_style, house_style_label = selectbox_with_display(
            "House Style",
            cat_options.get("House Style", ["1Story"]),
            HOUSE_STYLE_DISPLAY,
            default="1Story"
        )
        bldg_type, bldg_type_label = selectbox_with_display(
            "Building Type",
            cat_options.get("Bldg Type", ["1Fam"]),
            BLDG_TYPE_DISPLAY,
            default="1Fam"
        )
        central_air, central_air_label = selectbox_with_display(
            "Central Air Conditioning",
            cat_options.get("Central Air", ["Y", "N"]),
            CENTRAL_AIR_DISPLAY,
            default="Y"
        )
        kitchen_qual, kitchen_qual_label = selectbox_with_display(
            "Kitchen Quality",
            cat_options.get("Kitchen Qual", ["TA"]),
            QUALITY_DISPLAY,
            default="TA"
        )
        exter_qual, exter_qual_label = selectbox_with_display(
            "Exterior Quality",
            cat_options.get("Exter Qual", ["TA"]),
            QUALITY_DISPLAY,
            default="TA"
        )

        st.markdown("---")
        # Prediction button
        predict_button = st.button("Predict House Price", type="primary", use_container_width=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Prediction Results")

        if not predict_button:
            st.info("Enter the house specifications in the left sidebar and click **Predict House Price**.")
        else:
            try:
                model = load_model(MODEL_FILES[selected_model_name])
            except FileNotFoundError as e:
                st.error(f"{e}")
                st.stop()
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

            if year_remod < year_built:
                st.warning("Remodel year cannot be earlier than build year! Please check again.")
                st.stop()

            try:
                X_pred = preprocess_input(user_input, scaler, label_encoders, feature_names)
                predicted_price = model.predict(X_pred)[0]
                predicted_price = max(0, predicted_price)  # Price cannot be negative
            except Exception as e:
                st.error(f"Error during prediction: {e}")
                st.stop()

            st.success("Prediction successful!")


            st.metric(
                label=f"Predicted House Price ({selected_model_name.split('(')[0].strip()})",
                value=f"${predicted_price:,.0f} USD",
            )

            lower = predicted_price * 0.90
            upper = predicted_price * 1.10
            st.markdown(
                f"**Estimated Range (±10%):** `${lower:,.0f}` — `${upper:,.0f}` USD"
            )

            st.markdown("---")

            # Summary table of entered specifications
            st.markdown("**Entered Specifications:**")
            summary_data = {
                "Specification": [
                    "Living Area", "Basement Area", "First Floor Area",
                    "Lot Area", "Garage Area", "Bedrooms",
                    "Full Bathrooms", "Overall Quality", "Year Built",
                    "Neighborhood", "House Style", "Building Type", "Central Air",
                    "Kitchen Quality", "Exterior Quality"
                ],
                "Value": [
                    f"{gr_liv_area:,} sq ft", f"{total_bsmt_sf:,} sq ft", f"{first_flr_sf:,} sq ft",
                    f"{lot_area:,} sq ft", f"{garage_area:,} sq ft", f"{bedroom} rooms",
                    f"{full_bath} rooms", f"{overall_qual}/10", str(year_built),
                    neighborhood, house_style_label, bldg_type_label, central_air_label,
                    kitchen_qual_label, exter_qual_label
                ]
            }
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
