import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px

# ──────────────────────────────────────────────
# Giao diện thử nghiệm
# ──────────────────────────────────────────────

# ──────────────────────────────────────────────
# CẤU HÌNH TRANG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="🏠 House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CUSTOM CSS – Colorful & Friendly Theme
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&family=Poppins:wght@400;500;600;700&display=swap');

/* ── Root & body ── */
html, body, [class*="css"] {
    font-family: 'Nunito', sans-serif;
    color: #1f2d3d !important;
}

/* ── Background ── */
.stApp {
    background: #0a0a0a;
    color: #e6e6e6;
}

/* ── Hide Streamlit branding ── */
#MainMenu, footer { visibility: hidden; }

/* ── Page title banner ── */
.hero-banner {
    background: linear-gradient(135deg, #111827, #1f2937, #374151);
    background-size: 300% 300%;
    animation: gradientShift 8s ease infinite;
    border-radius: 24px;
    padding: 32px 40px;
    margin-bottom: 28px;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.45);
}
@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.hero-banner h1 {
    color: #ffffff;
    font-size: 2.4rem;
    font-weight: 900;
    text-shadow: 0 2px 12px rgba(0,0,0,0.22);
    margin: 0 0 8px 0;
}
.hero-banner p {
    color: rgba(255,255,255,0.95);
    font-size: 1.05rem;
    font-weight: 600;
    margin: 0;
}

/* ── Section cards ── */
.card {
    background: #111827;
    border-radius: 20px;
    padding: 28px 32px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.55);
    margin-bottom: 20px;
    border: 2px solid transparent;
    transition: border-color 0.3s;
}
.card:hover { border-color: #4f76e0; }

/* ── Result price card ── */
.price-card {
    background: linear-gradient(135deg, #2b3b59, #3f5481);
    border-radius: 24px;
    padding: 32px;
    text-align: center;
    color: #ffffff;
    box-shadow: 0 10px 30px rgba(38, 62, 101, 0.18);
    margin-bottom: 20px;
}
.price-card .label {
    font-size: 0.95rem;
    font-weight: 700;
    opacity: 0.92;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.price-card .price {
    font-size: 3rem;
    font-weight: 900;
    letter-spacing: -1px;
    line-height: 1.1;
}
.price-card .range {
    font-size: 0.95rem;
    opacity: 0.88;
    margin-top: 10px;
    font-weight: 600;
}

/* ── Range badges ── */
.range-badge {
    display: inline-block;
    background: rgba(255,255,255,0.2);
    border-radius: 30px;
    padding: 4px 16px;
    margin: 4px;
    font-weight: 700;
}

/* ── Stat pills ── */
.stat-row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin: 16px 0;
}
.stat-pill {
    background: #e8eef6;
    border-radius: 50px;
    padding: 8px 20px;
    font-weight: 700;
    font-size: 0.88rem;
    color: #1f2d3d;
    display: flex;
    align-items: center;
    gap: 6px;
}
.stat-pill.blue  { background: #4f76e0; color: #ffffff; }
.stat-pill.green { background: #2f9d70; color: #ffffff; }
.stat-pill.pink  { background: #bd5f7e; color: #ffffff; }
.stat-pill.purple{ background: #6f5fce; color: #ffffff; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617 0%, #111827 100%) !important;
}
[data-testid="stSidebar"] * {
    color: #cad6ea !important;
}
[data-testid="stSidebar"] .stText,
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] .stMarkdown p {
    color: #c1d0e6 !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #c5d7f4 !important;
}
[data-testid="stSidebar"] .sidebar-section-title {
    color: #b0c4e0 !important;
    font-weight: 800 !important;
    font-size: 0.82rem !important;
    letter-spacing: 1.8px !important;
    text-transform: uppercase !important;
}

/* ── Predict button ── */
[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    color: #ffffff !important;
    font-weight: 800 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 16px !important;
    padding: 14px !important;
    box-shadow: 0 4px 18px rgba(37, 99, 235, 0.42) !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(20, 43, 93, 0.44) !important;
}

/* ── Info / warning / success boxes ── */
.stAlert { border-radius: 16px !important; }

/* ── Dataframe ── */
.stDataFrame { border-radius: 16px; overflow: hidden; }

/* ── Section headers inside main area ── */
.section-header {
    font-size: 1.15rem;
    font-weight: 800;
    color: #1f2d3d;
    margin: 20px 0 12px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────
MODEL_DIR = "models"

MODEL_FILES = {
    "🔵 Linear Regression":   "linear_regression.pkl",
    "🌲 Random Forest":        "random_forest.pkl",
    "⚡ XGBoost":              "xgboost.pkl",
}

MODEL_DESCRIPTIONS = {
    "🔵 Linear Regression":  "Đơn giản, nhanh, dễ giải thích.",
    "🌲 Random Forest":       "Ensemble nhiều cây, ổn định, chính xác cao.",
    "⚡ XGBoost":             "Gradient boosting mạnh, thường đạt R² tốt nhất.",
}

NEIGHBORHOOD_MAP = {
    "Blmngtn":"Bloomington Heights","Blueste":"Bluestem","BrDale":"Briardale",
    "BrkSide":"Brookside","ClearCr":"Clear Creek","CollgCr":"College Creek",
    "Crawfor":"Crawford","Edwards":"Edwards","Gilbert":"Gilbert",
    "IDOTRR":"Iowa DOT and Rail Road","Greens":"Greenshire","GrnHill":"Green Hills",
    "Landmrk":"Landmark","MeadowV":"Meadow Village","Mitchel":"Mitchell",
    "Names":"North Ames","NAmes":"North Ames","NoRidge":"Northridge",
    "NPkVill":"Northpark Villa","NridgHt":"Northridge Heights","NWAmes":"Northwest Ames",
    "OldTown":"Old Town","SWISU":"South & West of ISU","Sawyer":"Sawyer",
    "SawyerW":"Sawyer West","Somerst":"Somerset","StoneBr":"Stone Brook",
    "Timber":"Timberland","Veenker":"Veenker",
}

NUMERIC_FEATURES = [
    "Gr Liv Area","Total Bsmt SF","1st Flr SF","Garage Area","Lot Area",
    "Year Built","Year Remod/Add","Overall Qual","Overall Cond",
    "TotRms AbvGrd","Full Bath","Bedroom AbvGr","Fireplaces","Garage Cars",
]
CATEGORICAL_FEATURES = [
    "Neighborhood","House Style","Bldg Type","Central Air","Kitchen Qual","Exter Qual",
]

HOUSE_STYLE_DISPLAY = {
    "1Story":"1 Story","1.5Fin":"1.5 Story Finished","1.5Unf":"1.5 Story Unfinished",
    "2Story":"2 Story","2.5Fin":"2.5 Story Finished","2.5Unf":"2.5 Story Unfinished",
    "SFoyer":"Split Foyer","SLvl":"Split Level",
}
BLDG_TYPE_DISPLAY = {
    "1Fam":"Single-family Detached","2fmCon":"Two-family Conversion",
    "Duplex":"Duplex","TwnhsE":"Townhouse End Unit","Twnhs":"Townhouse Inside Unit",
}
CENTRAL_AIR_DISPLAY = {"Y":"Yes ✅","N":"No ❌"}
QUALITY_DISPLAY = {"Ex":"Excellent ⭐⭐⭐⭐⭐","Gd":"Good ⭐⭐⭐⭐","TA":"Average ⭐⭐⭐","Fa":"Fair ⭐⭐","Po":"Poor ⭐"}

# ──────────────────────────────────────────────
# LOADERS
# ──────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    required = ["scaler.pkl","label_encoders.pkl","feature_names.pkl","categorical_options.pkl"]
    for f in required:
        if not os.path.exists(os.path.join(MODEL_DIR, f)):
            raise FileNotFoundError(f"File '{f}' not found. Run 'python model_training.py' first!")
    scaler        = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    label_encoders= joblib.load(os.path.join(MODEL_DIR, "label_encoders.pkl"))
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
    cat_options   = joblib.load(os.path.join(MODEL_DIR, "categorical_options.pkl"))
    return scaler, label_encoders, feature_names, cat_options

@st.cache_resource
def load_model(filename):
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model '{path}' not found. Run 'python model_training.py' first!")
    return joblib.load(path)

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def selectbox_with_display(label, values, display_map, default=None):
    options = [display_map.get(v, v) for v in values]
    idx = values.index(default) if default and default in values else 0
    selected_label = st.selectbox(label, options=options, index=idx)
    selected_code  = values[options.index(selected_label)]
    return selected_code, selected_label

def preprocess_input(user_input, scaler, label_encoders, feature_names):
    df = pd.DataFrame([user_input])
    for col in CATEGORICAL_FEATURES:
        le  = label_encoders[col]
        val = str(df[col].iloc[0])
        if val not in le.classes_:
            val = le.classes_[0]
        df[col] = le.transform([val])
    df = df[feature_names]
    df[NUMERIC_FEATURES] = scaler.transform(df[NUMERIC_FEATURES])
    return df

def gauge_chart(value, min_val=50000, max_val=750000):
    """Plotly gauge showing predicted price."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"prefix":"$","valueformat":",.0f","font":{"size":28,"color":"#2d3436","family":"Nunito"}},
        gauge={
            "axis":{"range":[min_val, max_val],"tickformat":"$,.0f",
                    "tickfont":{"size":10},"nticks":6},
            "bar":{"color":"#6c5ce7","thickness":0.25},
            "bgcolor":"white",
            "borderwidth":2,
            "bordercolor":"#dfe6e9",
            "steps":[
                {"range":[min_val, 200000],"color":"#55efc4"},
                {"range":[200000, 350000],"color":"#ffeaa7"},
                {"range":[350000, 500000],"color":"#fdcb6e"},
                {"range":[500000, max_val],"color":"#ff7675"},
            ],
            "threshold":{"line":{"color":"#6c5ce7","width":4},"thickness":0.75,"value":value},
        },
        domain={"x":[0,1],"y":[0,1]},
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=30,r=30,t=20,b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family":"Nunito"},
    )
    return fig

def feature_importance_chart(model, feature_names, model_name):
    """Bar chart of feature importances (RF / XGBoost only)."""
    if not hasattr(model, "feature_importances_"):
        return None
    importance = model.feature_importances_
    df_imp = pd.DataFrame({"Feature": feature_names, "Importance": importance})
    df_imp = df_imp.sort_values("Importance", ascending=True).tail(10)

    colors = px.colors.sequential.Sunset[::-1][:len(df_imp)]

    fig = go.Figure(go.Bar(
        x=df_imp["Importance"],
        y=df_imp["Feature"],
        orientation="h",
        marker=dict(color=colors, line=dict(color="rgba(0,0,0,0.1)", width=1)),
        text=[f"{v:.3f}" for v in df_imp["Importance"]],
        textposition="outside",
        textfont=dict(size=10, family="Nunito", color="#636e72"),
    ))
    fig.update_layout(
        title=dict(text="🔍 Top 10 Feature Importances", font=dict(size=14, family="Nunito", color="#2d3436")),
        xaxis=dict(showgrid=True, gridcolor="#f0f0f0", title=""),
        yaxis=dict(showgrid=False, title=""),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=340,
        margin=dict(l=10,r=60,t=40,b=10),
        font={"family":"Nunito"},
    )
    return fig

def price_range_chart(predicted, lower, upper):
    """Visual bar showing confidence range."""
    fig = go.Figure()
    # background track
    fig.add_trace(go.Bar(
        x=[upper - lower], base=[lower],
        y=["Price Range"], orientation="h",
        marker_color="rgba(108,92,231,0.15)",
        width=0.4, showlegend=False,
        hovertemplate=f"Range: ${lower:,.0f} – ${upper:,.0f}<extra></extra>",
    ))
    # point estimate
    fig.add_trace(go.Scatter(
        x=[predicted], y=["Price Range"],
        mode="markers+text",
        marker=dict(size=18, color="#6c5ce7", symbol="diamond",
                    line=dict(color="white", width=2)),
        text=[f"  ${predicted:,.0f}"],
        textfont=dict(size=13, color="#6c5ce7", family="Nunito"),
        textposition="middle right",
        showlegend=False,
        hovertemplate=f"Predicted: ${predicted:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        xaxis=dict(tickformat="$,.0f", showgrid=True, gridcolor="#f5f5f5"),
        yaxis=dict(showticklabels=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=110,
        margin=dict(l=10,r=10,t=10,b=10),
        font={"family":"Nunito"},
    )
    return fig

def specs_radar(user_input):
    """Mini radar chart of quality/room specs (normalized 0-1)."""
    cats = ["Overall\nQuality","Overall\nCondition","Full\nBath",
            "Bedrooms","Fireplaces","Garage\nCars"]
    raw  = [
        user_input["Overall Qual"] / 10,
        user_input["Overall Cond"] / 10,
        user_input["Full Bath"] / 4,
        user_input["Bedroom AbvGr"] / 8,
        user_input["Fireplaces"] / 4,
        user_input["Garage Cars"] / 4,
    ]
    fig = go.Figure(go.Scatterpolar(
        r=raw + [raw[0]],
        theta=cats + [cats[0]],
        fill="toself",
        fillcolor="rgba(162,155,254,0.3)",
        line=dict(color="#6c5ce7", width=2),
        marker=dict(size=7, color="#6c5ce7"),
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0,1], showticklabels=False, gridcolor="#e0e0e0"),
            angularaxis=dict(tickfont=dict(size=10, family="Nunito", color="#636e72")),
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        height=280,
        margin=dict(l=40,r=40,t=20,b=20),
    )
    return fig

# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    # Hero banner
    st.markdown("""
    <div class="hero-banner">
        <h1>🏠 House Price Predictor</h1>
        <p>Ames Housing Dataset · Machine Learning · 3 Models · Instant Estimate</p>
    </div>
    """, unsafe_allow_html=True)

    # Load artifacts
    try:
        scaler, label_encoders, feature_names, cat_options = load_artifacts()
    except FileNotFoundError as e:
        st.error(f"⚠️ {e}")
        st.info("Mở terminal và chạy: `python model_training.py`")
        st.stop()

    # ── SIDEBAR ──────────────────────────────────
    with st.sidebar:
        st.markdown("## 🏠 House Price Predictor")
        st.markdown("---")

        # Model selection with description
        st.markdown("### 🤖 Chọn Model")
        selected_model_name = st.selectbox(
            "Model dự báo:",
            options=list(MODEL_FILES.keys()),
        )
        st.caption(f"ℹ️ {MODEL_DESCRIPTIONS[selected_model_name]}")
        st.markdown("---")

        # ── Area & Structure
        st.markdown('<p class="sidebar-section-title">📐 Diện tích & Cấu trúc</p>', unsafe_allow_html=True)
        gr_liv_area   = st.number_input("Living Area (sq ft)",   300,  6000, 1500, 50)
        total_bsmt_sf = st.number_input("Basement Area (sq ft)",   0,  3000,  800, 50)
        first_flr_sf  = st.number_input("First Floor Area (sq ft)",300, 4000, 1000, 50)
        lot_area      = st.number_input("Lot Area (sq ft)",      1000,100000,10000,500)
        garage_area   = st.number_input("Garage Area (sq ft)",     0,  1500,  400, 50)

        st.markdown('<p class="sidebar-section-title">🛏️ Phòng & Tiện nghi</p>', unsafe_allow_html=True)
        bedroom    = st.slider("Bedrooms 🛏️",        0, 8, 3)
        full_bath  = st.slider("Full Bathrooms 🛁",   0, 4, 2)
        tot_rms    = st.slider("Total Rooms 🚪",       2,14, 7)
        fireplaces = st.slider("Fireplaces 🔥",        0, 4, 1)
        garage_cars= st.slider("Garage Capacity 🚗",   0, 4, 2)

        st.markdown('<p class="sidebar-section-title">📅 Thời gian</p>', unsafe_allow_html=True)
        year_built = st.number_input("Year Built 🏗️",     1872, 2010, 1980, 1)
        year_remod = st.number_input("Year Remodeled 🔨", 1950, 2010, 2000, 1)

        st.markdown('<p class="sidebar-section-title">⭐ Chất lượng</p>', unsafe_allow_html=True)
        overall_qual = st.slider("Overall Quality (1–10) 🏅", 1, 10, 6)
        overall_cond = st.slider("Overall Condition (1–10) 🔧",1, 10, 5)

        st.markdown('<p class="sidebar-section-title">📍 Vị trí & Loại nhà</p>', unsafe_allow_html=True)
        neighborhood = st.selectbox("Neighborhood 🗺️", options=cat_options.get("Neighborhood",["NAmes"]))
        house_style, house_style_label = selectbox_with_display(
            "House Style 🏡", cat_options.get("House Style",["1Story"]), HOUSE_STYLE_DISPLAY, "1Story")
        bldg_type, bldg_type_label = selectbox_with_display(
            "Building Type 🏢", cat_options.get("Bldg Type",["1Fam"]), BLDG_TYPE_DISPLAY, "1Fam")
        central_air, central_air_label = selectbox_with_display(
            "Central Air ❄️", cat_options.get("Central Air",["Y","N"]), CENTRAL_AIR_DISPLAY, "Y")
        kitchen_qual, kitchen_qual_label = selectbox_with_display(
            "Kitchen Quality 🍳", cat_options.get("Kitchen Qual",["TA"]), QUALITY_DISPLAY, "TA")
        exter_qual, exter_qual_label = selectbox_with_display(
            "Exterior Quality 🎨", cat_options.get("Exter Qual",["TA"]), QUALITY_DISPLAY, "TA")

        st.markdown("---")
        predict_button = st.button("🔮 Predict House Price", type="primary", use_container_width=True)

    # ── MAIN AREA ─────────────────────────────────
    user_input = {
        "Gr Liv Area": gr_liv_area, "Total Bsmt SF": total_bsmt_sf,
        "1st Flr SF": first_flr_sf, "Garage Area": garage_area,
        "Lot Area": lot_area, "Year Built": year_built,
        "Year Remod/Add": year_remod, "Overall Qual": overall_qual,
        "Overall Cond": overall_cond, "TotRms AbvGrd": tot_rms,
        "Full Bath": full_bath, "Bedroom AbvGr": bedroom,
        "Fireplaces": fireplaces, "Garage Cars": garage_cars,
        "Neighborhood": neighborhood, "House Style": house_style,
        "Bldg Type": bldg_type, "Central Air": central_air,
        "Kitchen Qual": kitchen_qual, "Exter Qual": exter_qual,
    }

    if not predict_button:
        # ── Placeholder state ──
        col_a, col_b = st.columns([3, 2])
        with col_a:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 👈 Nhập thông số và nhấn Predict!")
            st.info("Điền đầy đủ thông số ngôi nhà trong sidebar bên trái, sau đó nhấn **🔮 Predict House Price** để xem kết quả dự báo.")
            st.markdown("""
            **Hướng dẫn nhanh:**
            - 📐 Nhập diện tích các khu vực
            - 🛏️ Chọn số phòng, tiện nghi
            - ⭐ Đánh giá chất lượng nhà
            - 📍 Chọn khu vực và loại nhà
            - 🤖 Chọn model dự báo phù hợp
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        with col_b:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 🤖 Thông tin Model")
            for name, desc in MODEL_DESCRIPTIONS.items():
                st.markdown(f"**{name}** — {desc}")
            st.markdown('</div>', unsafe_allow_html=True)
        return

    # ── Validation ──
    if year_remod < year_built:
        st.warning("⚠️ Năm cải tạo không thể sớm hơn năm xây dựng! Vui lòng kiểm tra lại.")
        st.stop()

    # ── Load & predict ──
    try:
        model = load_model(MODEL_FILES[selected_model_name])
    except FileNotFoundError as e:
        st.error(f"⚠️ {e}")
        st.stop()

    try:
        X_pred = preprocess_input(user_input, scaler, label_encoders, feature_names)
        predicted_price = float(max(0, model.predict(X_pred)[0]))
    except Exception as e:
        st.error(f"❌ Lỗi khi dự báo: {e}")
        st.stop()

    lower = predicted_price * 0.90
    upper = predicted_price * 1.10

    # ── Layout: 3 columns ──
    col1, col2, col3 = st.columns([2, 2, 1.5])

    # ── Col 1: Price result ──
    with col1:
        st.markdown(f"""
        <div class="price-card">
            <div class="label">💰 Giá Dự Báo</div>
            <div class="price">${predicted_price:,.0f}</div>
            <div style="font-size:0.85rem;opacity:0.7;margin-top:4px">{selected_model_name}</div>
            <div class="range">
                Khoảng ±10%:
                <span class="range-badge">${lower:,.0f}</span> — <span class="range-badge">${upper:,.0f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Quick-stat pills
        age = 2025 - year_built
        price_per_sqft = predicted_price / gr_liv_area if gr_liv_area else 0
        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-pill blue">📐 {gr_liv_area:,} sq ft</div>
            <div class="stat-pill green">💲 ${price_per_sqft:,.0f}/sq ft</div>
            <div class="stat-pill pink">🏗️ {age} tuổi</div>
            <div class="stat-pill purple">⭐ Quality {overall_qual}/10</div>
        </div>
        """, unsafe_allow_html=True)

        # Price range bar
        st.markdown('<div class="section-header">📊 Khoảng giá ±10%</div>', unsafe_allow_html=True)
        st.plotly_chart(price_range_chart(predicted_price, lower, upper),
                        use_container_width=True, config={"displayModeBar": False})

    # ── Col 2: Gauge + Feature importance ──
    with col2:
        st.markdown('<div class="section-header">🎯 Gauge – Mức giá</div>', unsafe_allow_html=True)
        st.plotly_chart(gauge_chart(predicted_price), use_container_width=True,
                        config={"displayModeBar": False})

        fig_imp = feature_importance_chart(model, feature_names, selected_model_name)
        if fig_imp:
            st.plotly_chart(fig_imp, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("ℹ️ Linear Regression không có feature importance dạng bar chart.")

    # ── Col 3: Radar + Summary ──
    with col3:
        st.markdown('<div class="section-header">🕸️ Specs Radar</div>', unsafe_allow_html=True)
        st.plotly_chart(specs_radar(user_input), use_container_width=True,
                        config={"displayModeBar": False})

        st.markdown('<div class="section-header">📋 Thông số</div>', unsafe_allow_html=True)
        summary = pd.DataFrame({
            "Thông số": [
                "Living Area","Basement","1st Floor","Lot Area",
                "Bedrooms","Bathrooms","Quality","Year Built",
                "Neighborhood","Bldg Type","Central Air",
                "Kitchen","Exterior",
            ],
            "Giá trị": [
                f"{gr_liv_area:,} sqft", f"{total_bsmt_sf:,} sqft",
                f"{first_flr_sf:,} sqft", f"{lot_area:,} sqft",
                str(bedroom), str(full_bath),
                f"{overall_qual}/10", str(year_built),
                neighborhood, bldg_type_label, central_air_label,
                kitchen_qual_label.split()[0], exter_qual_label.split()[0],
            ]
        })
        st.dataframe(summary, use_container_width=True, hide_index=True, height=440)

if __name__ == "__main__":
    main()