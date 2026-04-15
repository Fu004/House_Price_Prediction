import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go

st.set_page_config(
    page_title="House Price Predictor · Ames Housing",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: #f5f4f0; }
#MainMenu, footer, header { visibility: hidden; }

[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #e8e6e0 !important;
}
[data-testid="stSidebar"] * { color: #000000 !important; }
[data-testid="stSidebar"] label {
    font-size: 12px !important;
    font-weight: 500 !important;
    color: #888780 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}
[data-testid="stSidebar"] .stButton > button {
    background-color: #1d6fca !important;
    color: #fff !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px !important;
    width: 100% !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #1558a8 !important;
}

.app-header {
    background: #ffffff;
    border-radius: 12px;
    padding: 20px 28px;
    margin-bottom: 16px;
    border: 1px solid #e8e6e0;
}
.app-header h1 { font-size: 20px; font-weight: 700; color: #2c2c2a; margin: 0 0 2px 0; }
.app-header p  { font-size: 13px; color: #888780; margin: 0; }

.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 16px;
}
.metric-card {
    background: #ffffff;
    border-radius: 10px;
    border: 1px solid #e8e6e0;
    padding: 18px 20px;
}
.metric-card .label {
    font-size: 12px; font-weight: 500; color: #888780;
    margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.4px;
}
.metric-card .value { font-size: 26px; font-weight: 700; color: #2c2c2a; line-height: 1.1; letter-spacing: -0.5px; }
.metric-card .value.blue { color: #1d6fca; }
.metric-card .sub { font-size: 12px; color: #888780; margin-top: 4px; }

.badge-up   { display:inline-block; background:#eaf3de; color:#3b6d11; border-radius:4px; padding:2px 8px; font-size:11px; font-weight:600; }
.badge-down { display:inline-block; background:#fcebeb; color:#a32d2d; border-radius:4px; padding:2px 8px; font-size:11px; font-weight:600; }

.chart-card {
    background: #ffffff;
    border-radius: 10px;
    border: 1px solid #e8e6e0;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.chart-title {
    font-size: 11px; font-weight: 600; color: #888780;
    text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 16px;
}

.corr-row  { display:flex; align-items:center; gap:10px; margin-bottom:10px; }
.corr-label{ font-size:12px; color:#2c2c2a; width:110px; flex-shrink:0; }
.corr-bar-bg  { flex:1; height:6px; background:#f0ede6; border-radius:3px; }
.corr-bar-fill{ height:6px; border-radius:3px; background:#1d6fca; }
.corr-val  { font-size:12px; font-weight:600; color:#2c2c2a; width:32px; text-align:right; }

.specs-table { width:100%; border-collapse:collapse; }
.specs-table tr { border-bottom:1px solid #f0ede6; }
.specs-table tr:last-child { border-bottom:none; }
.specs-table td { padding:7px 0; font-size:13px; }
.specs-table td:first-child { color:#888780; font-weight:500; width:55%; }
.specs-table td:last-child  { color:#2c2c2a; font-weight:600; text-align:right; }

.stAlert { border-radius:10px !important; }

[data-testid="stSelectbox"] div {
    background-color: #ffffff !important;
    color: #000000 !important;
}

/* Text trong selectbox */
[data-testid="stSelectbox"] * {
    color: #000000 !important;
}

/* Fix number input */
[data-testid="stNumberInput"] input {
    background-color: #ffffff !important;
    color: #000000 !important;
}

/* Nút tăng giảm */
[data-testid="stNumberInput"] button {
    background-color: #f0ede6 !important;
    color: #000000 !important;
    border: none !important;
}

/* Hover cho nút */
[data-testid="stNumberInput"] button:hover {
    background-color: #e0ddd6 !important;
}

/* Dropdown menu khi mở */
div[role="listbox"] {
    background-color: #ffffff !important;
    color: #000000 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────
MODEL_DIR = "models"
MODEL_FILES = {
    "Linear Regression": "linear_regression.pkl",
    "Random Forest":     "random_forest.pkl",
    "XGBoost":           "xgboost.pkl",
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
    "1Fam":"Single-family","2fmCon":"Two-family","Duplex":"Duplex",
    "TwnhsE":"Townhouse End","Twnhs":"Townhouse Inside",
}
CENTRAL_AIR_DISPLAY = {"Y":"Yes","N":"No"}
QUALITY_DISPLAY = {"Ex":"Excellent","Gd":"Good","TA":"Average","Fa":"Fair","Po":"Poor"}

CORRELATIONS = [
    ("Overall Qual",  0.80),("Gr Liv Area",  0.71),("Garage Cars",  0.65),
    ("Total Bsmt SF", 0.63),("1st Flr SF",   0.62),("Year Built",   0.56),
    ("Full Bath",     0.55),("TotRms AbvGrd",0.50),
]

# ── Loaders ───────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    for f in ["scaler.pkl","label_encoders.pkl","feature_names.pkl","categorical_options.pkl"]:
        if not os.path.exists(os.path.join(MODEL_DIR, f)):
            raise FileNotFoundError(f"'{f}' not found. Run model_training.py first.")
    return (
        joblib.load(os.path.join(MODEL_DIR, "scaler.pkl")),
        joblib.load(os.path.join(MODEL_DIR, "label_encoders.pkl")),
        joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl")),
        joblib.load(os.path.join(MODEL_DIR, "categorical_options.pkl")),
    )

@st.cache_resource
def load_model(filename):
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"'{path}' not found. Run model_training.py first.")
    return joblib.load(path)

# ── Helpers ───────────────────────────────────────────────────────
def selectbox_with_display(label, values, display_map, default=None):
    options = [display_map.get(v, v) for v in values]
    idx = values.index(default) if default and default in values else 0
    sel = st.selectbox(label, options=options, index=idx)
    return values[options.index(sel)], sel

def preprocess_input(user_input, scaler, label_encoders, feature_names):
    df = pd.DataFrame([user_input])
    for col in CATEGORICAL_FEATURES:
        le = label_encoders[col]
        val = str(df[col].iloc[0])
        if val not in le.classes_: val = le.classes_[0]
        df[col] = le.transform([val])
    df = df[feature_names]
    df[NUMERIC_FEATURES] = scaler.transform(df[NUMERIC_FEATURES])
    return df

def gauge_chart(value):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        number={"prefix":"$","valueformat":",.0f","font":{"size":20,"color":"#2c2c2a","family":"Inter"}},
        gauge={
            "axis":{"range":[50000,750000],"tickformat":"$,.0f",
                    "tickfont":{"size":9,"color":"#888780","family":"Inter"},"nticks":5},
            "bar":{"color":"#1d6fca","thickness":0.28},
            "bgcolor":"#f5f4f0","borderwidth":1,"bordercolor":"#e8e6e0",
            "steps":[
                {"range":[50000, 200000],"color":"#eaf3de"},
                {"range":[200000,350000],"color":"#e6f1fb"},
                {"range":[350000,500000],"color":"#faeeda"},
                {"range":[500000,750000],"color":"#fcebeb"},
            ],
            "threshold":{"line":{"color":"#1d6fca","width":3},"thickness":0.75,"value":value},
        },
    ))
    fig.update_layout(height=210, margin=dict(l=20,r=20,t=10,b=0),
                      paper_bgcolor="rgba(0,0,0,0)", font={"family":"Inter"})
    return fig

def range_chart(predicted, lo, hi):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[hi - lo], base=[lo], y=[""], orientation="h",
        marker_color="#b5d4f4", width=0.5, showlegend=False,
        hovertemplate=f"${lo:,.0f} – ${hi:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[predicted], y=[""], mode="markers+text",
        marker=dict(size=14, color="#1d6fca", symbol="diamond"),
        text=[f"${predicted:,.0f}"],
        textfont=dict(size=12, color="#1d6fca", family="Inter"),
        textposition="top center", showlegend=False,
        hovertemplate=f"Predicted: ${predicted:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        xaxis=dict(tickformat="$,.0f", showgrid=True, gridcolor="#f0ede6",
                   tickfont=dict(size=10,color="#888780")),
        yaxis=dict(showticklabels=False),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=90, margin=dict(l=0,r=0,t=28,b=0),
    )
    return fig

def importance_chart(model, feature_names):
    if not hasattr(model, "feature_importances_"):
        return None
    df_imp = pd.DataFrame({"f":feature_names,"v":model.feature_importances_})
    df_imp = df_imp.sort_values("v", ascending=True).tail(8)
    fig = go.Figure(go.Bar(
        x=df_imp["v"], y=df_imp["f"], orientation="h",
        marker_color="#1d6fca", opacity=0.85,
        text=[f"{v:.3f}" for v in df_imp["v"]],
        textposition="outside",
        textfont=dict(size=10,color="#888780",family="Inter"),
    ))
    fig.update_layout(
        xaxis=dict(showgrid=True,gridcolor="#f0ede6",showticklabels=False,title=""),
        yaxis=dict(showgrid=False,title="",tickfont=dict(size=11,color="#2c2c2a",family="Inter")),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=270, margin=dict(l=10,r=55,t=0,b=0),
    )
    return fig

def corr_html_block():
    html = ""
    for feat, val in CORRELATIONS:
        pct = int(val * 100)
        html += f"""<div class="corr-row">
            <span class="corr-label">{feat}</span>
            <div class="corr-bar-bg"><div class="corr-bar-fill" style="width:{pct}%"></div></div>
            <span class="corr-val">{val:.2f}</span>
        </div>"""
    return html

def neighborhood_chart():
    neighborhoods = ["NoRidge","NridgHt","StoneBr","Veenker","Timber","Somerst","Crawfor","CollgCr","Gilbert","NAmes"]
    prices        = [302000,318000,319000,250000,232000,226000,201000,200000,186000,153000]
    pairs = sorted(zip(prices, neighborhoods))
    prices_s, neigh_s = zip(*pairs)
    fig = go.Figure(go.Bar(
        x=list(prices_s), y=list(neigh_s), orientation="h",
        marker_color=["#1d6fca" if p == max(prices_s) else "#b5d4f4" for p in prices_s],
        text=[f"${p//1000}k" for p in prices_s],
        textposition="outside",
        textfont=dict(size=11,color="#888780",family="Inter"),
    ))
    fig.update_layout(
        xaxis=dict(showgrid=False,showticklabels=False,title=""),
        yaxis=dict(showgrid=False,title="",tickfont=dict(size=11,color="#2c2c2a",family="Inter")),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=310, margin=dict(l=0,r=60,t=0,b=0), showlegend=False,
    )
    return fig

# ── Main ──────────────────────────────────────────────────────────
def main():
    try:
        scaler, label_encoders, feature_names, cat_options = load_artifacts()
    except FileNotFoundError as e:
        st.error(f"⚠️ {e}")
        st.info("Run: `python model_training.py`")
        st.stop()

    # ── Sidebar ───────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🏠 House Price Predictor")
        st.caption("Ames Housing · Machine Learning")
        st.markdown("---")

        st.markdown("**Model**")
        selected_model_name = st.selectbox("", list(MODEL_FILES.keys()), label_visibility="collapsed")
        st.markdown("---")

        st.markdown("**Area (sq ft)**")
        c1, c2 = st.columns(2)
        with c1:
            gr_liv_area   = st.number_input("Living",   300,  6000, 1500, 50)
            total_bsmt_sf = st.number_input("Basement",   0,  3000,  800, 50)
        with c2:
            first_flr_sf  = st.number_input("1st Floor",300,  4000, 1000, 50)
            garage_area   = st.number_input("Garage",     0,  1500,  400, 50)
        lot_area = st.number_input("Lot Area (sq ft)", 1000, 100000, 10000, 500)

        st.markdown("**Rooms**")
        c3, c4 = st.columns(2)
        with c3:
            bedroom    = st.number_input("Bedrooms",    0, 8, 3, 1)
            full_bath  = st.number_input("Bathrooms",   0, 4, 2, 1)
            tot_rms    = st.number_input("Total rooms", 2,14, 7, 1)
        with c4:
            fireplaces  = st.number_input("Fireplaces", 0, 4, 1, 1)
            garage_cars = st.number_input("Garage cars",0, 4, 2, 1)

        st.markdown("**Year**")
        c5, c6 = st.columns(2)
        with c5: year_built = st.number_input("Built",    1872, 2010, 1980, 1)
        with c6: year_remod = st.number_input("Remodeled",1950, 2010, 2000, 1)

        st.markdown("**Quality**")
        overall_qual = st.slider("Overall Quality (1–10)", 1, 10, 6)
        overall_cond = st.slider("Overall Condition (1–10)", 1, 10, 5)

        st.markdown("**Location & Type**")
        neighborhood = st.selectbox("Neighborhood", cat_options.get("Neighborhood", ["NAmes"]))
        house_style, house_style_label = selectbox_with_display(
            "House Style", cat_options.get("House Style", ["1Story"]), HOUSE_STYLE_DISPLAY, "1Story")
        bldg_type, bldg_type_label = selectbox_with_display(
            "Building Type", cat_options.get("Bldg Type", ["1Fam"]), BLDG_TYPE_DISPLAY, "1Fam")
        central_air, central_air_label = selectbox_with_display(
            "Central Air", cat_options.get("Central Air", ["Y","N"]), CENTRAL_AIR_DISPLAY, "Y")
        kitchen_qual, kitchen_qual_label = selectbox_with_display(
            "Kitchen Quality", cat_options.get("Kitchen Qual", ["TA"]), QUALITY_DISPLAY, "TA")
        exter_qual, exter_qual_label = selectbox_with_display(
            "Exterior Quality", cat_options.get("Exter Qual", ["TA"]), QUALITY_DISPLAY, "TA")

        st.markdown("---")
        predict_button = st.button("Predict House Price →", type="primary", use_container_width=True)

    user_input = {
        "Gr Liv Area":gr_liv_area,"Total Bsmt SF":total_bsmt_sf,"1st Flr SF":first_flr_sf,
        "Garage Area":garage_area,"Lot Area":lot_area,"Year Built":year_built,
        "Year Remod/Add":year_remod,"Overall Qual":overall_qual,"Overall Cond":overall_cond,
        "TotRms AbvGrd":tot_rms,"Full Bath":full_bath,"Bedroom AbvGr":bedroom,
        "Fireplaces":fireplaces,"Garage Cars":garage_cars,"Neighborhood":neighborhood,
        "House Style":house_style,"Bldg Type":bldg_type,"Central Air":central_air,
        "Kitchen Qual":kitchen_qual,"Exter Qual":exter_qual,
    }

    # ── Header ────────────────────────────────────────────────────
    st.markdown("""
    <div class="app-header">
        <h1>Ames Housing Analytics</h1>
        <p>2,930 sales · 2006–2010 · Iowa, USA</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Placeholder (no prediction yet) ──────────────────────────
    if not predict_button:
        st.markdown("""
        <div class="metric-grid">
            <div class="metric-card">
                <div class="label">Median sale price</div>
                <div class="value">$160,000</div>
                <div class="sub">Mean $180,796 &nbsp;<span class="badge-down">−4.7% vs 2006</span></div>
            </div>
            <div class="metric-card">
                <div class="label">Total transactions</div>
                <div class="value">2,930</div>
                <div class="sub">Normal sales 82%</div>
            </div>
            <div class="metric-card">
                <div class="label">Price range</div>
                <div class="value" style="font-size:22px">$12.8k – $755k</div>
                <div class="sub">IQR $84,000</div>
            </div>
            <div class="metric-card">
                <div class="label">Top neighborhood</div>
                <div class="value blue">StoneBr</div>
                <div class="sub">Median $319,000 &nbsp;<span class="badge-up">+99% vs avg</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_l, col_r = st.columns([3, 2])
        with col_l:
            st.markdown('<div class="chart-card"><div class="chart-title">Median price by neighborhood · Top 10</div>', unsafe_allow_html=True)
            st.plotly_chart(neighborhood_chart(), use_container_width=True, config={"displayModeBar":False})
            st.markdown('</div>', unsafe_allow_html=True)

        with col_r:
            st.markdown('<div class="chart-card"><div class="chart-title">Correlations with sale price</div>', unsafe_allow_html=True)
            st.markdown(corr_html_block(), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("""
            <div class="chart-card">
                <div class="chart-title">Building type mix</div>
                <table class="specs-table">
                    <tr><td>Single-family</td><td>2,425</td></tr>
                    <tr><td>Townhouse End</td><td>233</td></tr>
                    <tr><td>Duplex</td><td>109</td></tr>
                    <tr><td>Townhouse Inside</td><td>101</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

        st.info("← Fill in house specifications in the sidebar and click **Predict House Price →**")
        return

    # ── Validation ────────────────────────────────────────────────
    if year_remod < year_built:
        st.warning("Remodel year cannot be earlier than build year.")
        st.stop()

    # ── Predict ───────────────────────────────────────────────────
    try:
        model = load_model(MODEL_FILES[selected_model_name])
        X_pred = preprocess_input(user_input, scaler, label_encoders, feature_names)
        predicted_price = float(max(0, model.predict(X_pred)[0]))
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.stop()

    lower = predicted_price * 0.90
    upper = predicted_price * 1.10
    age = 2025 - year_built
    price_per_sqft = predicted_price / gr_liv_area if gr_liv_area else 0
    vs_pct = ((predicted_price - 160000) / 160000) * 100
    vs_label = f"+{vs_pct:.1f}% vs median" if vs_pct >= 0 else f"{vs_pct:.1f}% vs median"
    badge_cls = "badge-up" if vs_pct >= 0 else "badge-down"

    # ── 4 metric cards ────────────────────────────────────────────
    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card">
            <div class="label">Predicted price</div>
            <div class="value blue">${predicted_price:,.0f}</div>
            <div class="sub"><span class="{badge_cls}">{vs_label}</span></div>
        </div>
        <div class="metric-card">
            <div class="label">Price range (±10%)</div>
            <div class="value" style="font-size:20px">${lower:,.0f} – ${upper:,.0f}</div>
            <div class="sub">Confidence interval</div>
        </div>
        <div class="metric-card">
            <div class="label">Price per sq ft</div>
            <div class="value">${price_per_sqft:,.0f}</div>
            <div class="sub">{gr_liv_area:,} sqft living area</div>
        </div>
        <div class="metric-card">
            <div class="label">House age</div>
            <div class="value">{age} yrs</div>
            <div class="sub">Built {year_built} · Remodeled {year_remod}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 3-column charts ───────────────────────────────────────────
    col1, col2, col3 = st.columns([2, 2, 1.6])

    with col1:
        st.markdown('<div class="chart-card"><div class="chart-title">Price estimate · gauge</div>', unsafe_allow_html=True)
        st.plotly_chart(gauge_chart(predicted_price), use_container_width=True, config={"displayModeBar":False})
        st.markdown('<div class="chart-title" style="margin-top:12px">Price range · USD</div>', unsafe_allow_html=True)
        st.plotly_chart(range_chart(predicted_price, lower, upper), use_container_width=True, config={"displayModeBar":False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="chart-card"><div class="chart-title">Correlations with sale price</div>', unsafe_allow_html=True)
        st.markdown(corr_html_block(), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        fig_imp = importance_chart(model, feature_names)
        if fig_imp:
            st.markdown(f'<div class="chart-card"><div class="chart-title">Feature importance · {selected_model_name}</div>', unsafe_allow_html=True)
            st.plotly_chart(fig_imp, use_container_width=True, config={"displayModeBar":False})
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="chart-card"><div class="chart-title">Feature importance</div><p style="color:#888780;font-size:13px">Not available for Linear Regression.</p></div>', unsafe_allow_html=True)

    with col3:
        specs = [
            ("Living area",   f"{gr_liv_area:,} sqft"),
            ("Basement",      f"{total_bsmt_sf:,} sqft"),
            ("1st floor",     f"{first_flr_sf:,} sqft"),
            ("Lot area",      f"{lot_area:,} sqft"),
            ("Garage area",   f"{garage_area:,} sqft"),
            ("Bedrooms",      str(bedroom)),
            ("Bathrooms",     str(full_bath)),
            ("Total rooms",   str(tot_rms)),
            ("Fireplaces",    str(fireplaces)),
            ("Garage cars",   str(garage_cars)),
            ("Quality",       f"{overall_qual}/10"),
            ("Condition",     f"{overall_cond}/10"),
            ("Neighborhood",  neighborhood),
            ("Style",         house_style_label),
            ("Type",          bldg_type_label),
            ("Central air",   central_air_label),
            ("Kitchen qual",  kitchen_qual_label),
            ("Exterior qual", exter_qual_label),
        ]
        rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in specs)
        st.markdown(f'<div class="chart-card"><div class="chart-title">House specifications</div><table class="specs-table">{rows}</table></div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()