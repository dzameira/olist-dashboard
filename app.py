import os
import json
import math
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import shap
import streamlit as st
from xgboost import XGBClassifier

# ==============================================================================
# PAGE CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(
    page_title="Olist — Delivery Delay & Weather Risk Radar",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    /* Clean light canvas & remove default sidebar/header clutter */
    [data-testid="stSidebar"] { display: none; }
    .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
        max-width: 1280px;
    }

    /* Brand lockup (left) */
    .brand-box {
        display: flex;
        align-items: center;
        font-weight: 750;
        font-size: 1.2rem;
        color: inherit;
        letter-spacing: -0.01em;
    }

    /* Dark left input panel matching reference UI */
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.dark-panel-marker) {
        background-color: #141812;
        border: 1px solid #262e22;
        border-radius: 22px;
        padding: 20px 22px;
        color: #f3f4f6;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.dark-panel-marker) label,
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.dark-panel-marker) p {
        color: #e5e7eb !important;
    }

    /* White bento output cards */
    .bento-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 20px;
        padding: 22px 24px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.03);
        margin-bottom: 14px;
        color: #111827;
    }
    .bento-label {
        font-size: 0.85rem;
        color: #6b7280;
        font-weight: 500;
        margin-bottom: 6px;
    }
    .bento-big-num {
        font-size: 2.85rem;
        font-weight: 800;
        line-height: 1.05;
        color: #111827;
        letter-spacing: -0.02em;
    }
    .bento-unit {
        font-size: 1.05rem;
        font-weight: 500;
        color: #6b7280;
        margin-left: 4px;
    }
    .bento-pill-warn {
        display: inline-block;
        background: #fee2e2;
        color: #b91c1c;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 999px;
        margin-top: 10px;
    }
    .bento-pill-ok {
        display: inline-block;
        background: #dcfce7;
        color: #15803d;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 999px;
        margin-top: 10px;
    }

    /* Right-hand vertical gradient risk level card */
    .risk-pillar-high {
        background: linear-gradient(165deg, #f87171 0%, #ef4444 50%, #dc2626 100%);
        border-radius: 20px;
        padding: 22px 20px;
        color: #ffffff;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 355px;
    }
    .risk-pillar-mod {
        background: linear-gradient(165deg, #fbbf24 0%, #f59e0b 55%, #d97706 100%);
        border-radius: 20px;
        padding: 22px 20px;
        color: #ffffff;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 355px;
    }
    .risk-pillar-low {
        background: linear-gradient(165deg, #4ade80 0%, #22c55e 55%, #16a34a 100%);
        border-radius: 20px;
        padding: 22px 20px;
        color: #ffffff;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 355px;
    }
    .risk-badge {
        background: rgba(255, 255, 255, 0.25);
        padding: 4px 12px;
        border-radius: 8px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        display: inline-block;
        margin-top: 8px;
    }

    /* Dotted progress bar */
    .dot-meter {
        letter-spacing: 3px;
        font-size: 1.05rem;
        margin: 14px 0 6px 0;
        font-family: monospace;
    }

    /* Blue rounded hero banner */
    .hero-banner {
        background: linear-gradient(125deg, #1d4ed8 0%, #2563eb 55%, #3b82f6 100%);
        border-radius: 22px;
        padding: 44px 48px;
        margin-top: 14px;
        margin-bottom: 28px;
        color: #ffffff;
        box-shadow: 0 12px 28px rgba(37, 99, 235, 0.18);
    }
    .hero-eyebrow {
        color: #bfdbfe;
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 10px;

    }
    .hero-title {
        font-size: 2.35rem;
        font-weight: 800;
        line-height: 1.15;
        margin: 0 0 14px 0;
        color: #ffffff;
    }
    .hero-sub {
        font-size: 1.02rem;
        line-height: 1.6;
        color: #d1d5db;
        max-width: 720px;
        margin: 0;
    }

    /* Editorial numbered section headers */
    .sec-eyebrow {
        color: #65a30d;
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .sec-title {
        font-size: 1.55rem;
        font-weight: 750;
        color: #111827;
        margin: 0 0 6px 0;
    }
    .sec-desc {
        color: #6b7280;
        font-size: 0.95rem;
        margin-bottom: 18px;
    }

    /* Clean info cards */
    .info-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 20px 22px;
        height: 100%;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    .info-card h4 {
        margin: 0 0 8px 0;
        font-size: 1.05rem;
        color: #111827;
    }
    .info-card p {
        margin: 0;
        font-size: 0.92rem;
        line-height: 1.55;
        color: #4b5563;
    }

    /* Risk alert cards for Page 2 */
    .risk-card-high {
        background-color: #fef2f2;
        border-left: 6px solid #dc2626;
        padding: 18px 20px;
        border-radius: 10px;
        color: #7f1d1d;
        margin: 12px 0;
    }
    .risk-card-mod {
        background-color: #fffbeb;
        border-left: 6px solid #d97706;
        padding: 18px 20px;
        border-radius: 10px;
        color: #78350f;
        margin: 12px 0;
    }
    .risk-card-low {
        background-color: #f0fdf4;
        border-left: 6px solid #16a34a;
        padding: 18px 20px;
        border-radius: 10px;
        color: #14532d;
        margin: 12px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

STATE_COORDS = {
    "AC": (-9.97, -67.81, "Rio Branco"),
    "AL": (-9.67, -35.74, "Maceió"),
    "AM": (-3.10, -60.02, "Manaus"),
    "AP": (0.03, -51.05, "Macapá"),
    "BA": (-12.97, -38.50, "Salvador"),
    "CE": (-3.72, -38.54, "Fortaleza"),
    "DF": (-15.78, -47.93, "Brasília"),
    "ES": (-20.32, -40.34, "Vitória"),
    "GO": (-16.68, -49.25, "Goiânia"),
    "MA": (-2.53, -44.30, "São Luís"),
    "MG": (-19.92, -43.94, "Belo Horizonte"),
    "MS": (-20.44, -54.65, "Campo Grande"),
    "MT": (-15.60, -56.10, "Cuiabá"),
    "PA": (-1.46, -48.50, "Belém"),
    "PB": (-7.12, -34.86, "João Pessoa"),
    "PE": (-8.05, -34.88, "Recife"),
    "PI": (-5.09, -42.80, "Teresina"),
    "PR": (-25.43, -49.27, "Curitiba"),
    "RJ": (-22.91, -43.17, "Rio de Janeiro"),
    "RN": (-5.79, -35.21, "Natal"),
    "RO": (-8.76, -63.90, "Porto Velho"),
    "RR": (2.82, -60.67, "Boa Vista"),
    "RS": (-30.03, -51.23, "Porto Alegre"),
    "SC": (-27.59, -48.55, "Florianópolis"),
    "SE": (-10.91, -37.07, "Aracaju"),
    "SP": (-23.55, -46.63, "São Paulo"),
    "TO": (-10.18, -48.33, "Palmas"),
}

RAIN_LEVEL_MAP = {
    "None": {"precip": 0.0, "rain_7d": 1.0},
    "Lightly": {"precip": 4.0, "rain_7d": 12.0},
    "Moderate": {"precip": 15.0, "rain_7d": 35.0},
    "Heavy": {"precip": 35.0, "rain_7d": 85.0},
}

WIND_LEVEL_MAP = {
    "Calm": 15.0,
    "Moderate": 35.0,
    "Strong": 68.0,
}
# ==============================================================================
# CACHED ARTIFACT & DATA LOADERS
# ==============================================================================
@st.cache_resource
def load_model_artifacts():
    if not os.path.exists("models/best_model.json"):
        return None

    model = XGBClassifier()
    model.load_model("models/best_model.json")

    with open("models/schema.json") as f:
        schema = json.load(f)

    with open("models/results.json") as f:
        results = json.load(f)

    lookups = joblib.load("models/inference_lookups.joblib")
    shap_sample = pd.read_parquet("models/shap_sample.parquet")
    global_shap_values = np.load("models/global_shap_values.npy")

    for col in schema["categorical_columns"]:
        if col in shap_sample.columns:
            shap_sample[col] = shap_sample[col].astype("category")

    explainer = shap.TreeExplainer(model)

    return {
        "model": model,
        "schema": schema,
        "results": results,
        "lookups": lookups,
        "shap_sample": shap_sample,
        "global_shap_values": global_shap_values,
        "explainer": explainer,
    }


@st.cache_data
def load_master_dataset():
    if os.path.exists("olist_master_featured_clean.parquet"):
        return pd.read_parquet("olist_master_featured_clean.parquet"), False
    if os.path.exists("olist_master_featured_clean.csv"):
        return pd.read_csv("olist_master_featured_clean.csv"), False

    rng = np.random.default_rng(42)
    n = 1000
    demo_df = pd.DataFrame({
        "is_late": rng.choice([0, 1], size=n, p=[0.9316, 0.0684]),
        "seller_state": rng.choice(["SP", "PR", "MG", "SC", "RJ"], size=n),
        "customer_state": rng.choice(["SP", "RJ", "MG", "RS", "BA", "SC"], size=n),
        "primary_category": rng.choice(["bed_bath_table", "health_beauty", "sports_leisure", "furniture_decor"], size=n),
        "promised_delivery_days": rng.uniform(5, 35, size=n).round(1),
        "delivery_distance_km": rng.uniform(20, 2200, size=n).round(1),
        "total_weight_g": rng.uniform(100, 15000, size=n).round(0),
        "total_freight": rng.uniform(10, 120, size=n).round(2),
        "customer_est_delivery_rain_7d": rng.exponential(18, size=n).round(1),
        "seller_purchase_apparent_temp_mean": rng.normal(24, 4, size=n).round(1),
        "any_severe_weather_flag": rng.choice([0, 1], size=n, p=[0.57, 0.43]),
    })
    return demo_df, True


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def classify_domain(feat):
    if feat.startswith(("seller_purchase_", "customer_est_delivery_", "route_")) or feat == "any_severe_weather_flag":
        return "Weather Bottlenecks (Open-Meteo)"
    if "hist_late_rate" in feat or feat in ("seller_order_count", "is_top_seller", "primary_category"):
        return "Seller & Category Reliability"
    if any(k in feat for k in ("sla", "promised", "confirmation", "month", "day", "hour", "weekend", "holiday")):
        return "SLA Buffer & Timing"
    if any(k in feat for k in ("distance", "lat", "lng", "state", "zip", "same_city", "complexity")):
        return "Route Distance & Geography"
    return "Package Dimensions & Freight"


def build_feature_row(inputs, schema, lookups):
    row = dict(schema["numeric_medians"])

    for col in schema["categorical_columns"]:
        choices = schema["category_choices"].get(col, ["unknown"])
        row[col] = choices[0] if choices else "unknown"

    s_state = inputs["seller_state"]
    c_state = inputs["customer_state"]
    cat = inputs["primary_category"]

    row["seller_state"] = s_state
    row["customer_state"] = c_state
    row["primary_category"] = cat
    row["primary_payment_type"] = inputs["primary_payment_type"]

    s_lat, s_lng, _ = STATE_COORDS.get(s_state, (-23.55, -46.63, "São Paulo"))
    c_lat, c_lng, _ = STATE_COORDS.get(c_state, (-22.91, -43.17, "Rio de Janeiro"))
    dist_km = max(5.0, haversine_km(s_lat, s_lng, c_lat, c_lng))

    row["seller_lat"] = s_lat
    row["seller_lng"] = s_lng
    row["customer_lat"] = c_lat
    row["customer_lng"] = c_lng
    row["delivery_distance_km"] = dist_km
    row["is_same_state"] = int(s_state == c_state)
    row["is_same_city"] = int(s_state == c_state and dist_km < 35)

    purchase_dt = pd.Timestamp(inputs["purchase_date"])
    promised_days = float(inputs["promised_delivery_days"])
    sla_days = float(inputs["seller_sla_window_days"])

    row["order_month"] = purchase_dt.month
    row["order_day_of_week"] = purchase_dt.dayofweek
    row["is_weekend"] = int(purchase_dt.dayofweek >= 5)
    row["is_holiday_season"] = int(purchase_dt.month in (11, 12))
    row["promised_delivery_days"] = promised_days
    row["seller_sla_window_days"] = sla_days
    row["sla_to_promised_ratio"] = sla_days / max(promised_days, 1.0)

    weight_g = float(inputs["total_weight_g"])
    vol_cm3 = float(inputs["total_volume_cm3"])
    freight = float(inputs["total_freight"])
    price = float(inputs["total_price"])

    row["total_weight_g"] = weight_g
    row["max_item_weight_g"] = weight_g
    row["total_volume_cm3"] = vol_cm3
    row["max_item_volume_cm3"] = vol_cm3
    row["total_volumetric_weight_g"] = vol_cm3 / 6.0
    row["avg_density_ratio"] = weight_g / max(vol_cm3, 1.0)
    row["total_freight"] = freight
    row["total_price"] = price
    row["max_item_price"] = price
    row["avg_price_per_item"] = price
    row["total_payment_value"] = price + freight
    row["total_shipping_ratio"] = freight / max(price, 1.0)
    row["freight_per_km"] = freight / max(dist_km, 1.0)
    row["freight_per_kg"] = freight / max(weight_g / 1000.0, 0.1)

    vol = int(inputs["seller_order_count"])
    row["seller_order_count"] = vol
    row["is_top_seller"] = int(vol >= 100)

    prior = schema["global_prior_late_rate"]
    row["primary_seller_id_hist_late_rate"] = float(inputs["seller_hist_late_rate"])
    row["customer_state_hist_late_rate"] = prior
    row["primary_category_hist_late_rate"] = prior

    try:
        te = lookups["target_encoder"]
        te_probe = pd.DataFrame({
            "primary_seller_id": ["__unseen_seller__"],
            "customer_state": [c_state],
            "primary_category": [cat],
        })
        encoded = te.transform(te_probe)
        row["customer_state_hist_late_rate"] = float(encoded[0, 1])
        row["primary_category_hist_late_rate"] = float(encoded[0, 2])
    except Exception:
        pass

    precip = float(inputs["precipitation_sum"])
    rain_7d = float(inputs["rain_7d"])
    wind_max = float(inputs["wind_gusts_max"])
    temp_mean = float(inputs["temperature_mean"])

    for prefix in ("seller_purchase_", "customer_est_delivery_"):
        if f"{prefix}precipitation_sum" in row:
            row[f"{prefix}precipitation_sum"] = precip
        if f"{prefix}rain_3d" in row:
            row[f"{prefix}rain_3d"] = rain_7d * 0.5
        if f"{prefix}rain_7d" in row:
            row[f"{prefix}rain_7d"] = rain_7d
        if f"{prefix}rain_14d" in row:
            row[f"{prefix}rain_14d"] = rain_7d * 1.6
        if f"{prefix}wind_gusts_max" in row:
            row[f"{prefix}wind_gusts_max"] = wind_max
        if f"{prefix}temperature_mean" in row:
            row[f"{prefix}temperature_mean"] = temp_mean
        if f"{prefix}apparent_temp_mean" in row:
            row[f"{prefix}apparent_temp_mean"] = temp_mean + (1.5 if temp_mean > 25 else 0.0)
        if f"{prefix}heavy_rain_flag" in row:
            row[f"{prefix}heavy_rain_flag"] = int(precip >= 10.0)
        if f"{prefix}very_heavy_rain_flag" in row:
            row[f"{prefix}very_heavy_rain_flag"] = int(precip >= 25.0)
        if f"{prefix}high_wind_flag" in row:
            row[f"{prefix}high_wind_flag"] = int(wind_max >= 45.0)
        if f"{prefix}extreme_heat_flag" in row:
            row[f"{prefix}extreme_heat_flag"] = int(temp_mean >= 32.0)

    if "any_severe_weather_flag" in row:
        row["any_severe_weather_flag"] = int(precip >= 10.0 or wind_max >= 45.0 or temp_mean >= 32.0 or rain_7d >= 30.0)
    if "route_precip_diff" in row:
        row["route_precip_diff"] = 0.0
    if "route_rain_7d_diff" in row:
        row["route_rain_7d_diff"] = 0.0

    df_row = pd.DataFrame([row])[schema["feature_columns"]]
    for col in schema["categorical_columns"]:
        cats = schema["category_choices"].get(col, [])
        df_row[col] = pd.Categorical(df_row[col], categories=cats)

    return df_row


# ==============================================================================
# LOAD RESOURCES & SIDEBAR NAVIGATION
# ==============================================================================
artifacts = load_model_artifacts()
master_df, is_demo = load_master_dataset()

PAGES = ["Olist Dashboard", "Prediction", "Model comparison", "SHAP explanations", "Key insights"]

c_left, c_right = st.columns([1.5, 3.5], vertical_alignment="center")

with c_left:
    st.markdown(
        '<div class="brand-box">Olist Delivery Dashboard</div>',
        unsafe_allow_html=True,
    )

with c_right:
    if "_goto" in st.session_state:
        st.session_state["nav"] = st.session_state.pop("_goto")
    st.session_state.setdefault("nav", "Olist Dashboard")
    page = st.segmented_control(
        "Navigate",
        PAGES,
        key="nav",
        label_visibility="collapsed",
    ) or "Olist Dashboard"

# ==============================================================================
# PAGE 1: INTRODUCTION & OVERVIEW
# ==============================================================================
if page == "Olist Dashboard":
    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-eyebrow">EXPLAINABLE LOGISTICS • OLIST + OPEN-METEO</div>
            <h1 class="hero-title">Predict Delivery Delays</h1>
            <p class="hero-sub">
                An early detection machine learning framework using Brazilian e-commerce order dataset with historical
                Open-Meteo weather data to detect late shipments at and explain every risk factor with SHAP.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 01 · PROBLEM STATEMENT & OBJECTIVE
    st.markdown(
        """
        <div class="sec-eyebrow">01 · Introduction</div>
        <h2 class="sec-title">Problem Statement & Research Objective</h2>
        <p class="sec-desc">Why early detection of delivery delays matters for customer retention.</p>
        """,
        unsafe_allow_html=True,
    )

    p_col, o_col = st.columns(2)
    with p_col:
        st.markdown(
            """
            <div class="info-card">
                <h4>Problem Statement</h4>
                <p>
                    Current e-commerce logistics systems still rely on reactive tracking, where they only flag delivery delays after they happen 
                    when it is already too late to prevent customer frustration. While advanced AI models can predict delays early with 
                    high accuracy, they often function as black-box systems that label shipments as high-risk without explaining the reasons. 
                    Additionally, most existing models leave out external weather factors, missing an important cause of delivery disruptions.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with o_col:
        st.markdown(
            """
            <div class="info-card">
                <h4>Research Objective</h4>
                <p>
                RO 1:  Combine Olist order data with Open-Meteo historical weather data by matching locations and dates.   
                <br> RO 2:  Train and compare XGBoost and LightGBM against Logistic Regression to predict delivery delays on imbalanced data.   
                <br> RO 3:  Use SHAP to find the main delay causes and risk limits, and show them in an interactive Streamlit dashboard. 
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    # 02 · DATASET KPIS
    st.markdown(
        """
        <div class="sec-eyebrow">02 · DATASET OVERVIEW</div>
        <h2 class="sec-title">Dataset KPIs</h2>
        <p class="sec-desc">Consolidated from 9 Olist tables and Open-Meteo data.</p>
        """,
        unsafe_allow_html=True,
    )

    total_rows = len(master_df)
    total_features = len(artifacts["schema"]["feature_columns"]) if artifacts else master_df.shape[1] - 1
    late_rate = master_df["is_late"].mean() * 100
    severe_wx_rate = master_df["any_severe_weather_flag"].mean() * 100 if "any_severe_weather_flag" in master_df.columns else 43.4

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Clean Orders", f"{total_rows:,}", "Zero-Null Master Matrix")
    k2.metric("Engineered Features", f"{total_features}", "Leakage control")
    k3.metric("Delayed Orders (Class 1)", f"{late_rate:.2f}%", f"{int(master_df['is_late'].sum()):,} Late Shipments", delta_color="inverse")
    k4.metric("Severe Weather Exposure", f"{severe_wx_rate:.1f}%", "Origin or Destination Window")

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    # 03 · TARGET DISTRIBUTION & SUMMARY STATISTICS
    st.markdown(
        """
        <div class="sec-eyebrow">03 · Dataset Distribution</div>
        <h2 class="sec-title">Target Distribution & Statistics</h2>
        """,
        unsafe_allow_html=True,
    )

    left_chart, right_stats = st.columns([1.05, 1.95])
    with left_chart:
        class_counts = master_df["is_late"].value_counts().rename({0: "On-Time (0)", 1: "Delayed (1)"}).reset_index()
        class_counts.columns = ["Delivery Status", "Orders"]
        fig_pie = px.pie(
            class_counts,
            names="Delivery Status",
            values="Orders",
            hole=0.52,
            color="Delivery Status",
            color_discrete_map={"On-Time (0)": "#16a34a", "Delayed (1)": "#dc2626"},
        )
        fig_pie.update_layout(
            height=330,
            margin=dict(t=20, b=20, l=10, r=10),
            legend=dict(orientation="h", yanchor="bottom", y=-0.12, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with right_stats:
        key_summary_cols = [
            c for c in [
                "promised_delivery_days",
                "delivery_distance_km",
                "total_weight_g",
                "total_freight",
                "customer_est_delivery_rain_7d",
                "seller_purchase_apparent_temp_mean",
            ]
            if c in master_df.columns
        ]
        summary_tbl = master_df[key_summary_cols].describe().T[["count", "mean", "std", "50%", "max"]].round(2)
        summary_tbl.columns = ["Count", "Mean", "Std Dev", "Median", "Max"]
        st.dataframe(summary_tbl, use_container_width=True, height=330)


# ==============================================================================
# PAGE 2: PREDICTION (LIVE "WHAT-IF" EARLY-WARNING SIMULATOR)
# ==============================================================================
elif page == "Prediction":
    if artifacts is None:
        st.error("Model artifacts missing. Please run `train_models.py` first.")
        st.stop()

    schema = artifacts["schema"]
    lookups = artifacts["lookups"]
    model = artifacts["model"]
    explainer = artifacts["explainer"]
    shap_sample = artifacts["shap_sample"]

    if "ref_probs" not in st.session_state:
        st.session_state["ref_probs"] = model.predict_proba(shap_sample)[:, 1]
    ref_probs = st.session_state["ref_probs"]

    states_sorted = sorted(STATE_COORDS.keys())
    categories_sorted = schema["category_choices"].get("primary_category", ["bed_bath_table", "health_beauty"])
    payment_sorted = schema["category_choices"].get("primary_payment_type", ["credit_card", "boleto"])

    # --------------------------------------------------------------------------
    # TOP SECTION: TITLE ON LEFT | WIDE PREDICTORS CONTAINER ON RIGHT
    # --------------------------------------------------------------------------
    hdr_col, form_col = st.columns([1.05, 2.55], gap="large", vertical_alignment="top")

    with hdr_col:
        st.markdown(
            """
            <h1 style="font-size:2.3rem; font-weight:800; line-height:1.12; margin:4px 0 12px 0;">
                Your delivery<br>risk profile
            </h1>
            <p style="color:#6b7280; font-size:0.94rem; line-height:1.55; margin:0;">
                Adjust the route, package, and weather factors beside this panel and the delay risk estimate will updates straight away below.
            </p>
            """,
            unsafe_allow_html=True,
        )

    with form_col:
        with st.container(border=True):
            st.markdown('<span class="dark-panel-marker"></span>', unsafe_allow_html=True)

            # Row 1: Date, From (Seller), To (Customer)
            r1_c1, r1_c2, r1_c3 = st.columns(3, gap="medium")
            purchase_date = r1_c1.date_input("Purchase date", value=pd.to_datetime("2026-11-12"))
            seller_state = r1_c2.selectbox(
                "From (Seller)",
                states_sorted,
                index=states_sorted.index("SP"),
                format_func=lambda s: f"{s} ({STATE_COORDS[s][2]})",
            )
            customer_state = r1_c3.selectbox(
                "To (Customer)",
                states_sorted,
                index=states_sorted.index("RJ"),
                format_func=lambda s: f"{s} ({STATE_COORDS[s][2]})",
            )

            # Row 2: Package Weight, Freight Cost, Order Value
            r2_c1, r2_c2, r2_c3 = st.columns(3, gap="medium")
            total_weight_g = r2_c1.number_input("Package weight (g)", min_value=50, max_value=50000, value=1800, step=100)
            total_freight = r2_c2.number_input("Freight cost (BRL)", min_value=5.0, max_value=500.0, value=28.5, step=1.0)
            total_price = r2_c3.number_input("Order value (BRL)", min_value=10.0, max_value=5000.0, value=149.9, step=10.0)

            st.markdown("<hr style='border-color:#2c3527; margin:10px 0 12px 0;'>", unsafe_allow_html=True)

            # Row 3: Weather Controls (Rain, Wind, Temperature)
            r3_c1, r3_c2, r3_c3 = st.columns(3, gap="medium")
            rain_label = r3_c1.select_slider(
                "Rain condition",
                options=["None", "Lightly", "Moderate", "Heavy"],
                value="Moderate",
            )
            wind_label = r3_c2.select_slider(
                "Wind condition",
                options=["Calm", "Moderate", "Strong"],
                value="Moderate",
            )
            temperature_mean = r3_c3.slider(
                "Temperature (°C)",
                min_value=10.0,
                max_value=40.0,
                value=28.0,
                step=0.5,
            )

    # --------------------------------------------------------------------------
    # RUN INFERENCE & SHAP ATTRIBUTION
    # --------------------------------------------------------------------------
    rain_vals = RAIN_LEVEL_MAP[rain_label]
    wind_val = WIND_LEVEL_MAP[wind_label]

    sim_inputs = {
        "seller_state": seller_state,
        "customer_state": customer_state,
        "primary_category": "bed_bath_table" if "bed_bath_table" in categories_sorted else categories_sorted[0],
        "purchase_date": purchase_date,
        "promised_delivery_days": float(schema["numeric_medians"].get("promised_delivery_days", 16.0)),
        "seller_sla_window_days": float(schema["numeric_medians"].get("seller_sla_window_days", 4.0)),
        "seller_hist_late_rate": float(schema["global_prior_late_rate"]),
        "total_weight_g": total_weight_g,
        "total_volume_cm3": float(schema["numeric_medians"].get("total_volume_cm3", 8000.0)),
        "total_freight": total_freight,
        "total_price": total_price,
        "precipitation_sum": rain_vals["precip"],
        "rain_7d": rain_vals["rain_7d"],
        "wind_gusts_max": wind_val,
        "temperature_mean": temperature_mean,
        "seller_order_count": int(schema["numeric_medians"].get("seller_order_count", 50)),
        "primary_payment_type": payment_sorted[0],
    }

    X_sim = build_feature_row(sim_inputs, schema, lookups)
    prob = float(model.predict_proba(X_sim)[0, 1])
    shap_exp = explainer(X_sim)
    local_shap = shap_exp.values[0]

    st.session_state["last_sim_X"] = X_sim
    st.session_state["last_sim_prob"] = prob
    st.session_state["last_sim_shap"] = shap_exp

    base_rate = float(schema["global_prior_late_rate"])
    diff_pp = (prob - base_rate) * 100
    percentile = int(round((ref_probs < prob).mean() * 100))
    dist_km = float(X_sim["delivery_distance_km"].iloc[0])

    if prob >= 0.69:
        tier_label = "HIGH"
        pillar_cls = "risk-pillar-high"
    elif prob >= 0.40:
        tier_label = "MODERATE"
        pillar_cls = "risk-pillar-mod"
    else:
        tier_label = "LOW"
        pillar_cls = "risk-pillar-low"

    filled_dots = min(36, max(1, int(round(prob * 36))))
    empty_dots = 36 - filled_dots
    dot_html = (
        f'<span style="color:#111827;">{"●" * (filled_dots - 1)}○</span>'
        f'<span style="color:#d1d5db;">{"●" * empty_dots}</span>'
    )

    st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # BOTTOM SECTION: FULL-WIDTH BENTO RESULTS GRID
    # --------------------------------------------------------------------------
    res_left, res_right = st.columns([2.35, 1.0], gap="medium")

    with res_left:
        pill_cls = "bento-pill-warn" if diff_pp > 0 else "bento-pill-ok"
        pill_txt = (
            f"{abs(diff_pp):.1f}% above the dataset average"
            if diff_pp > 0
            else f"{abs(diff_pp):.1f}% below the dataset average"
        )
        st.markdown(
            f"""
            <div class="bento-card">
                <div class="bento-label">Predicted delay probability</div>
                <div>
                    <span class="bento-big-num">{prob * 100:.1f}%</span>
                    <span class="bento-unit">disruption risk</span>
                </div>
                <div class="{pill_cls}">{pill_txt}</div>
                <div class="dot-meter">{dot_html}</div>
                <div style="display:flex; justify-content:space-between; font-size:0.78rem; color:#6b7280;">
                    <span>Lower risk</span>
                    <span>Higher than {percentile}% of the reference sample</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        sub1, sub2 = st.columns(2, gap="medium")
        with sub1:
            corridor_type = "Interstate corridor" if seller_state != customer_state else "Intra-state local route"
            st.markdown(
                f"""
                <div class="bento-card" style="margin-bottom:0;">
                    <div class="bento-label">Route distance</div>
                    <div style="font-size:1.85rem; font-weight:800; color:#111827;">
                        {dist_km:,.0f} <span style="font-size:0.95rem; font-weight:500; color:#6b7280;">km</span>
                    </div>
                    <p style="font-size:0.8rem; color:#6b7280; margin:6px 0 0 0;">
                        {seller_state} → {customer_state} ({corridor_type}). Longer routes makes travel times less predictable.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with sub2:
            ratio = prob / max(base_rate, 1e-4)
            st.markdown(
                f"""
                <div class="bento-card" style="margin-bottom:0;">
                    <div class="bento-label">Dataset average</div>
                    <div style="font-size:1.85rem; font-weight:800; color:#111827;">
                        {base_rate * 100:.2f}% <span style="font-size:0.95rem; font-weight:500; color:#6b7280;">late</span>
                    </div>
                    <p style="font-size:0.8rem; color:#6b7280; margin:6px 0 0 0;">
                        Your estimate is {ratio:.1f}× the baseline rate. SHAP bars below explain the gap.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with res_right:
        st.markdown(
            f"""
            <div class="{pillar_cls}">
                <div>
                    <div style="font-size:0.85rem; font-weight:600; opacity:0.92;">Disruption Risk Level</div>
                    <div class="risk-badge">{tier_label}</div>
                </div>
                <div style="text-align:center; opacity:0.85; font-size:2.4rem;">◉</div>
                <div>
                    <div style="font-size:2.4rem; font-weight:800; line-height:1;">{percentile}%</div>
                    <p style="font-size:0.78rem; opacity:0.9; margin:6px 0 0 0; line-height:1.4;">
                        of orders in the reference sample have a lower delay risk than this shipment
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

    # Full-Width Bottom Card: SHAP Drivers + Proactive Mitigation
    PRETTY_NAMES = {
        "customer_state": "Customer Destination State",
        "seller_state": "Seller Origin State",
        "delivery_distance_km": "Route Distance (km)",
        "is_same_state": "Same-State vs. Interstate",
        "order_month": "Order Month (Seasonality)",
        "total_weight_g": "Package Weight (g)",
        "total_freight": "Freight Cost (BRL)",
        "total_price": "Order Value (BRL)",
        "seller_purchase_precipitation_sum": "Origin Daily Rain",
        "customer_est_delivery_rain_7d": "Destination 7-Day Rain",
        "seller_purchase_rain_7d": "Origin 7-Day Rain",
        "customer_est_delivery_apparent_temp_mean": "Destination Temperature",
        "seller_purchase_apparent_temp_mean": "Origin Temperature",
        "customer_est_delivery_wind_gusts_max": "Destination Wind Speed",
        "seller_purchase_wind_gusts_max": "Origin Wind Speed",
        "promised_delivery_days": "Promised SLA Window",
    }

    top_idx = np.argsort(np.abs(local_shap))[::-1][:6]
    max_abs = max(float(np.max(np.abs(local_shap[top_idx]))), 1e-4)

    bars_html = ""
    domain_push = {}
    for feat_name, s_val in zip(X_sim.columns, local_shap):
        dom = classify_domain(feat_name)
        domain_push[dom] = domain_push.get(dom, 0.0) + max(0.0, float(s_val))

    for i in top_idx:
        fname = X_sim.columns[i]
        val = float(local_shap[i])
        label = PRETTY_NAMES.get(fname, fname.replace("_", " ").title())
        bar_pct = min(100, int(round((abs(val) / max_abs) * 100)))
        if val >= 0:
            bar_color = "#f43f5e"
            val_str = f"+{val:.2f}"
            val_color = "#e11d48"
            bar_align = f"width:{bar_pct / 2:.1f}%; margin-left:50%; background:{bar_color};"
        else:
            bar_color = "#65a30d"
            val_str = f"{val:.2f}"
            val_color = "#4d7c0f"
            bar_align = f"width:{bar_pct / 2:.1f}%; margin-left:{50 - bar_pct / 2:.1f}%; background:{bar_color};"

        bars_html += f"""
        <div style="display:flex; align-items:center; margin:9px 0; font-size:0.88rem;">
            <div style="width:28%; text-align:right; padding-right:16px; color:#374151; font-weight:500;">
                {label}
            </div>
            <div style="width:62%; background:#f3f4f6; height:10px; border-radius:999px; overflow:hidden;">
                <div style="height:100%; border-radius:999px; {bar_align}"></div>
            </div>
            <div style="width:10%; text-align:right; font-weight:700; color:{val_color}; font-family:monospace;">
                {val_str}
            </div>
        </div>
        """

    st.markdown(
        f"""
        <div class="bento-card">
            <div class="bento-label" style="margin-bottom:12px;">What drives this estimate (SHAP log-odds contribution)</div>
            {bars_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    top_root_cause = max(domain_push, key=domain_push.get)
    if prob >= 0.40:
        if "Weather" in top_root_cause:
            st.warning(
                "**Proactive Action (Weather):** Switch to an express courier "
                "that can bypass regional weather, and send alert to the customer."
            )
        elif "Package" in top_root_cause or "Seller" in top_root_cause:
            st.warning(
                "💡 **Proactive Action (Origin Terminal Priority):** Prioritize order for carrier pickup at the origin terminal "
                "to compensate for heavy package sorting time."
            )
        else:
            st.warning(
                "💡 **Proactive Action (SLA Buffer Adjustment):** Add 2 to 4 buffer days to the checkout delivery estimate for this " \
                "interstate route to prevent late-delivery penalties."
            )

# ==============================================================================
# PAGE 3: MODEL COMPARISON
# ==============================================================================

elif page == "Model comparison":
    if artifacts is None:
        st.error("Model artifacts missing. Please run `train_models.py` first.")
        st.stop()

    metrics_dict = artifacts["results"]["metrics"]
    best_model_name = artifacts["results"]["best_model"]

    # Order models with the winning model first
    ordered_models = [best_model_name] + [m for m in ["LightGBM", "Logistic Regression", "XGBoost"] if m in metrics_dict and m != best_model_name]

    cards_cols = st.columns(len(ordered_models), gap="medium")

    for col, m_name in zip(cards_cols, ordered_models):
        m_vals = metrics_dict[m_name]
        is_best = m_name == best_model_name
        lift_x = m_vals["PR_AUC"] / 0.0685

        card_style = (
            "background: linear-gradient(155deg, #4ade80 0%, #22c55e 50%, #16a34a 100%); color: #ffffff; border: none;"
            if is_best
            else "background: #ffffff; color: #111827; border: 1px solid #e5e7eb;"
        )
        muted_color = "rgba(255,255,255,0.85)" if is_best else "#6b7280"
        divider_color = "rgba(255,255,255,0.22)" if is_best else "#e5e7eb"

        badge_html = (
            '<div style="display:inline-block; background:rgba(255,255,255,0.22); border:1px solid rgba(255,255,255,0.45); '
            'border-radius:999px; padding:3px 12px; font-size:0.75rem; font-weight:700; margin-bottom:10px;">Best model</div>'
            if is_best
            else '<div style="height:33px; margin-bottom:10px;"></div>'
        )

        with col:
            html_card = "\n".join(
                line.lstrip()
                for line in f"""
                <div style="{card_style} border-radius:22px; padding:26px 26px 22px 26px; box-shadow:0 4px 16px rgba(15,23,42,0.04);">
                    {badge_html}
                    <div style="font-size:0.95rem; font-weight:600; color:{muted_color}; margin-bottom:4px;">{m_name}</div>
                    <div style="margin-bottom:18px;">
                        <span style="font-size:2.75rem; font-weight:800; line-height:1.05; letter-spacing:-0.02em;">
                            {m_vals['PR_AUC']:.4f}
                        </span>
                        <span style="font-size:0.98rem; font-weight:500; color:{muted_color}; margin-left:4px;">
                            PR-AUC
                        </span>
                        <div style="font-size:0.78rem; color:{muted_color}; margin-top:4px; font-weight:600;">
                            {lift_x:.1f}× above 0.0685 random baseline
                        </div>
                    </div>
                    <div style="border-top:1px solid {divider_color}; padding:11px 0; display:flex; justify-content:space-between; font-size:0.88rem;">
                        <span style="color:{muted_color};">ROC-AUC</span>
                        <span style="font-weight:700;">{m_vals['ROC_AUC']:.4f}</span>
                    </div>
                    <div style="border-top:1px solid {divider_color}; padding:11px 0; display:flex; justify-content:space-between; font-size:0.88rem;">
                        <span style="color:{muted_color};">F1-Score (Opt)</span>
                        <span style="font-weight:700;">{m_vals['F1_at_Opt']:.4f}</span>
                    </div>
                    <div style="border-top:1px solid {divider_color}; padding:11px 0; display:flex; justify-content:space-between; font-size:0.88rem;">
                        <span style="color:{muted_color};">Precision / Recall</span>
                        <span style="font-weight:700;">{m_vals['Precision_at_Opt']*100:.1f}% / {m_vals['Recall_at_Opt']*100:.1f}%</span>
                    </div>
                    <div style="border-top:1px solid {divider_color}; padding-top:11px; display:flex; justify-content:space-between; font-size:0.88rem;">
                        <span style="color:{muted_color};">Optimal Threshold</span>
                        <span style="font-weight:700;">{m_vals['Optimal_Threshold']:.2f}</span>
                    </div>
                </div>
                """.splitlines()
                if line.strip()
            )
            st.markdown(html_card, unsafe_allow_html=True)

    # Metric comparison across models bar chart
    # Metric comparison across models (matching SHAP explanations UI)
    st.subheader("Metric comparison across models")

    metric_specs = [
        ("PR-AUC (Primary)", "PR_AUC"),
        ("ROC-AUC", "ROC_AUC"),
        ("F1-Score (Opt)", "F1_at_Opt"),
        ("Precision (Opt)", "Precision_at_Opt"),
        ("Recall (Opt)", "Recall_at_Opt"),
    ]

    model_bar_styles = {
        "XGBoost": "linear-gradient(90deg, #16a34a 0%, #4ade80 100%)",
        "LightGBM": "linear-gradient(90deg, #2563eb 0%, #60a5fa 100%)",
        "Logistic Regression": "linear-gradient(90deg, #6b7280 0%, #9ca3af 100%)",
    }

    legend_html = (
        '<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:20px; flex-wrap:wrap; gap:10px;">'
        '<span style="color:#6b7280; font-size:0.84rem;">Held-out stratified test set performance (0.000 – 1.000 scale)</span>'
        '<div style="display:flex; gap:18px; font-size:0.82rem; font-weight:600; color:#374151;">'
        '<span><span style="display:inline-block; width:10px; height:10px; border-radius:50%; background:#22c55e; margin-right:6px;"></span>XGBoost</span>'
        '<span><span style="display:inline-block; width:10px; height:10px; border-radius:50%; background:#3b82f6; margin-right:6px;"></span>LightGBM</span>'
        '<span><span style="display:inline-block; width:10px; height:10px; border-radius:50%; background:#9ca3af; margin-right:6px;"></span>Logistic Regression</span>'
        '</div>'
        '</div>'
    )

    rows_html = ""
    for metric_label, metric_key in metric_specs:
        bars_stack = ""
        for m_name in ordered_models:
            val = float(metrics_dict[m_name][metric_key])
            pct = min(100.0, max(4.0, val * 100.0))
            grad = model_bar_styles.get(m_name, "linear-gradient(90deg, #2563eb 0%, #60a5fa 100%)")
            bars_stack += (
                f'<div style="display:flex; align-items:center; margin:4px 0;">'
                f'<div style="flex:1; background:#f1f5f9; height:9px; border-radius:999px; overflow:hidden;">'
                f'<div style="width:{pct:.1f}%; height:100%; border-radius:999px; background:{grad};"></div>'
                f'</div>'
                f'<div style="width:58px; text-align:right; color:#374151; font-size:0.82rem; font-weight:700; font-family:monospace;">'
                f'{val:.4f}'
                f'</div>'
                f'</div>'
            )

        rows_html += (
            f'<div style="display:flex; align-items:center; padding:12px 0; border-top:1px solid #f3f4f6;">'
            f'<div style="width:22%; text-align:right; padding-right:20px; color:#111827; font-size:0.9rem; font-weight:600;">'
            f'{metric_label}'
            f'</div>'
            f'<div style="width:78%;">'
            f'{bars_stack}'
            f'</div>'
            f'</div>'
        )

    comp_card_html = (
        f'<div style="background:#ffffff; border:1px solid #e5e7eb; border-radius:24px; '
        f'padding:26px 32px; margin-top:8px; box-shadow:0 4px 16px rgba(15,23,42,0.04);">'
        f'{legend_html}'
        f'{rows_html}'
        f'</div>'
    )
    st.markdown(comp_card_html, unsafe_allow_html=True)


# ==============================================================================
# PAGE 4: SHAP EXPLANATIONS (DARK GRADIENT BAR CARD + LOCAL WATERFALL)
# ==============================================================================
elif page == "SHAP explanations":
    st.markdown(
        """
        <h1 style="font-size:2.45rem; font-weight:800; line-height:1.1; margin:0 0 10px 0;">
            Delivery delay causes
        </h1>
        <p style="color:#6b7280; font-size:0.95rem; margin:0 0 22px 0;">
            Global TreeSHAP feature attributions across 1,000 test shipments, plus single-order local breakdowns.
        </p>
        """,
        unsafe_allow_html=True,
    )

    if artifacts is None:
        st.error("Model artifacts missing. Please run `train_models.py` first.")
        st.stop()

    shap_sample = artifacts["shap_sample"]
    global_shap_values = artifacts["global_shap_values"]
    explainer = artifacts["explainer"]
    model = artifacts["model"]

    # View toggle matching reference UI
    st.markdown('<div style="font-size:0.82rem; color:#6b7280; margin-bottom:4px;">View</div>', unsafe_allow_html=True)
    shap_view = st.segmented_control(
        "SHAP View",
        ["By individual feature", "By operational domain"],
        default="By individual feature",
        label_visibility="collapsed",
    ) or "By individual feature"

    PRETTY_FEATURE_LABELS = {
        "promised_delivery_days": "Promised Delivery Window (Days)",
        "customer_state": "Customer Destination State",
        "primary_category": "Product Category",
        "order_month": "Order Month (Seasonality)",
        "primary_seller_id_hist_late_rate": "Seller Historical Late Rate",
        "seller_purchase_apparent_temp_mean": "Origin Apparent Temperature",
        "sla_to_promised_ratio": "Seller SLA-to-Promised Ratio",
        "customer_est_delivery_apparent_temp_mean": "Destination Apparent Temperature",
        "total_freight": "Freight Cost (BRL)",
        "customer_est_delivery_rain_14d": "Destination 14-Day Rainfall",
        "primary_category_hist_late_rate": "Category Historical Late Rate",
        "is_same_state": "Same-State vs. Interstate",
        "seller_purchase_rain_7d": "Origin 7-Day Rainfall",
        "delivery_distance_km": "Route Distance (km)",
        "customer_zip_code_prefix": "Customer Zip Region",
    }

    mean_abs = np.abs(global_shap_values).mean(axis=0)

    if shap_view == "By operational domain":
        dom_totals = {}
        for feat_name, val in zip(shap_sample.columns, mean_abs):
            dom = classify_domain(feat_name)
            dom_totals[dom] = dom_totals.get(dom, 0.0) + float(val)
        sorted_items = sorted(dom_totals.items(), key=lambda x: x[1], reverse=True)
        unit_suffix = ""
    else:
        top_indices = np.argsort(mean_abs)[::-1][:12]
        sorted_items = [
            (PRETTY_FEATURE_LABELS.get(shap_sample.columns[i], shap_sample.columns[i].replace("_", " ").title()), float(mean_abs[i]))
            for i in top_indices
        ]
        unit_suffix = ""

    max_val = max(item[1] for item in sorted_items) if sorted_items else 1.0

    rows_html = ""
    for label, val in sorted_items:
        pct = min(100.0, max(4.0, (val / max_val) * 100.0))
        rows_html += (
            f'<div style="display:flex; align-items:center; margin:13px 0; font-size:0.88rem;">'
            f'<div style="width:27%; text-align:right; padding-right:18px; color:#111827; font-weight:500;">{label}</div>'
            f'<div style="width:63%; background:#f1f5f9; height:12px; border-radius:999px; overflow:hidden;">'
            f'<div style="width:{pct:.1f}%; height:100%; border-radius:999px; '
            f'background:linear-gradient(90deg, #2563eb 0%, #60a5fa 100%);"></div>'
            f'</div>'
            f'<div style="width:10%; text-align:right; color:#374151; font-weight:600; font-family:monospace;">'
            f'{val:.3f}{unit_suffix}'
            f'</div>'
            f'</div>'
        )

    light_card_html = (
        f'<div style="background:#ffffff; border:1px solid #e5e7eb; border-radius:24px; '
        f'padding:28px 32px; margin-top:12px; box-shadow:0 4px 16px rgba(15,23,42,0.04);">'
        f'<div style="color:#6b7280; font-size:0.84rem; margin-bottom:18px;">'
        f'Mean absolute SHAP value (log-odds impact on delay probability)'
        f'</div>'
        f'{rows_html}'
        f'</div>'
    )
    st.markdown(light_card_html, unsafe_allow_html=True)

# ==============================================================================
# PAGE 5: KEY INSIGHTS (WEATHER VERDICT + GREEN TIPPING CARDS + SHAP SHARE BAR)
# ==============================================================================
elif page == "Key insights":
    st.markdown(
        """
        <h1 style="font-size:2.45rem; font-weight:800; line-height:1.1; margin:0 0 12px 0;">
            Does weather influence delivery delays prediction?
        </h1>
        <p style="color:#6b7280; font-size:0.98rem; line-height:1.6; max-width:820px; margin:0 0 26px 0;">
            <strong style="color:#16a34a;">Yes — measurably.</strong> Open-Meteo weather features account for
            <strong>26.1%</strong> of total model predictive importance (the #1 domain overall) and lift test PR-AUC from
            <strong>0.2921 → 0.3132 (+7.22%)</strong> across all orders and <strong>0.3113 → 0.3344</strong> during severe weather.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------------------------
    # 1. OPERATIONAL TIPPING POINTS (4 ALL-GREEN GRADIENT CARDS)
    # --------------------------------------------------------------------------
    st.subheader("Discovered operational tipping points")

    tipping_cards = [
        {
            "title": "Rainfall Risk Tipping Point",
            "rank": "Weather • Rank 1",
            "value": "> 13.1 mm",
            "sub": "7-day rain • 100% positive SHAP",
        },
        {
            "title": "Distance Risk Tipping Point",
            "rank": "Distance • Rank 2",
            "value": "> 444 km",
            "sub": "100% positive risk above 831 km",
        },
        {
            "title": "Seller Late-Rate Threshold",
            "rank": "Seller Reliability • Rank 3",
            "value": "> 6.0%",
            "sub": "99.2% positive risk above 9.25%",
        },
        {
            "title": "Interstate vs. Same-State",
            "rank": "Crossing State • Rank 4",
            "value": "1.78×",
            "sub": "8.12% vs. 4.56% ground-truth late",
        },
    ]

    t_cols = st.columns(4, gap="medium")
    for col, card in zip(t_cols, tipping_cards):
        with col:
            card_html = (
                f'<div style="background:linear-gradient(155deg, #4ade80 0%, #22c55e 52%, #16a34a 100%); '
                f'border-radius:22px; padding:22px 20px; color:#ffffff; min-height:205px; '
                f'display:flex; flex-direction:column; justify-content:space-between; '
                f'box-shadow:0 6px 18px rgba(22,163,74,0.16);">'
                f'<div>'
                f'<div style="font-size:1.12rem; font-weight:750; line-height:1.25; margin-bottom:6px;">{card["title"]}</div>'
                f'<div style="font-size:0.8rem; opacity:0.88; font-weight:500;">{card["rank"]}</div>'
                f'</div>'
                f'<div style="background:rgba(255,255,255,0.14); border:1px solid rgba(255,255,255,0.32); '
                f'border-radius:16px; padding:12px 14px; margin-top:16px;">'
                f'<div style="font-size:1.65rem; font-weight:800; line-height:1.1; letter-spacing:-0.02em;">{card["value"]}</div>'
                f'<div style="font-size:0.76rem; opacity:0.92; margin-top:4px;">{card["sub"]}</div>'
                f'</div>'
                f'</div>'
            )
            st.markdown(card_html, unsafe_allow_html=True)

    st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # 2. CONTROLLED WEATHER ABLATION STUDY (WITHOUT VS. WITH OPEN-METEO)
    # --------------------------------------------------------------------------
    st.subheader("Controlled weather ablation study")

    ablation_subsets = [
        {
            "subset": "Full Test Set",
            "orders": "19,019 orders",
            "without": 0.2921,
            "with": 0.3132,
            "lift": "+7.22%",
        },
        {
            "subset": "Severe Weather Orders",
            "orders": "12,250 orders",
            "without": 0.3113,
            "with": 0.3344,
            "lift": "+7.42%",
        },
    ]

    ab_cols = st.columns(2, gap="medium")
    for col, item in zip(ab_cols, ablation_subsets):
        w_without = (item["without"] / 0.35) * 100.0
        w_with = (item["with"] / 0.35) * 100.0
        with col:
            ab_card_html = (
                f'<div style="background:#ffffff; border:1px solid #e5e7eb; border-radius:24px; '
                f'padding:24px 28px; box-shadow:0 4px 16px rgba(15,23,42,0.04);">'
                f'<div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:16px;">'
                f'<div>'
                f'<div style="font-size:1.08rem; font-weight:750; color:#111827;">{item["subset"]}</div>'
                f'<div style="font-size:0.82rem; color:#6b7280; margin-top:2px;">{item["orders"]} </div>'
                f'</div>'
                f'<span style="background:#dcfce7; color:#15803d; font-weight:750; font-size:0.84rem; '
                f'padding:5px 12px; border-radius:999px;">{item["lift"]} PR-AUC</span>'
                f'</div>'
                f'<div style="margin:12px 0;">'
                f'<div style="display:flex; justify-content:space-between; font-size:0.84rem; color:#4b5563; margin-bottom:5px;">'
                f'<span>Without Weather</span>'
                f'<span style="font-family:monospace; font-weight:600;">{item["without"]:.4f}</span>'
                f'</div>'
                f'<div style="background:#f1f5f9; height:11px; border-radius:999px; overflow:hidden;">'
                f'<div style="width:{w_without:.1f}%; height:100%; background:#94a3b8; border-radius:999px;"></div>'
                f'</div>'
                f'</div>'
                f'<div style="margin:14px 0 4px 0;">'
                f'<div style="display:flex; justify-content:space-between; font-size:0.84rem; color:#111827; font-weight:600; margin-bottom:5px;">'
                f'<span>With Open-Meteo Weather (All 86 features)</span>'
                f'<span style="font-family:monospace; font-weight:700; color:#16a34a;">{item["with"]:.4f}</span>'
                f'</div>'
                f'<div style="background:#f1f5f9; height:11px; border-radius:999px; overflow:hidden;">'
                f'<div style="width:{w_with:.1f}%; height:100%; background:linear-gradient(90deg, #16a34a 0%, #4ade80 100%); border-radius:999px;"></div>'
                f'</div>'
                f'</div>'
                f'</div>'
            )
            st.markdown(ab_card_html, unsafe_allow_html=True)

    st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # 3. SHARE OF MEAN |SHAP| BY DOMAIN (MATCHING SHAP EXPLANATIONS UI)
    # --------------------------------------------------------------------------
    st.subheader("Predictive share by operational domain")

    domain_items = [
        (" Weather (Open-Meteo)", 26.13),
        ("SLA Buffer & Temporal", 24.43),
        ("Spatial & Route Geography", 22.95),
        ("Seller & Category Reliability", 15.78),
        ("Package Dimensions & Freight", 10.72),
    ]

    max_share = max(val for _, val in domain_items)
    rows_html = ""
    for label, share in domain_items:
        pct = min(100.0, max(6.0, (share / max_share) * 100.0))
        rows_html += (
            f'<div style="display:flex; align-items:center; margin:15px 0; font-size:0.9rem;">'
            f'<div style="width:28%; text-align:right; padding-right:18px; color:#111827; font-weight:500;">{label}</div>'
            f'<div style="width:62%; background:#f1f5f9; height:12px; border-radius:999px; overflow:hidden;">'
            f'<div style="width:{pct:.1f}%; height:100%; border-radius:999px; '
            f'background:linear-gradient(90deg, #16a34a 0%, #4ade80 100%);"></div>'
            f'</div>'
            f'<div style="width:10%; text-align:right; color:#374151; font-weight:700; font-family:monospace;">'
            f'{share:.1f}%'
            f'</div>'
            f'</div>'
        )

    share_card_html = (
        f'<div style="background:#ffffff; border:1px solid #e5e7eb; border-radius:24px; '
        f'padding:28px 32px; margin-top:8px; box-shadow:0 4px 16px rgba(15,23,42,0.04);">'
        f'<div style="color:#6b7280; font-size:0.84rem; margin-bottom:18px;">'
        f'Share of total mean |SHAP| importance across 1,000 held-out test shipments'
        f'</div>'
        f'{rows_html}'
        f'</div>'
    )
    st.markdown(share_card_html, unsafe_allow_html=True)