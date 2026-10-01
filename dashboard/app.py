"""
Streamlit Web Application: Amazon India Executive BI Dashboard
Designed for Manoj to present to non-technical Manager Ravi (Sapphire IQ Case Study)
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure root directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.prediction import predict_order_return_risk
from src.forecasting import generate_future_forecast

# -----------------------------------------------------------------------------
# PAGE SETUP & MODERN STYLING (HIGH-CONTRAST FOR DARK & LIGHT MODES)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Amazon India Executive Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Contrast CSS for both Dark and Light themes
st.markdown("""
<style>
    /* Main Top Banner */
    .main-banner {
        background: linear-gradient(135deg, #131921 0%, #232F3E 100%);
        padding: 18px 24px;
        border-radius: 10px;
        border-left: 6px solid #FF9900;
        margin-bottom: 22px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    }
    .main-banner-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #FF9900 !important;
        margin-bottom: 6px;
        letter-spacing: -0.5px;
    }
    .main-banner-subtitle {
        font-size: 1.15rem;
        color: #FFFFFF !important;
        font-weight: 500;
        line-height: 1.5;
    }
    .main-banner-highlight {
        color: #FF9900 !important;
        font-weight: 800;
        font-size: 1.25rem;
    }

    /* Executive Insight Notice */
    .badge-insight {
        background: #1A232F !important;
        border-left: 5px solid #FF9900 !important;
        border: 1px solid #2E3B4E !important;
        padding: 14px 18px !important;
        border-radius: 8px !important;
        color: #FFFFFF !important;
        font-size: 0.95rem !important;
        margin-bottom: 18px !important;
        line-height: 1.5;
    }
    .badge-insight b, .badge-insight strong {
        color: #FF9900 !important;
    }

    /* Override Streamlit Metric Cards for 100% Visibility */
    div[data-testid="stMetric"] {
        background-color: #232F3E !important;
        border-radius: 10px !important;
        padding: 14px 16px !important;
        border-left: 5px solid #FF9900 !important;
        border: 1px solid #37475A !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.25) !important;
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricLabel"],
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] p {
        color: #FF9900 !important;
        font-size: 0.95rem !important;
        font-weight: 700 !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"],
    div[data-testid="stMetric"] [data-testid="stMetricValue"] div {
        color: #FFFFFF !important;
        font-size: 1.8rem !important;
        font-weight: 800 !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricDelta"],
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] div {
        color: #E2E8F0 !important;
        font-weight: 600 !important;
    }

    /* Custom KPI Card class - Equal length & breadth with no text overflow */
    .custom-kpi-card {
        background-color: #232F3E;
        border-radius: 10px;
        padding: 14px 16px;
        border-left: 5px solid #FF9900;
        border: 1px solid #3d4a5d;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        height: 125px;
        min-height: 125px;
        max-height: 125px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-sizing: border-box;
        overflow: hidden;
    }
    .custom-kpi-title {
        color: #FF9900;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.3px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .custom-kpi-val {
        color: #FFFFFF;
        font-size: 1.55rem;
        font-weight: 800;
        line-height: 1.2;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .custom-kpi-sub {
        color: #A0AEC0;
        font-size: 0.76rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# DATA LOADING (CACHED)
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    path = "data/processed/amazon_sales_cleaned.csv"
    if not os.path.exists(path):
        path = "Amazon Sales Data India.xlsx"
        df = pd.read_excel(path) if path.endswith(".xlsx") else pd.read_csv(path)
        df["Order_Date"] = pd.to_datetime(df["Order_Date"])
        df["Year_Month"] = df["Order_Date"].dt.strftime("%Y-%m")
        df["Lost_Sales_INR"] = np.where(df["Order_Status"].isin(["Returned", "Cancelled"]), df["Total_Sales_INR"], 0.0)
        df["Is_Returned"] = (df["Order_Status"] == "Returned").astype(int)
        df["Is_Cancelled"] = (df["Order_Status"] == "Cancelled").astype(int)
        return df

    df = pd.read_csv(path)
    df["Order_Date"] = pd.to_datetime(df["Order_Date"])
    return df


@st.cache_resource
def load_ml_models():
    f_path = "models/sales_forecasting_model.pkl"
    c_path = "models/return_prediction_model.pkl"
    f_art = joblib.load(f_path) if os.path.exists(f_path) else None
    c_art = joblib.load(c_path) if os.path.exists(c_path) else None
    return f_art, c_art


df = load_data()
forecast_art, clf_art = load_ml_models()

# Category image assets mapping
CAT_IMAGE_MAP = {
    "Electronics & Mobiles": "assets/cat_electronics.png",
    "Apparel & Fashion": "assets/cat_apparel.png",
    "Home & Kitchen": "assets/cat_home.png",
    "Beauty & Personal Care": "assets/cat_beauty.png",
    "Pantry & Groceries": "assets/cat_pantry.png",
}

def format_inr(val):
    """Format large currency numbers cleanly (e.g. ₹15.58 Cr) to prevent text overflow"""
    if not isinstance(val, (int, float, np.number)):
        return str(val)
    if val >= 10_000_000:
        return f"₹{val / 10_000_000:.2f} Cr"
    elif val >= 100_000:
        return f"₹{val / 100_000:.2f} L"
    else:
        return f"₹{val:,.0f}"

def render_kpi(icon, label, value, sub_text=""):
    """Render high-contrast, beautiful custom KPI card with exact uniform length and breadth"""
    st.markdown(f"""
    <div class="custom-kpi-card">
        <div class="custom-kpi-title">{icon} {label}</div>
        <div class="custom-kpi-val" title="{value}">{value}</div>
        <div class="custom-kpi-sub">{sub_text if sub_text else '&nbsp;'}</div>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# SIDEBAR FILTERS (LOCAL LOGO & HIGH-CONTRAST CONTROLS)
# -----------------------------------------------------------------------------
with st.sidebar:
    # 1. Reliable local Amazon logo
    logo_path = "assets/amazon_logo.png"
    if os.path.exists(logo_path):
        st.image(logo_path, width=220)
    else:
        st.markdown("<h2 style='color:#FF9900;'>amazon.in</h2>", unsafe_allow_html=True)

    st.markdown("### 🎛️ Quick Filters")
    st.caption("Filter data across all dashboard sections")

    # Category Filter
    categories = ["All Categories"] + sorted(df["Category"].dropna().unique().tolist())
    selected_cat = st.selectbox("Product Category", categories)

    # Show thumbnail card if category selected
    if selected_cat != "All Categories" and selected_cat in CAT_IMAGE_MAP:
        cat_thumb = CAT_IMAGE_MAP[selected_cat]
        if os.path.exists(cat_thumb):
            st.image(cat_thumb, use_container_width=True)

    # State Filter
    states = ["All States"] + sorted(df["Ship_State"].dropna().unique().tolist())
    selected_state = st.selectbox("Customer State", states)

    # Fulfillment Filter
    channels = ["All Channels"] + sorted(df["Fulfillment"].dropna().unique().tolist())
    selected_ful = st.selectbox("Fulfillment Mode", channels)

    st.markdown("---")
    st.caption("💼 **Audience:** Non-Technical Executive Management (Ravi)")
    st.caption("📌 **Objective:** Simple, KPI-driven business visibility")


# Apply Filters
filtered_df = df.copy()
if selected_cat != "All Categories":
    filtered_df = filtered_df[filtered_df["Category"] == selected_cat]
if selected_state != "All States":
    filtered_df = filtered_df[filtered_df["Ship_State"] == selected_state]
if selected_ful != "All Channels":
    filtered_df = filtered_df[filtered_df["Fulfillment"] == selected_ful]


# -----------------------------------------------------------------------------
# EXECUTIVE HEADER BANNER (ALL WHITE & AMAZON ORANGE COLORS)
# -----------------------------------------------------------------------------
st.markdown("""
<div class="main-banner">
    <div class="main-banner-title">
        Amazon India Management Dashboard
    </div>
    <div class="main-banner-subtitle">
        Prepared by <span class="main-banner-highlight">Manoj</span> for <span class="main-banner-highlight">Manager Ravi</span> &nbsp;|&nbsp; <span style="color: #FFFFFF; font-weight: 600;">Business Performance &amp; AI-Powered Intelligence System</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Top 5 Core Executive KPIs using High-Contrast Custom Cards
tot_sales = filtered_df["Total_Sales_INR"].sum()
tot_profit = filtered_df["Profit_INR"].sum()
tot_orders = filtered_df["Order_ID"].nunique()
tot_units = filtered_df["Quantity"].sum()
aov = tot_sales / max(tot_orders, 1)
margin_pct = (tot_profit / max(tot_sales, 1)) * 100

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
with kpi1:
    render_kpi("💰", "Total Sales", format_inr(tot_sales), f"{tot_units:,} units shipped")
with kpi2:
    render_kpi("📈", "Total Profit", format_inr(tot_profit), f"{margin_pct:.1f}% net margin")
with kpi3:
    render_kpi("🛒", "Total Orders", f"{tot_orders:,}", "Across 10 key states")
with kpi4:
    render_kpi("🏷️", "Avg Order Value", format_inr(aov), "Revenue per order")
with kpi5:
    render_kpi("📊", "Profit Margin", f"{margin_pct:.1f}%", "Healthy profitability")

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# SIMPLE 4-TAB NAVIGATION
# -----------------------------------------------------------------------------
tab_overview, tab_products, tab_health, tab_ai = st.tabs([
    "📊 1. Sales & Profit Trends",
    "📦 2. Categories & Products",
    "🚚 3. Order Status & States",
    "🤖 4. AI Forecast & Return Risk"
])


# =============================================================================
# TAB 1: OVERALL SALES & PROFIT TRENDS
# =============================================================================
with tab_overview:
    st.subheader("How is the business performing overall?")
    st.markdown('<div class="badge-insight">💡 <b>Executive Insight:</b> Sales and profitability demonstrate consistent growth with peak seasonal surges in festive periods. Overall profit margins maintain a healthy 21.2%.</div>', unsafe_allow_html=True)

    # Monthly aggregation
    monthly = (
        filtered_df.groupby("Year_Month")
        .agg(Sales=("Total_Sales_INR", "sum"), Profit=("Profit_INR", "sum"), Orders=("Order_ID", "count"))
        .reset_index()
    )

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Bar(
        x=monthly["Year_Month"], y=monthly["Sales"],
        name="Total Sales (INR)", marker_color="#FF9900"
    ))
    fig_trend.add_trace(go.Scatter(
        x=monthly["Year_Month"], y=monthly["Profit"],
        name="Net Profit (INR)", mode="lines+markers", line=dict(color="#107C41", width=3)
    ))
    fig_trend.update_layout(
        title="<b>Monthly Sales (Bars) vs Profit (Line) Trend</b>",
        xaxis_title="Month",
        yaxis_title="Amount (INR)",
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Monthly Sales Breakdown (Recent 6 Months)**")
        st.dataframe(
            monthly.tail(6)[["Year_Month", "Sales", "Profit", "Orders"]].style.format({
                "Sales": "₹{:,.2f}", "Profit": "₹{:,.2f}", "Orders": "{:,}"
            }),
            use_container_width=True
        )
    with c2:
        st.markdown("**Business Health Indicators**")
        delivered_count = (filtered_df["Order_Status"] == "Delivered").sum()
        delivery_rate = (delivered_count / max(len(filtered_df), 1)) * 100
        lost_rev = filtered_df["Lost_Sales_INR"].sum()

        st.write(f"- **Fulfillment Success Rate**: **{delivery_rate:.1f}%** ({delivered_count:,} orders successfully delivered)")
        st.write(f"- **Total Units Shipped**: **{tot_units:,} units** across all catalog segments")
        st.write(f"- **Lost Revenue from Returns/Cancellations**: **₹{lost_rev:,.2f}**")


# =============================================================================
# TAB 2: CATEGORY & PRODUCT PERFORMANCE WITH VISUAL CARDS
# =============================================================================
with tab_products:
    st.subheader("Which products and categories are driving business?")
    st.markdown('<div class="badge-insight">💡 <b>Executive Insight:</b> <b>Electronics & Mobiles</b> brings in 50% of top-line revenue, while <b>Apparel & Fashion</b> and <b>Home & Kitchen</b> produce the highest profit margins (>21.5%).</div>', unsafe_allow_html=True)

    # Visual Category Cards Showcase from Database
    st.markdown("### 🖼️ Catalog Categories Gallery")
    st.caption("Visual overview of categories present in Amazon India sales dataset")

    cat_cols = st.columns(5)
    catalog_meta = [
        ("Electronics & Mobiles", "cat_electronics.png", "📱 ₹7.84 Cr Sales"),
        ("Apparel & Fashion", "cat_apparel.png", "👕 ₹3.71 Cr Sales"),
        ("Home & Kitchen", "cat_home.png", "🍳 ₹2.98 Cr Sales"),
        ("Beauty & Personal Care", "cat_beauty.png", "💄 ₹65.8 Lakhs"),
        ("Pantry & Groceries", "cat_pantry.png", "🌾 ₹39.5 Lakhs")
    ]
    for idx, (cat_name, img_name, rev_text) in enumerate(catalog_meta):
        with cat_cols[idx]:
            img_p = os.path.join("assets", img_name)
            if os.path.exists(img_p):
                st.image(img_p, use_container_width=True)
            st.markdown(f"<div style='text-align:center; font-weight:700; color:#FF9900; margin-top:-6px;'>{rev_text}</div>", unsafe_allow_html=True)

    st.markdown("---")

    col_cat, col_prod = st.columns(2)

    cat_df = (
        filtered_df.groupby("Category")
        .agg(Sales=("Total_Sales_INR", "sum"), Profit=("Profit_INR", "sum"), Orders=("Order_ID", "count"))
        .reset_index()
        .sort_values("Sales", ascending=True)
    )
    cat_df["Margin_%"] = (cat_df["Profit"] / cat_df["Sales"]) * 100

    with col_cat:
        fig_cat = px.bar(
            cat_df, x="Sales", y="Category", orientation="h",
            title="<b>Sales by Category (INR)</b>", color="Sales",
            color_continuous_scale="Oranges"
        )
        fig_cat.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_cat, use_container_width=True)

    with col_prod:
        fig_margin = px.bar(
            cat_df, x="Margin_%", y="Category", orientation="h",
            title="<b>Profit Margin by Category (%)</b>", color="Margin_%",
            color_continuous_scale="Greens"
        )
        fig_margin.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_margin, use_container_width=True)

    # Top 10 Best Sellers
    st.markdown("### 🏆 Top 10 Best-Selling Products")
    prod_df = (
        filtered_df.groupby(["Product", "Category"])
        .agg(Sales=("Total_Sales_INR", "sum"), Profit=("Profit_INR", "sum"), Units=("Quantity", "sum"))
        .reset_index()
        .sort_values("Sales", ascending=False)
        .head(10)
    )
    prod_df["Profit_Margin_%"] = (prod_df["Profit"] / prod_df["Sales"]) * 100

    fig_top10 = px.bar(
        prod_df, x="Sales", y="Product", color="Category", orientation="h",
        title="<b>Top 10 Products by Total Revenue</b>"
    )
    fig_top10.update_layout(yaxis=dict(autorange="reversed"), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_top10, use_container_width=True)


# =============================================================================
# TAB 3: ORDER STATUS & GEOGRAPHY
# =============================================================================
with tab_health:
    st.subheader("Order Completion, Returns & Geographic Reach")

    # Status summary
    status_counts = filtered_df["Order_Status"].value_counts()
    ret_rate = (filtered_df["Order_Status"] == "Returned").mean() * 100
    canc_rate = (filtered_df["Order_Status"] == "Cancelled").mean() * 100

    col_st1, col_st2 = st.columns(2)

    with col_st1:
        fig_pie = px.pie(
            values=status_counts.values, names=status_counts.index, hole=0.45,
            title="<b>Order Status Breakdown</b>",
            color=status_counts.index,
            color_discrete_map={"Delivered": "#107C41", "Shipped": "#1f77b4", "Returned": "#D83B01", "Cancelled": "#797775"}
        )
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_st2:
        st.markdown("#### 🚨 Return & Cancellation Rates")
        r_c1, r_c2 = st.columns(2)
        with r_c1:
            render_kpi("↩️", "Return Rate", f"{ret_rate:.2f}%", f"{status_counts.get('Returned', 0):,} orders")
        with r_c2:
            render_kpi("❌", "Cancellation Rate", f"{canc_rate:.2f}%", f"{status_counts.get('Cancelled', 0):,} orders")
        lost_money = filtered_df["Lost_Sales_INR"].sum()
        st.markdown(f"""
        <div style="background-color: #3b2222; border-left: 5px solid #d62728; padding: 12px 16px; border-radius: 6px; color: #ffcccc; margin-top: 10px;">
            ⚠️ <b>Total Revenue Lost to Returns/Cancellations:</b> <span style="font-size: 1.2rem; font-weight:800; color: #FFFFFF;">₹{lost_money:,.2f}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🗺️ Top 10 States by Sales & Logistics Performance")

    state_df = (
        filtered_df.groupby("Ship_State")
        .agg(Sales=("Total_Sales_INR", "sum"), Profit=("Profit_INR", "sum"), Orders=("Order_ID", "count"), Returns=("Is_Returned", "sum"))
        .reset_index()
        .sort_values("Sales", ascending=False)
    )
    state_df["Return_Rate_%"] = (state_df["Returns"] / state_df["Orders"]) * 100

    fig_state = px.bar(
        state_df, x="Ship_State", y="Sales", color="Profit",
        title="<b>Sales and Profit by State</b>", color_continuous_scale="Blues"
    )
    fig_state.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_state, use_container_width=True)


# =============================================================================
# TAB 4: AI SALES FORECAST & RETURN RISK PREDICTION
# =============================================================================
with tab_ai:
    st.subheader("AI Machine Learning Solutions")
    st.markdown('<div class="badge-insight">🤖 <b>AI Integration:</b> This system utilizes two trained models: a <b>Random Forest Regressor</b> for future 30-day demand forecasting and an <b>XGBoost Classifier</b> for order return risk prediction.</div>', unsafe_allow_html=True)

    subtab_fc, subtab_risk = st.tabs(["🔮 30-Day Sales Forecast", "🛡️ Single-Order Return Risk Checker"])

    # 1. FORECASTING
    with subtab_fc:
        st.markdown("#### Predict Expected Future Daily Sales")
        if forecast_art is not None:
            model_name = forecast_art.get("model_name", "Random Forest")
            st.caption(f"Forecasting Model: **{model_name}** (Auto-selected as best model during training)")

            days_ahead = st.slider("Select Forecast Horizon (Days)", min_value=7, max_value=60, value=30, step=1)
            future_df = generate_future_forecast(forecast_art, horizon_days=days_ahead)
            proj_rev = future_df["Predicted_Sales_INR"].sum()
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                render_kpi("📅", f"Projected Sales", format_inr(proj_rev), f"{days_ahead}-day expected revenue")
            with f_col2:
                render_kpi("📊", "Avg Daily Sales", format_inr(future_df['Predicted_Sales_INR'].mean()), "Daily average expectation")

            # Plot
            hist_recent = df[df["Order_Status"] != "Cancelled"].groupby("Order_Date")["Total_Sales_INR"].sum().reset_index().tail(45)

            fig_forecast = go.Figure()
            fig_forecast.add_trace(go.Scatter(
                x=hist_recent["Order_Date"], y=hist_recent["Total_Sales_INR"],
                mode="lines", name="Recent Historical Sales", line=dict(color="#1f77b4", width=2)
            ))
            fig_forecast.add_trace(go.Scatter(
                x=future_df["Date"], y=future_df["Predicted_Sales_INR"],
                mode="lines+markers", name=f"AI Forecast ({model_name})", line=dict(color="#FF9900", width=3)
            ))
            fig_forecast.add_trace(go.Scatter(
                x=future_df["Date"], y=future_df["Upper_Bound_INR"],
                mode="lines", line=dict(width=0), showlegend=False
            ))
            fig_forecast.add_trace(go.Scatter(
                x=future_df["Date"], y=future_df["Lower_Bound_INR"],
                mode="lines", fill="tonexty", fillcolor="rgba(255, 153, 0, 0.15)",
                name="Confidence Range (±15%)", line=dict(width=0)
            ))
            fig_forecast.update_layout(
                title=f"<b>Next {days_ahead} Days AI Sales Forecast</b>",
                xaxis_title="Date",
                yaxis_title="Sales (INR)",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_forecast, use_container_width=True)
        else:
            st.warning("Forecasting model not found. Run model training script to generate artifact.")

    # 2. RETURN RISK CHECKER
    with subtab_risk:
        st.markdown("#### Estimate Return Risk for an Incoming Order")
        st.caption("Check if an incoming order is likely to be returned to take proactive action.")

        if clf_art is not None:
            r1, r2, r3 = st.columns(3)
            with r1:
                sel_cat = st.selectbox("Category", clf_art["categories"], key="r_cat")
                # Show visual thumbnail for selected category
                if sel_cat in CAT_IMAGE_MAP and os.path.exists(CAT_IMAGE_MAP[sel_cat]):
                    st.image(CAT_IMAGE_MAP[sel_cat], use_container_width=True)
                # Filter products for selected category
                prod_opts = df[df["Category"] == sel_cat]["Product"].unique().tolist() or clf_art["products"]
                sel_prod = st.selectbox("Product", prod_opts, key="r_prod")
                sel_state = st.selectbox("Destination State", clf_art["states"], key="r_state")
            with r2:
                sel_qty = st.number_input("Quantity", min_value=1, max_value=5, value=2, step=1, key="r_qty")
                sel_price = st.number_input("Unit Price (₹)", min_value=100.0, max_value=40000.0, value=2500.0, step=250.0, key="r_price")
                sel_disc = st.slider("Discount (%)", min_value=0.0, max_value=0.3, value=0.10, step=0.05, key="r_disc")
            with r3:
                sel_pay = st.selectbox("Payment Method", clf_art["payment_methods"], key="r_pay")
                sel_ful = st.selectbox("Fulfillment Mode", clf_art["fulfillments"], key="r_ful")
                order_val = sel_qty * sel_price * (1.0 - sel_disc)
                render_kpi("💵", "Estimated Order", format_inr(order_val), f"Exact: ₹{order_val:,.2f}")

            if st.button("🚀 Check Return Probability", type="primary", use_container_width=True):
                order_input = {
                    "Category": sel_cat,
                    "Product": sel_prod,
                    "Payment_Method": sel_pay,
                    "Fulfillment": sel_ful,
                    "Ship_State": sel_state,
                    "Quantity": sel_qty,
                    "Unit_Price_INR": sel_price,
                    "Discount_Pct": sel_disc,
                    "Total_Sales_INR": order_val
                }
                res = predict_order_return_risk(order_input, clf_art)

                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    render_kpi("🎯", "Return Probability", f"{res['return_probability']}%", "Calculated by XGBoost")
                    st.progress(res['return_probability'] / 100.0)
                with col_res2:
                    if res["risk_level"] == "High Risk":
                        st.markdown(f"""
                        <div style="background-color: #3b2222; border-left: 5px solid #d62728; padding: 16px; border-radius: 8px; color: #ffcccc; margin-top: 10px;">
                            <span style="font-size: 1.3rem; font-weight:800; color: #ff4d4d;">🔴 {res['risk_level']}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    elif res["risk_level"] == "Medium Risk":
                        st.markdown(f"""
                        <div style="background-color: #3a321d; border-left: 5px solid #ff9900; padding: 16px; border-radius: 8px; color: #ffe6aa; margin-top: 10px;">
                            <span style="font-size: 1.3rem; font-weight:800; color: #ffaa00;">🟡 {res['risk_level']}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background-color: #1b3322; border-left: 5px solid #2ca02c; padding: 16px; border-radius: 8px; color: #ccffdd; margin-top: 10px;">
                            <span style="font-size: 1.3rem; font-weight:800; color: #00ff66;">🟢 {res['risk_level']}</span>
                        </div>
                        """, unsafe_allow_html=True)

                st.info(f"📋 **Actionable Advice for Ravi:** {res['recommendation']}")
        else:
            st.warning("Classifier model not found. Run model training script to generate artifact.")
