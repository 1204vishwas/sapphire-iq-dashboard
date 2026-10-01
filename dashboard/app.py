"""
Streamlit Web Application: Amazon India Business Intelligence & AI System
Executive Dashboard designed for Manoj & Manager Ravi
"""

import os
import sys
import sqlite3
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.prediction import predict_order_return_risk
from src.forecasting import generate_future_forecast

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & CUSTOM CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Amazon India AI Business Intelligence System",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #FF9900;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #232F3E;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #F8F9FA;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #FF9900;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #111;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #555;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# DATA LOADING & CACHING
# -----------------------------------------------------------------------------
@st.cache_data
def load_sales_data():
    file_path = "data/processed/amazon_sales_cleaned.csv"
    if not os.path.exists(file_path):
        file_path = "Amazon Sales Data India.xlsx"
        if file_path.endswith(".xlsx"):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)
        df["Order_Date"] = pd.to_datetime(df["Order_Date"])
        df["Year_Month"] = df["Order_Date"].dt.strftime("%Y-%m")
        df["Profit_Margin_Pct"] = np.where(df["Total_Sales_INR"] > 0, (df["Profit_INR"] / df["Total_Sales_INR"]) * 100, 0.0)
        df["Lost_Sales_INR"] = np.where(df["Order_Status"].isin(["Returned", "Cancelled"]), df["Total_Sales_INR"], 0.0)
        df["Is_Returned"] = (df["Order_Status"] == "Returned").astype(int)
        df["Is_Cancelled"] = (df["Order_Status"] == "Cancelled").astype(int)
        return df

    df = pd.read_csv(file_path)
    df["Order_Date"] = pd.to_datetime(df["Order_Date"])
    return df

@st.cache_resource
def load_models():
    forecast_model_path = "models/sales_forecasting_model.pkl"
    classifier_model_path = "models/return_prediction_model.pkl"

    forecast_art = joblib.load(forecast_model_path) if os.path.exists(forecast_model_path) else None
    classifier_art = joblib.load(classifier_model_path) if os.path.exists(classifier_model_path) else None
    return forecast_art, classifier_art

df = load_sales_data()
forecast_artifact, classifier_artifact = load_models()

# -----------------------------------------------------------------------------
# SIDEBAR FILTERS (Non-technical Manager Friendly)
# -----------------------------------------------------------------------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg", width=140)
st.sidebar.markdown("### 📊 Executive Filters")

# Date Filter
min_date = df["Order_Date"].min().date()
max_date = df["Order_Date"].max().date()
date_range = st.sidebar.date_input("Select Date Range", (min_date, max_date), min_value=min_date, max_value=max_date)

# Category Filter
all_categories = ["All Categories"] + sorted(df["Category"].dropna().unique().tolist())
selected_category = st.sidebar.selectbox("Category", all_categories)

# Fulfillment Filter
all_fulfillments = ["All Channels"] + sorted(df["Fulfillment"].dropna().unique().tolist())
selected_fulfillment = st.sidebar.selectbox("Fulfillment Channel", all_fulfillments)

# State Filter
all_states = ["All States"] + sorted(df["Ship_State"].dropna().unique().tolist())
selected_state = st.sidebar.selectbox("State / Region", all_states)

# Apply Filters
filtered_df = df.copy()
if len(date_range) == 2:
    start_d, end_d = date_range
    filtered_df = filtered_df[(filtered_df["Order_Date"].dt.date >= start_d) & (filtered_df["Order_Date"].dt.date <= end_d)]

if selected_category != "All Categories":
    filtered_df = filtered_df[filtered_df["Category"] == selected_category]

if selected_fulfillment != "All Channels":
    filtered_df = filtered_df[filtered_df["Fulfillment"] == selected_fulfillment]

if selected_state != "All States":
    filtered_df = filtered_df[filtered_df["Ship_State"] == selected_state]

st.sidebar.markdown("---")
st.sidebar.info("💡 **Designed for Executive Review**: Monitor sales trends, product profitability, order health, and predictive AI insights.")

# -----------------------------------------------------------------------------
# MAIN APP HEADER
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">Amazon India Sales Analytics & AI BI System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Executive Management Dashboard | Sapphire IQ Business Requirements</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📈 1. Overall Performance",
    "📦 2. Category & Products",
    "🔄 3. Order Status & Losses",
    "🚚 4. Payments & Geography",
    "🔮 5. AI Sales Forecast",
    "🛡️ 6. Return Risk Predictor",
    "💾 7. SQL Analytics Explorer"
])

# -----------------------------------------------------------------------------
# TAB 1: OVERALL SALES & PROFIT PERFORMANCE
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("Executive KPI Overview (Question 1: Overall Performance)")

    tot_sales = filtered_df["Total_Sales_INR"].sum()
    tot_profit = filtered_df["Profit_INR"].sum()
    tot_orders = filtered_df["Order_ID"].nunique()
    tot_units = filtered_df["Quantity"].sum()
    aov = tot_sales / max(tot_orders, 1)
    profit_margin = (tot_profit / max(tot_sales, 1)) * 100
    lost_rev = filtered_df["Lost_Sales_INR"].sum()

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Sales", f"₹{tot_sales:,.0f}")
    with col2:
        st.metric("Total Profit", f"₹{tot_profit:,.0f}")
    with col3:
        st.metric("Total Orders", f"{tot_orders:,}")
    with col4:
        st.metric("Average Order Value (AOV)", f"₹{aov:,.0f}")
    with col5:
        st.metric("Profit Margin %", f"{profit_margin:.2f}%")

    st.markdown("---")

    # Monthly Trend
    monthly_data = (
        filtered_df.groupby("Year_Month")
        .agg(Monthly_Sales=("Total_Sales_INR", "sum"), Monthly_Profit=("Profit_INR", "sum"), Orders=("Order_ID", "count"))
        .reset_index()
    )

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=monthly_data["Year_Month"], y=monthly_data["Monthly_Sales"],
        mode="lines+markers", name="Total Sales (INR)", line=dict(color="#FF9900", width=3)
    ))
    fig_trend.add_trace(go.Bar(
        x=monthly_data["Year_Month"], y=monthly_data["Monthly_Profit"],
        name="Net Profit (INR)", marker_color="#2ca02c", opacity=0.6
    ))
    fig_trend.update_layout(
        title="<b>Monthly Sales & Profit Trend (2024 - 2026)</b>",
        xaxis_title="Month",
        yaxis_title="Amount (INR)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # Executive Insight Note
    st.success("""
    📌 **Key Management Insight (for Ravi)**: 
    Overall monthly performance has remained consistently strong, averaging over ₹45-50 Lakhs in monthly gross revenue. 
    Net profit margins hold healthy around 21.2%, with seasonal peaks in festive quarters (Q3-Q4).
    """)

# -----------------------------------------------------------------------------
# TAB 2: CATEGORY & PRODUCT PERFORMANCE
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Category & Product Performance (Question 2)")

    c_col1, c_col2 = st.columns(2)

    cat_summary = (
        filtered_df.groupby("Category")
        .agg(Total_Sales=("Total_Sales_INR", "sum"), Total_Profit=("Profit_INR", "sum"), Units=("Quantity", "sum"))
        .reset_index()
    )
    cat_summary["Margin_%"] = (cat_summary["Total_Profit"] / cat_summary["Total_Sales"]) * 100
    cat_summary = cat_summary.sort_values("Total_Sales", ascending=False)

    with c_col1:
        fig_cat_sales = px.bar(
            cat_summary, x="Total_Sales", y="Category", orientation="h",
            title="<b>Category-wise Sales (INR)</b>", color="Total_Sales", color_continuous_scale="Blues"
        )
        st.plotly_chart(fig_cat_sales, use_container_width=True)

    with c_col2:
        fig_cat_margin = px.bar(
            cat_summary, x="Margin_%", y="Category", orientation="h",
            title="<b>Category-wise Profit Margin (%)</b>", color="Margin_%", color_continuous_scale="Greens"
        )
        st.plotly_chart(fig_cat_margin, use_container_width=True)

    # Top 10 Products by Sales
    st.markdown("### 🏆 Top 10 Best-Selling Products by Revenue")
    prod_summary = (
        filtered_df.groupby(["Product", "Category"])
        .agg(Sales=("Total_Sales_INR", "sum"), Profit=("Profit_INR", "sum"), Units_Sold=("Quantity", "sum"))
        .reset_index()
        .sort_values("Sales", ascending=False)
        .head(10)
    )
    prod_summary["Profit_Margin_%"] = (prod_summary["Profit"] / prod_summary["Sales"]) * 100

    fig_prod = px.bar(
        prod_summary, x="Sales", y="Product", color="Category", orientation="h",
        title="<b>Top 10 Products by Sales Revenue</b>"
    )
    fig_prod.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_prod, use_container_width=True)

    st.dataframe(prod_summary.style.format({
        "Sales": "₹{:,.2f}", "Profit": "₹{:,.2f}", "Profit_Margin_%": "{:.2f}%", "Units_Sold": "{:,}"
    }), use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 3: ORDER STATUS & REVENUE LOSS
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("Order Status & Revenue Loss Analysis (Question 3)")

    status_counts = filtered_df["Order_Status"].value_counts()
    ret_rate = (filtered_df["Order_Status"] == "Returned").mean() * 100
    canc_rate = (filtered_df["Order_Status"] == "Cancelled").mean() * 100

    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        st.metric("Delivered Orders", f"{status_counts.get('Delivered', 0):,}")
    with sc2:
        st.metric("Shipped Orders", f"{status_counts.get('Shipped', 0):,}")
    with sc3:
        st.metric("Return Rate %", f"{ret_rate:.2f}%", delta=f"{status_counts.get('Returned', 0):,} orders", delta_color="inverse")
    with sc4:
        st.metric("Cancellation Rate %", f"{canc_rate:.2f}%", delta=f"{status_counts.get('Cancelled', 0):,} orders", delta_color="inverse")

    st.markdown("---")
    oc1, oc2 = st.columns(2)

    with oc1:
        fig_pie = px.pie(
            values=status_counts.values, names=status_counts.index, hole=0.45,
            title="<b>Overall Order Fulfillment Status Breakdown</b>",
            color=status_counts.index,
            color_discrete_map={"Delivered": "#2ca02c", "Shipped": "#1f77b4", "Returned": "#d62728", "Cancelled": "#ff7f0e"}
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with oc2:
        cat_ret_canc = (
            filtered_df.groupby("Category")
            .agg(Total=("Order_ID", "count"), Returns=("Is_Returned", "sum"), Cancellations=("Is_Cancelled", "sum"), Lost_Rev=("Lost_Sales_INR", "sum"))
            .reset_index()
        )
        cat_ret_canc["Return_Rate_%"] = (cat_ret_canc["Returns"] / cat_ret_canc["Total"]) * 100
        cat_ret_canc["Cancel_Rate_%"] = (cat_ret_canc["Cancellations"] / cat_ret_canc["Total"]) * 100

        fig_loss = go.Figure()
        fig_loss.add_trace(go.Bar(x=cat_ret_canc["Category"], y=cat_ret_canc["Return_Rate_%"], name="Return Rate %", marker_color="#d62728"))
        fig_loss.add_trace(go.Bar(x=cat_ret_canc["Category"], y=cat_ret_canc["Cancel_Rate_%"], name="Cancel Rate %", marker_color="#ff7f0e"))
        fig_loss.update_layout(title="<b>Category-wise Return & Cancellation Rate (%)</b>", barmode="group", yaxis_title="Percentage (%)")
        st.plotly_chart(fig_loss, use_container_width=True)

    st.warning(f"⚠️ **Total Revenue Lost**: ₹{filtered_df['Lost_Sales_INR'].sum():,.2f} has been lost to returned and cancelled orders. Electronics & Mobiles shows the highest financial return exposure due to high average ticket size.")

# -----------------------------------------------------------------------------
# TAB 4: PAYMENT, FULFILLMENT & GEOGRAPHY
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("Payment, Fulfillment & Geographic Performance (Question 4)")

    g_col1, g_col2 = st.columns(2)

    with g_col1:
        pay_df = (
            filtered_df.groupby("Payment_Method")
            .agg(Sales=("Total_Sales_INR", "sum"), Profit=("Profit_INR", "sum"), Orders=("Order_ID", "count"))
            .reset_index()
            .sort_values("Sales", ascending=False)
        )
        fig_pay = px.bar(pay_df, x="Sales", y="Payment_Method", orientation="h", color="Profit",
                         title="<b>Sales & Profit by Payment Channel</b>", color_continuous_scale="Viridis")
        st.plotly_chart(fig_pay, use_container_width=True)

    with g_col2:
        ful_df = (
            filtered_df.groupby("Fulfillment")
            .agg(Sales=("Total_Sales_INR", "sum"), Profit=("Profit_INR", "sum"), Orders=("Order_ID", "count"))
            .reset_index()
            .sort_values("Sales", ascending=False)
        )
        fig_ful = px.pie(ful_df, values="Sales", names="Fulfillment", hole=0.4,
                         title="<b>Sales Share by Fulfillment Channel</b>", color_discrete_sequence=px.colors.qualitative.Bold)
        st.plotly_chart(fig_ful, use_container_width=True)

    st.markdown("### 🗺️ State-wise Performance Breakdown")
    state_df = (
        filtered_df.groupby("Ship_State")
        .agg(Sales=("Total_Sales_INR", "sum"), Profit=("Profit_INR", "sum"), Orders=("Order_ID", "count"), Returns=("Is_Returned", "sum"))
        .reset_index()
    )
    state_df["AOV"] = state_df["Sales"] / state_df["Orders"]
    state_df["Margin_%"] = (state_df["Profit"] / state_df["Sales"]) * 100
    state_df["Return_Rate_%"] = (state_df["Returns"] / state_df["Orders"]) * 100
    state_df = state_df.sort_values("Sales", ascending=False)

    fig_state = px.bar(
        state_df, x="Ship_State", y="Sales", color="Profit",
        title="<b>State-wise Sales Revenue (INR) & Profit Contribution</b>",
        color_continuous_scale="Teal"
    )
    st.plotly_chart(fig_state, use_container_width=True)

    st.dataframe(state_df.style.format({
        "Sales": "₹{:,.2f}", "Profit": "₹{:,.2f}", "AOV": "₹{:,.2f}", "Margin_%": "{:.2f}%", "Return_Rate_%": "{:.2f}%"
    }), use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 5: AI SALES FORECAST (NEXT 30 DAYS)
# -----------------------------------------------------------------------------
with tab5:
    st.subheader("🔮 AI Sales Forecasting (Next 30 Days)")
    st.markdown("Machine learning model trained on historical daily sales series to project future demand.")

    if forecast_artifact is not None:
        best_model_name = forecast_artifact.get("model_name", "Random Forest")
        st.info(f"🏆 **Active Model:** {best_model_name} (Selected as best performing model across candidate benchmarks).")

        horizon = st.slider("Select Forecast Horizon (Days)", min_value=7, max_value=60, value=30, step=1)
        forecast_df = generate_future_forecast(forecast_artifact, horizon_days=horizon)

        proj_total = forecast_df["Predicted_Sales_INR"].sum()
        proj_daily_avg = forecast_df["Predicted_Sales_INR"].mean()

        m1, m2 = st.columns(2)
        with m1:
            st.metric(f"Projected {horizon}-Day Revenue", f"₹{proj_total:,.2f}")
        with m2:
            st.metric("Expected Daily Average", f"₹{proj_daily_avg:,.2f}")

        # Plot Historical + Forecast
        recent_sales = df[df["Order_Status"] != "Cancelled"].groupby("Order_Date")["Total_Sales_INR"].sum().reset_index().tail(60)

        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(
            x=recent_sales["Order_Date"], y=recent_sales["Total_Sales_INR"],
            mode="lines", name="Historical Daily Sales (Recent 60 Days)", line=dict(color="#1f77b4", width=2)
        ))
        fig_fc.add_trace(go.Scatter(
            x=forecast_df["Date"], y=forecast_df["Predicted_Sales_INR"],
            mode="lines+markers", name=f"AI Forecast ({best_model_name})", line=dict(color="#d62728", width=3)
        ))
        fig_fc.add_trace(go.Scatter(
            x=forecast_df["Date"], y=forecast_df["Upper_Bound_INR"],
            mode="lines", line=dict(width=0), showlegend=False
        ))
        fig_fc.add_trace(go.Scatter(
            x=forecast_df["Date"], y=forecast_df["Lower_Bound_INR"],
            mode="lines", fill="tonexty", fillcolor="rgba(214, 39, 40, 0.15)",
            name="Confidence Interval (±15%)", line=dict(width=0)
        ))
        fig_fc.update_layout(
            title=f"<b>Historical Daily Sales & {horizon}-Day Predictive AI Projection</b>",
            xaxis_title="Date",
            yaxis_title="Daily Revenue (INR)",
            hovermode="x unified"
        )
        st.plotly_chart(fig_fc, use_container_width=True)

        st.dataframe(forecast_df.style.format({
            "Predicted_Sales_INR": "₹{:,.2f}",
            "Lower_Bound_INR": "₹{:,.2f}",
            "Upper_Bound_INR": "₹{:,.2f}"
        }), use_container_width=True)
    else:
        st.error("Forecasting model artifact not found. Please run the model training script.")

# -----------------------------------------------------------------------------
# TAB 6: AI RETURN RISK PREDICTOR
# -----------------------------------------------------------------------------
with tab6:
    st.subheader("🛡️ AI Return Risk Prediction System")
    st.markdown("Assess incoming order parameters to compute likelihood of return and prevent revenue loss.")

    if classifier_artifact is not None:
        best_clf_name = classifier_artifact.get("model_name", "XGBoost Classifier")
        st.info(f"🏆 **Active Classification Engine:** {best_clf_name} (Top ROC-AUC & F1 score in candidate benchmarking).")

        c1, c2, c3 = st.columns(3)
        with c1:
            in_cat = st.selectbox("Product Category", classifier_artifact["categories"])
            in_prod = st.selectbox("Product", [p for p in classifier_artifact["products"] if p in df[df["Category"] == in_cat]["Product"].unique()] or classifier_artifact["products"])
            in_state = st.selectbox("Shipping Destination State", classifier_artifact["states"])
        with c2:
            in_qty = st.number_input("Quantity", min_value=1, max_value=10, value=2, step=1)
            in_price = st.number_input("Unit Price (INR)", min_value=100.0, max_value=50000.0, value=12000.0, step=500.0)
            in_disc = st.slider("Discount Applied (%)", min_value=0.0, max_value=0.5, value=0.10, step=0.05)
        with c3:
            in_pay = st.selectbox("Payment Method", classifier_artifact["payment_methods"])
            in_ful = st.selectbox("Fulfillment Channel", classifier_artifact["fulfillments"])
            in_tot_sales = in_qty * in_price * (1.0 - in_disc)
            st.metric("Estimated Order Sales", f"₹{in_tot_sales:,.2f}")

        predict_btn = st.button("🔍 Predict Return Risk", use_container_width=True, type="primary")

        if predict_btn:
            order_dict = {
                "Category": in_cat,
                "Product": in_prod,
                "Payment_Method": in_pay,
                "Fulfillment": in_ful,
                "Ship_State": in_state,
                "Quantity": in_qty,
                "Unit_Price_INR": in_price,
                "Discount_Pct": in_disc,
                "Total_Sales_INR": in_tot_sales
            }
            res = predict_order_return_risk(order_dict, classifier_artifact)

            r_col1, r_col2 = st.columns(2)
            with r_col1:
                st.markdown(f"### Predicted Return Probability: **{res['return_probability']}%**")
                st.progress(res['return_probability'] / 100.0)
            with r_col2:
                if res['risk_level'] == "High Risk":
                    st.error(f"⚠️ **Risk Level:** {res['risk_level']}")
                elif res['risk_level'] == "Medium Risk":
                    st.warning(f"⚡ **Risk Level:** {res['risk_level']}")
                else:
                    st.success(f"✅ **Risk Level:** {res['risk_level']}")

            st.markdown(f"**Operational Recommendation:** {res['recommendation']}")

        # Top feature importance display
        st.markdown("---")
        st.markdown("### 🔑 Top Features Driving Return Probability")
        feat_df = pd.DataFrame(list(classifier_artifact["feature_importances"].items()), columns=["Feature", "Importance"])
        fig_feat = px.bar(feat_df.head(10), x="Importance", y="Feature", orientation="h", title="Top 10 Feature Importances")
        fig_feat.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_feat, use_container_width=True)
    else:
        st.error("Classifier model artifact not found. Please run the model training script.")

# -----------------------------------------------------------------------------
# TAB 7: SQL ANALYTICS EXPLORER
# -----------------------------------------------------------------------------
with tab7:
    st.subheader("💾 SQL Business Analytics Explorer")
    st.markdown("Run verified production SQL queries directly against the Amazon Sales database.")

    db_path = "database/amazon_sales.db"
    if os.path.exists(db_path):
        query_options = {
            "1. Executive KPI Summary": """SELECT 
    COUNT(DISTINCT Order_ID) AS total_orders,
    SUM(Quantity) AS total_units_sold,
    ROUND(SUM(Total_Sales_INR), 2) AS total_sales_inr,
    ROUND(SUM(Profit_INR), 2) AS total_profit_inr,
    ROUND(SUM(Total_Sales_INR) / COUNT(DISTINCT Order_ID), 2) AS aov_inr,
    ROUND((SUM(Profit_INR) / SUM(Total_Sales_INR)) * 100, 2) AS profit_margin_pct
FROM amazon_sales;""",
            "2. Category Performance & Window Ranking": """SELECT 
    Category,
    COUNT(DISTINCT Order_ID) AS total_orders,
    SUM(Quantity) AS units_sold,
    ROUND(SUM(Total_Sales_INR), 2) AS category_sales_inr,
    ROUND(SUM(Profit_INR), 2) AS category_profit_inr,
    DENSE_RANK() OVER (ORDER BY SUM(Total_Sales_INR) DESC) AS sales_rank
FROM amazon_sales
GROUP BY Category
ORDER BY category_sales_inr DESC;""",
            "3. Top 10 Best-Selling Products": """SELECT 
    Product, Category,
    SUM(Quantity) AS units_sold,
    ROUND(SUM(Total_Sales_INR), 2) AS total_sales_inr,
    ROUND(SUM(Profit_INR), 2) AS total_profit_inr
FROM amazon_sales
GROUP BY Product, Category
ORDER BY total_sales_inr DESC
LIMIT 10;""",
            "4. Return & Cancellation Rate Breakdown": """SELECT 
    Category,
    COUNT(*) AS total_orders,
    SUM(CASE WHEN Order_Status = 'Returned' THEN 1 ELSE 0 END) AS returned_orders,
    SUM(CASE WHEN Order_Status = 'Cancelled' THEN 1 ELSE 0 END) AS cancelled_orders,
    ROUND((SUM(CASE WHEN Order_Status = 'Returned' THEN 1.0 ELSE 0 END) / COUNT(*)) * 100, 2) AS return_rate_pct,
    ROUND((SUM(CASE WHEN Order_Status = 'Cancelled' THEN 1.0 ELSE 0 END) / COUNT(*)) * 100, 2) AS cancel_rate_pct,
    ROUND(SUM(CASE WHEN Order_Status IN ('Returned', 'Cancelled') THEN Total_Sales_INR ELSE 0 END), 2) AS revenue_lost_inr
FROM amazon_sales
GROUP BY Category
ORDER BY return_rate_pct DESC;""",
            "5. Payment Method & Return Rate Analysis": """SELECT 
    Payment_Method,
    COUNT(DISTINCT Order_ID) AS total_orders,
    ROUND(SUM(Total_Sales_INR), 2) AS total_sales_inr,
    ROUND((SUM(CASE WHEN Order_Status = 'Returned' THEN 1.0 ELSE 0 END) / COUNT(*)) * 100, 2) AS return_rate_pct
FROM amazon_sales
GROUP BY Payment_Method
ORDER BY total_sales_inr DESC;"""
        }

        selected_q_name = st.selectbox("Choose a Pre-Engineered Business Query:", list(query_options.keys()))
        sql_text = st.text_area("SQL Query Editor", query_options[selected_q_name], height=180)

        if st.button("▶️ Execute Query", type="primary"):
            try:
                conn = sqlite3.connect(db_path)
                result_df = pd.read_sql_query(sql_text, conn)
                conn.close()

                st.success(f"Query returned {len(result_df)} rows.")
                st.dataframe(result_df, use_container_width=True)

                csv_data = result_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Query Result as CSV", csv_data, "query_results.csv", "text/csv")
            except Exception as e:
                st.error(f"SQL Execution Error: {e}")
    else:
        st.error(f"Database file not found at {db_path}.")
