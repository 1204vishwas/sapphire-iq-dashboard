"""
Feature Engineering Module
Amazon India Sales Analytics & AI-Powered Business Intelligence System
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def prepare_forecasting_data(
    df: pd.DataFrame,
    date_col: str = "Order_Date",
    target_col: str = "Total_Sales_INR",
    exclude_cancelled: bool = True
) -> pd.DataFrame:
    """
    Aggregate sales to daily level and generate rich time-series lag and rolling window features.
    """
    df_filtered = df.copy()
    if exclude_cancelled and "Order_Status" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["Order_Status"] != "Cancelled"]

    # Daily aggregation
    df_filtered[date_col] = pd.to_datetime(df_filtered[date_col])
    daily = (
        df_filtered.groupby(date_col)[target_col]
        .sum()
        .asfreq("D", fill_value=0.0)
        .reset_index()
    )

    # 1. Autoregressive Lags
    for lag in [1, 2, 3, 7, 14, 21, 30]:
        daily[f"lag_{lag}"] = daily[target_col].shift(lag)

    # 2. Rolling Window Statistics
    for window in [7, 14, 30]:
        daily[f"rolling_mean_{window}"] = daily[target_col].shift(1).rolling(window).mean()
        daily[f"rolling_std_{window}"] = daily[target_col].shift(1).rolling(window).std()

    # 3. Calendar & Seasonal Indicators
    daily["day_of_week"] = daily[date_col].dt.dayofweek
    daily["day_of_month"] = daily[date_col].dt.day
    daily["month"] = daily[date_col].dt.month
    daily["quarter"] = daily[date_col].dt.quarter
    daily["is_weekend"] = daily["day_of_week"].isin([5, 6]).astype(int)
    daily["is_month_start"] = daily[date_col].dt.is_month_start.astype(int)
    daily["is_month_end"] = daily[date_col].dt.is_month_end.astype(int)

    # Drop warm-up rows where 30-day lag/rolling windows are NaN
    daily_featured = daily.dropna().reset_index(drop=True)
    return daily_featured


def get_forecasting_features() -> List[str]:
    """
    List of feature names used for time-series sales forecasting.
    """
    return [
        "lag_1", "lag_2", "lag_3", "lag_7", "lag_14", "lag_21", "lag_30",
        "rolling_mean_7", "rolling_std_7",
        "rolling_mean_14", "rolling_std_14",
        "rolling_mean_30", "rolling_std_30",
        "day_of_week", "day_of_month", "month", "quarter",
        "is_weekend", "is_month_start", "is_month_end"
    ]


def prepare_classification_data(
    df: pd.DataFrame,
    target_col: str = "Is_Returned",
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """
    Prepare order-level features for return risk classification.
    One-hot encodes categorical columns and performs stratified split.
    """
    data = df.copy()
    if "Order_Status" in data.columns and target_col not in data.columns:
        data[target_col] = (data["Order_Status"] == "Returned").astype(int)

    # Base feature columns
    cat_cols = ["Category", "Product", "Payment_Method", "Fulfillment", "Ship_State"]
    num_cols = ["Quantity", "Unit_Price_INR", "Discount_Pct", "Total_Sales_INR"]

    # Filter to available columns
    cat_cols = [c for c in cat_cols if c in data.columns]
    num_cols = [c for c in num_cols if c in data.columns]

    X_raw = data[cat_cols + num_cols].copy()
    y = data[target_col].copy()

    # One-hot encode categoricals
    X_encoded = pd.get_dummies(X_raw, columns=cat_cols, drop_first=True)
    feature_names = list(X_encoded.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, feature_names
