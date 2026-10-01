"""
Sales Forecasting Module
Amazon India Sales Analytics & AI-Powered Business Intelligence System
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.feature_engineering import prepare_forecasting_data, get_forecasting_features


def compare_forecasting_models(
    df_featured: pd.DataFrame,
    features: List[str] = None,
    target_col: str = "Total_Sales_INR",
    test_split_ratio: float = 0.2
) -> Tuple[Dict[str, Any], pd.DataFrame, Any]:
    """
    Train and compare multiple forecasting models:
    - Linear Regression (Baseline)
    - Ridge Regression
    - Random Forest Regressor
    - XGBoost Regressor

    Returns evaluation metrics, leaderboard dataframe, and the best model.
    """
    if features is None:
        features = get_forecasting_features()

    # Time-based chronological split
    train_size = int(len(df_featured) * (1 - test_split_ratio))
    train = df_featured.iloc[:train_size]
    test = df_featured.iloc[train_size:]

    X_train, y_train = train[features], train[target_col]
    X_test, y_test = test[features], test[target_col]

    candidate_models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=10.0),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_split=4, random_state=42),
        "XGBoost Regressor": XGBRegressor(n_estimators=80, max_depth=4, learning_rate=0.05, random_state=42)
    }

    results = []
    trained_models = {}

    for name, model in candidate_models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        trained_models[name] = model

        # Metrics
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))
        mape = float(np.mean(np.abs((y_test - preds) / np.maximum(y_test, 1))) * 100)

        results.append({
            "Model": name,
            "MAE (INR)": round(mae, 2),
            "RMSE (INR)": round(rmse, 2),
            "MAPE (%)": round(mape, 2),
            "R2 Score": round(r2, 4)
        })

    leaderboard = pd.DataFrame(results).sort_values("RMSE (INR)").reset_index(drop=True)
    best_model_name = leaderboard.iloc[0]["Model"]
    best_model = trained_models[best_model_name]

    print("=== Sales Forecasting Model Evaluation ===")
    print(leaderboard.to_string(index=False))
    print(f"\nBest Model Selected: {best_model_name}")

    eval_summary = {
        "leaderboard": leaderboard,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "test_dates": test["Order_Date"],
        "y_test": y_test,
        "predictions": {name: m.predict(X_test) for name, m in trained_models.items()}
    }

    return eval_summary, leaderboard, best_model


def train_and_save_best_forecaster(
    df: pd.DataFrame,
    model_path: str = "models/sales_forecasting_model.pkl"
) -> Dict[str, Any]:
    """
    Train candidate models, identify the best one, refit on full historical data,
    and save the model along with feature metadata.
    """
    df_featured = prepare_forecasting_data(df)
    features = get_forecasting_features()

    eval_summary, leaderboard, best_candidate = compare_forecasting_models(df_featured, features)

    # Re-train best model on full available series
    X_full = df_featured[features]
    y_full = df_featured["Total_Sales_INR"]

    best_model_name = eval_summary["best_model_name"]
    if best_model_name == "Linear Regression":
        final_model = LinearRegression()
    elif best_model_name == "Ridge Regression":
        final_model = Ridge(alpha=10.0)
    elif best_model_name == "Random Forest":
        final_model = RandomForestRegressor(n_estimators=120, max_depth=8, min_samples_split=4, random_state=42)
    else:
        final_model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)

    final_model.fit(X_full, y_full)

    artifact = {
        "model": final_model,
        "model_name": best_model_name,
        "features": features,
        "last_known_date": df_featured["Order_Date"].max(),
        "recent_history": df_featured.tail(45),
        "leaderboard": leaderboard
    }

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(artifact, model_path)
    print(f"Saved best forecasting model ({best_model_name}) to: {model_path}")

    return artifact


def generate_future_forecast(
    artifact: Dict[str, Any],
    horizon_days: int = 30
) -> pd.DataFrame:
    """
    Perform multi-step recursive autoregressive forecasting for the next N days.
    """
    model = artifact["model"]
    features = artifact["features"]
    history = artifact["recent_history"].copy()

    last_date = pd.to_datetime(artifact["last_known_date"])
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, horizon_days + 1)]

    # We maintain a running series of sales to compute lags dynamically
    all_dates = list(history["Order_Date"])
    all_sales = list(history["Total_Sales_INR"])

    future_predictions = []

    for f_date in future_dates:
        # Build features for current future date
        row = {}
        for lag in [1, 2, 3, 7, 14, 21, 30]:
            row[f"lag_{lag}"] = all_sales[-lag]

        for window in [7, 14, 30]:
            slice_vals = all_sales[-window:]
            row[f"rolling_mean_{window}"] = np.mean(slice_vals)
            row[f"rolling_std_{window}"] = np.std(slice_vals) if len(slice_vals) > 1 else 0.0

        row["day_of_week"] = f_date.dayofweek
        row["day_of_month"] = f_date.day
        row["month"] = f_date.month
        row["quarter"] = f_date.quarter
        row["is_weekend"] = int(row["day_of_week"] in [5, 6])
        row["is_month_start"] = int(f_date.is_month_start)
        row["is_month_end"] = int(f_date.is_month_end)

        X_step = pd.DataFrame([row])[features]
        pred_val = float(model.predict(X_step)[0])
        # Ensure predicted revenue is strictly non-negative
        pred_val = max(0.0, pred_val)

        all_dates.append(f_date)
        all_sales.append(pred_val)
        future_predictions.append({
            "Date": f_date,
            "Predicted_Sales_INR": round(pred_val, 2),
            "Lower_Bound_INR": round(pred_val * 0.85, 2),
            "Upper_Bound_INR": round(pred_val * 1.15, 2)
        })

    forecast_df = pd.DataFrame(future_predictions)
    return forecast_df
