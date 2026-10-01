"""
Data Cleaning and Preprocessing Module
Amazon India Sales Analytics & AI-Powered Business Intelligence System
"""

import os
import pandas as pd
import numpy as np


def load_raw_data(file_path: str = None) -> pd.DataFrame:
    """
    Load raw Amazon sales data from Excel or CSV file.
    """
    if file_path is None:
        if os.path.exists("data/raw/amazon_sales.csv"):
            file_path = "data/raw/amazon_sales.csv"
        elif os.path.exists("data/raw/amazon_sales.xlsx"):
            file_path = "data/raw/amazon_sales.xlsx"
        else:
            file_path = "Amazon Sales Data India.xlsx"

    print(f"Loading raw dataset from: {file_path}")
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    print(f"Successfully loaded {len(df):,} records and {df.shape[1]} columns.")
    return df


def clean_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform thorough data cleaning, type casting, validation, and feature enrichment.
    """
    cleaned_df = df.copy()

    # 1. Deduplication
    duplicate_count = cleaned_df.duplicated().sum()
    if duplicate_count > 0:
        cleaned_df = cleaned_df.drop_duplicates()
        print(f"Removed {duplicate_count} duplicate rows.")

    # 2. Date conversion and sorting
    cleaned_df["Order_Date"] = pd.to_datetime(cleaned_df["Order_Date"])
    cleaned_df = cleaned_df.sort_values("Order_Date").reset_index(drop=True)

    # 3. Numeric type validation and formatting
    numeric_cols = ["Quantity", "Unit_Price_INR", "Discount_Pct", "Total_Sales_INR", "Profit_INR"]
    for col in numeric_cols:
        cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors="coerce")

    # 4. Fill any unexpected missing values if present
    cleaned_df["Quantity"] = cleaned_df["Quantity"].fillna(1).astype(int)
    cleaned_df["Unit_Price_INR"] = cleaned_df["Unit_Price_INR"].fillna(0.0)
    cleaned_df["Discount_Pct"] = cleaned_df["Discount_Pct"].fillna(0.0)
    cleaned_df["Total_Sales_INR"] = cleaned_df["Total_Sales_INR"].fillna(
        cleaned_df["Quantity"] * cleaned_df["Unit_Price_INR"] * (1 - cleaned_df["Discount_Pct"])
    )
    cleaned_df["Profit_INR"] = cleaned_df["Profit_INR"].fillna(0.0)

    # 5. String cleaning and trimming
    string_cols = ["Order_ID", "Category", "Product", "Payment_Method", "Fulfillment", "Order_Status", "Ship_State"]
    for col in string_cols:
        if col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].astype(str).str.strip()

    # 6. Time and Calendar Feature Engineering
    cleaned_df["Year"] = cleaned_df["Order_Date"].dt.year
    cleaned_df["Month"] = cleaned_df["Order_Date"].dt.month
    cleaned_df["Month_Name"] = cleaned_df["Order_Date"].dt.strftime("%b")
    cleaned_df["Year_Month"] = cleaned_df["Order_Date"].dt.strftime("%Y-%m")
    cleaned_df["Day"] = cleaned_df["Order_Date"].dt.day
    cleaned_df["Day_of_Week"] = cleaned_df["Order_Date"].dt.dayofweek
    cleaned_df["Day_Name"] = cleaned_df["Order_Date"].dt.strftime("%A")
    cleaned_df["Is_Weekend"] = cleaned_df["Day_of_Week"].isin([5, 6]).astype(int)
    cleaned_df["Quarter"] = cleaned_df["Order_Date"].dt.quarter

    # 7. Financial and Status Indicators
    cleaned_df["Profit_Margin_Pct"] = np.where(
        cleaned_df["Total_Sales_INR"] > 0,
        np.round((cleaned_df["Profit_INR"] / cleaned_df["Total_Sales_INR"]) * 100, 2),
        0.0,
    )
    cleaned_df["Is_Delivered"] = (cleaned_df["Order_Status"] == "Delivered").astype(int)
    cleaned_df["Is_Shipped"] = (cleaned_df["Order_Status"] == "Shipped").astype(int)
    cleaned_df["Is_Returned"] = (cleaned_df["Order_Status"] == "Returned").astype(int)
    cleaned_df["Is_Cancelled"] = (cleaned_df["Order_Status"] == "Cancelled").astype(int)
    cleaned_df["Is_Loss_Order"] = cleaned_df["Order_Status"].isin(["Returned", "Cancelled"]).astype(int)

    # Realized vs Lost Revenue
    cleaned_df["Realized_Sales_INR"] = np.where(
        cleaned_df["Order_Status"].isin(["Returned", "Cancelled"]),
        0.0,
        cleaned_df["Total_Sales_INR"]
    )
    cleaned_df["Lost_Sales_INR"] = np.where(
        cleaned_df["Order_Status"].isin(["Returned", "Cancelled"]),
        cleaned_df["Total_Sales_INR"],
        0.0
    )

    print(f"Data cleaning completed. Final shape: {cleaned_df.shape}")
    return cleaned_df


def save_processed_data(df: pd.DataFrame, output_path: str = "data/processed/amazon_sales_cleaned.csv") -> str:
    """
    Save cleaned dataframe to processed CSV directory.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved processed dataset to: {output_path}")
    return output_path


if __name__ == "__main__":
    raw_df = load_raw_data()
    clean_df = clean_sales_data(raw_df)
    save_processed_data(clean_df)
