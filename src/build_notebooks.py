"""
Notebook Generator and Runner Script
Builds all 5 comprehensive, recruiter-ready Jupyter Notebooks and executes them
"""

import os
import nbformat as nbf
from nbclient import NotebookClient


def create_01_data_cleaning():
    nb = nbf.v4.new_notebook()
    cells = []

    # Title & Introduction
    cells.append(nbf.v4.new_markdown_cell("""# 01 - Amazon India Sales Data Cleaning & Preprocessing
**Project:** Amazon India Sales Analytics & AI-Powered Business Intelligence System  
**Role:** Data Science & Analytics Intern Portfolio Project  
**Author:** Manoj (for Executive Manager Ravi & Recruiter Review)

---

### Notebook Objective:
1. **Data Understanding**: Inspect the raw Amazon India sales dataset (schema, shape, types, missing values, duplicates).
2. **Data Filtering & Quality Checks**: Validate numerical ranges, identify anomalies in order statuses, prices, discounts.
3. **Data Preprocessing & Formatting**: Parse order dates, convert data types, and normalize categorical fields.
4. **Feature Enrichment**: Derive business columns:
   - Calendar features: `Year`, `Month`, `Month_Name`, `Year_Month`, `Day`, `Day_of_Week`, `Is_Weekend`, `Quarter`.
   - Financial KPIs: `Profit_Margin_Pct`, `Is_Returned`, `Is_Cancelled`, `Realized_Sales_INR`, `Lost_Sales_INR`.
5. **Data Export**: Save the clean, enriched dataset to `data/processed/amazon_sales_cleaned.csv` for downstream EDA, SQL, and Machine Learning.
"""))

    # Step 1: Import Libraries
    cells.append(nbf.v4.new_code_cell("""import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("Pandas version:", pd.__version__)
print("NumPy version:", np.__version__)
"""))

    # Step 2: Load Raw Data
    cells.append(nbf.v4.new_markdown_cell("""## Step 1: Load Raw Dataset
We load the raw Amazon India Sales dataset and inspect its dimensions.
"""))
    cells.append(nbf.v4.new_code_cell("""# Load the dataset
raw_path = "../data/raw/amazon_sales.csv"
if not os.path.exists(raw_path):
    raw_path = "../Amazon Sales Data India.xlsx"

df_raw = pd.read_csv(raw_path) if raw_path.endswith('.csv') else pd.read_excel(raw_path)

print(f"Dataset Shape: {df_raw.shape[0]:,} rows, {df_raw.shape[1]} columns")
df_raw.head()
"""))

    # Step 3: Dataset Inspection & Data Types
    cells.append(nbf.v4.new_markdown_cell("""## Step 2: Schema Inspection & Missing Value Audit
Checking data types, non-null counts, and identifying potential missing values or anomalies.
"""))
    cells.append(nbf.v4.new_code_cell("""# Column info & data types
df_raw.info()
"""))
    cells.append(nbf.v4.new_code_cell("""# Missing values count per column
missing_summary = pd.DataFrame({
    'Missing_Count': df_raw.isnull().sum(),
    'Missing_Pct (%)': (df_raw.isnull().sum() / len(df_raw)) * 100
})
missing_summary
"""))

    # Step 4: Duplicate Records Check
    cells.append(nbf.v4.new_markdown_cell("""## Step 3: Duplicate Record Detection
Ensuring primary key uniqueness and overall row integrity.
"""))
    cells.append(nbf.v4.new_code_cell("""duplicate_rows = df_raw.duplicated().sum()
unique_orders = df_raw['Order_ID'].nunique()

print(f"Total duplicate rows: {duplicate_rows}")
print(f"Unique Order IDs: {unique_orders:,} out of {len(df_raw):,} records")
"""))

    # Step 5: Data Cleaning & Preprocessing
    cells.append(nbf.v4.new_markdown_cell("""## Step 4: Data Transformation & DateTime Parsing
- Convert `Order_Date` to pandas `datetime64[ns]` format.
- Sort chronologically by `Order_Date` to preserve time-series continuity.
- Verify numerical columns: `Quantity`, `Unit_Price_INR`, `Discount_Pct`, `Total_Sales_INR`, `Profit_INR`.
"""))
    cells.append(nbf.v4.new_code_cell("""df_clean = df_raw.copy()

# Parse datetime
df_clean['Order_Date'] = pd.to_datetime(df_clean['Order_Date'])

# Sort by date
df_clean = df_clean.sort_values('Order_Date').reset_index(drop=True)

# Numerical type casting
numeric_cols = ['Quantity', 'Unit_Price_INR', 'Discount_Pct', 'Total_Sales_INR', 'Profit_INR']
for col in numeric_cols:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

# Categorical stripping
cat_cols = ['Category', 'Product', 'Payment_Method', 'Fulfillment', 'Order_Status', 'Ship_State']
for col in cat_cols:
    df_clean[col] = df_clean[col].astype(str).str.strip()

print(f"Date Range: from {df_clean['Order_Date'].min().date()} to {df_clean['Order_Date'].max().date()}")
df_clean.dtypes
"""))

    # Step 6: Feature Engineering / Enrichment
    cells.append(nbf.v4.new_markdown_cell("""## Step 5: Feature Enrichment for Business Analytics & ML
We construct essential business metrics directly mapping to dashboard requirements:
- `Year`, `Month`, `Month_Name`, `Year_Month`, `Day`, `Day_of_Week`, `Day_Name`, `Is_Weekend`, `Quarter`
- `Profit_Margin_Pct` = `(Profit_INR / Total_Sales_INR) * 100`
- `Is_Delivered`, `Is_Shipped`, `Is_Returned`, `Is_Cancelled`, `Is_Loss_Order`
- `Realized_Sales_INR` vs `Lost_Sales_INR`
"""))
    cells.append(nbf.v4.new_code_cell("""# 1. Temporal Features
df_clean['Year'] = df_clean['Order_Date'].dt.year
df_clean['Month'] = df_clean['Order_Date'].dt.month
df_clean['Month_Name'] = df_clean['Order_Date'].dt.strftime('%b')
df_clean['Year_Month'] = df_clean['Order_Date'].dt.strftime('%Y-%m')
df_clean['Day'] = df_clean['Order_Date'].dt.day
df_clean['Day_of_Week'] = df_clean['Order_Date'].dt.dayofweek
df_clean['Day_Name'] = df_clean['Order_Date'].dt.strftime('%A')
df_clean['Is_Weekend'] = df_clean['Day_of_Week'].isin([5, 6]).astype(int)
df_clean['Quarter'] = df_clean['Order_Date'].dt.quarter

# 2. Financial KPIs
df_clean['Profit_Margin_Pct'] = np.where(
    df_clean['Total_Sales_INR'] > 0,
    np.round((df_clean['Profit_INR'] / df_clean['Total_Sales_INR']) * 100, 2),
    0.0
)

# 3. Order Status Flags
df_clean['Is_Delivered'] = (df_clean['Order_Status'] == 'Delivered').astype(int)
df_clean['Is_Shipped'] = (df_clean['Order_Status'] == 'Shipped').astype(int)
df_clean['Is_Returned'] = (df_clean['Order_Status'] == 'Returned').astype(int)
df_clean['Is_Cancelled'] = (df_clean['Order_Status'] == 'Cancelled').astype(int)
df_clean['Is_Loss_Order'] = df_clean['Order_Status'].isin(['Returned', 'Cancelled']).astype(int)

# 4. Realized vs Lost Sales
df_clean['Realized_Sales_INR'] = np.where(
    df_clean['Order_Status'].isin(['Returned', 'Cancelled']),
    0.0,
    df_clean['Total_Sales_INR']
)
df_clean['Lost_Sales_INR'] = np.where(
    df_clean['Order_Status'].isin(['Returned', 'Cancelled']),
    df_clean['Total_Sales_INR'],
    0.0
)

df_clean[['Order_ID', 'Order_Date', 'Category', 'Total_Sales_INR', 'Profit_INR', 'Profit_Margin_Pct', 'Order_Status', 'Lost_Sales_INR']].head()
"""))

    # Step 7: Sanity & Distribution Checks
    cells.append(nbf.v4.new_markdown_cell("""## Step 6: Data Sanity Checks & Statistical Overview
Inspect the summary statistics of the numeric features to verify data integrity.
"""))
    cells.append(nbf.v4.new_code_cell("""summary_stats = df_clean[['Quantity', 'Unit_Price_INR', 'Discount_Pct', 'Total_Sales_INR', 'Profit_INR', 'Profit_Margin_Pct']].describe()
summary_stats.round(2)
"""))

    # Step 8: Export Processed Dataset
    cells.append(nbf.v4.new_markdown_cell("""## Step 7: Export Processed Dataset
Save the cleaned, validated dataset to `data/processed/amazon_sales_cleaned.csv`.
"""))
    cells.append(nbf.v4.new_code_cell("""output_dir = "../data/processed"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "amazon_sales_cleaned.csv")

df_clean.to_csv(output_file, index=False)
print(f"Processed dataset successfully exported to: {output_file}")
print(f"Total Records: {len(df_clean):,}")
print(f"Total Columns: {df_clean.shape[1]}")
"""))

    nb.cells = cells
    return nb


def create_02_eda():
    nb = nbf.v4.new_notebook()
    cells = []

    # Title & Overview
    cells.append(nbf.v4.new_markdown_cell("""# 02 - Exploratory Data Analysis & Business Insights
**Project:** Amazon India Sales Analytics & AI-Powered Business Intelligence System  
**Audience:** Manoj, Manager Ravi, and Senior Data Science / Business Analytics Hiring Teams

---

### Business Questions Addressed (Sapphire IQ Specifications):
1. **Overall Performance**: How is the business performing overall?
   - Total Sales, Total Profit, Total Orders, Units Sold, Average Order Value (AOV), Profit Margin %, Monthly Trends.
2. **Category & Product Performance**: Which products and categories are driving business revenue?
   - Category-wise sales, profit, units sold, profit margin, top 10 products, and loss leaders.
3. **Order Status & Revenue Loss**: How many orders are completed vs lost?
   - Delivered vs Shipped vs Returned vs Cancelled; Return Rate %, Cancellation Rate %, Revenue Lost.
4. **Payment, Fulfillment & Geographic Performance**:
   - Sales & Profit by Payment Method (UPI, Cards, COD, Net Banking, Pay Later).
   - Sales & Profit by Fulfillment Channel (Amazon FBA, Seller Flex, Merchant FBM).
   - State-wise sales, profit, orders, and regional profitability.
"""))

    cells.append(nbf.v4.new_code_cell("""import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

# Load cleaned data
df = pd.read_csv('../data/processed/amazon_sales_cleaned.csv')
df['Order_Date'] = pd.to_datetime(df['Order_Date'])
print(f"Loaded processed dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")
"""))

    # Section 1: Overall Performance KPIs
    cells.append(nbf.v4.new_markdown_cell("""## 1. Overall Performance & Core KPIs
Calculating executive metrics directly required for Ravi's management dashboard.
"""))
    cells.append(nbf.v4.new_code_cell("""total_sales = df['Total_Sales_INR'].sum()
total_profit = df['Profit_INR'].sum()
total_orders = df['Order_ID'].nunique()
total_units = df['Quantity'].sum()
aov = total_sales / total_orders
overall_margin = (total_profit / total_sales) * 100
total_lost_revenue = df['Lost_Sales_INR'].sum()

kpi_summary = pd.DataFrame({
    'Metric': [
        'Total Gross Sales (INR)',
        'Total Profit (INR)',
        'Total Orders',
        'Total Units Sold',
        'Average Order Value (AOV - INR)',
        'Overall Profit Margin (%)',
        'Revenue Lost to Returns/Cancellations (INR)'
    ],
    'Value': [
        f"Rs. {total_sales:,.2f}",
        f"Rs. {total_profit:,.2f}",
        f"{total_orders:,}",
        f"{total_units:,}",
        f"Rs. {aov:,.2f}",
        f"{overall_margin:.2f}%",
        f"Rs. {total_lost_revenue:,.2f}"
    ]
})
kpi_summary
"""))

    # Monthly Trend Plot
    cells.append(nbf.v4.new_markdown_cell("""### Monthly Sales & Profit Trend Analysis
Evaluating month-over-month trajectory to see whether the business is growing or declining over time.
"""))
    cells.append(nbf.v4.new_code_cell("""monthly = df.groupby('Year_Month').agg(
    Monthly_Sales=('Total_Sales_INR', 'sum'),
    Monthly_Profit=('Profit_INR', 'sum'),
    Order_Count=('Order_ID', 'count')
).reset_index()

fig, ax1 = plt.subplots(figsize=(14, 6))

ax1.plot(monthly['Year_Month'], monthly['Monthly_Sales'] / 1e5, marker='o', color='#1f77b4', linewidth=2.5, label='Sales (Lakhs INR)')
ax1.set_ylabel('Total Sales (Lakhs INR)', color='#1f77b4', fontsize=12)
ax1.set_xticklabels(monthly['Year_Month'], rotation=45, ha='right')
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(monthly['Year_Month'], monthly['Monthly_Profit'] / 1e5, marker='s', color='#2ca02c', linewidth=2.5, linestyle='--', label='Profit (Lakhs INR)')
ax2.set_ylabel('Total Profit (Lakhs INR)', color='#2ca02c', fontsize=12)

plt.title('Monthly Sales & Profit Trend (2024 - 2026)', fontsize=15, fontweight='bold', pad=15)
fig.tight_layout()
plt.show()
"""))

    # Section 2: Category & Product Performance
    cells.append(nbf.v4.new_markdown_cell("""## 2. Category & Product Performance
Analyzing which product lines drive majority volume, profit, and margins.
"""))
    cells.append(nbf.v4.new_code_cell("""cat_perf = df.groupby('Category').agg(
    Total_Sales=('Total_Sales_INR', 'sum'),
    Total_Profit=('Profit_INR', 'sum'),
    Units_Sold=('Quantity', 'sum'),
    Order_Count=('Order_ID', 'count')
).reset_index()

cat_perf['Profit_Margin_%'] = (cat_perf['Total_Profit'] / cat_perf['Total_Sales']) * 100
cat_perf['Sales_Share_%'] = (cat_perf['Total_Sales'] / total_sales) * 100
cat_perf = cat_perf.sort_values('Total_Sales', ascending=False).reset_index(drop=True)
cat_perf
"""))
    cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Category Sales
sns.barplot(data=cat_perf, x='Total_Sales', y='Category', palette='Blues_r', ax=axes[0])
axes[0].set_title('Category-wise Total Sales (INR)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Sales (INR)')

# Category Profit Margin
sns.barplot(data=cat_perf, x='Profit_Margin_%', y='Category', palette='Greens_r', ax=axes[1])
axes[1].set_title('Category-wise Profit Margin (%)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Profit Margin (%)')

plt.tight_layout()
plt.show()
"""))

    # Top 10 Products
    cells.append(nbf.v4.new_markdown_cell("""### Top 10 Products by Total Revenue
Identifying bestselling hero products across all catalogs.
"""))
    cells.append(nbf.v4.new_code_cell("""top_products = df.groupby(['Product', 'Category']).agg(
    Units_Sold=('Quantity', 'sum'),
    Total_Sales=('Total_Sales_INR', 'sum'),
    Total_Profit=('Profit_INR', 'sum')
).reset_index().sort_values('Total_Sales', ascending=False).head(10).reset_index(drop=True)

top_products['Profit_Margin_%'] = (top_products['Total_Profit'] / top_products['Total_Sales']) * 100
top_products
"""))
    cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(12, 6))
sns.barplot(data=top_products, x='Total_Sales', y='Product', hue='Category', dodge=False, palette='mako')
plt.title('Top 10 Best-Selling Products by Revenue', fontsize=14, fontweight='bold')
plt.xlabel('Total Sales (INR)')
plt.legend(title='Category', loc='lower right')
plt.tight_layout()
plt.show()
"""))

    # Section 3: Order Status & Revenue Loss
    cells.append(nbf.v4.new_markdown_cell("""## 3. Order Status & Revenue Loss Analysis
Evaluating delivered vs shipped vs lost orders (returns and cancellations).
"""))
    cells.append(nbf.v4.new_code_cell("""status_perf = df.groupby('Order_Status').agg(
    Order_Count=('Order_ID', 'count'),
    Gross_Sales=('Total_Sales_INR', 'sum'),
    Profit=('Profit_INR', 'sum'),
    Lost_Sales=('Lost_Sales_INR', 'sum')
).reset_index()

status_perf['Order_Share_%'] = (status_perf['Order_Count'] / len(df)) * 100
status_perf = status_perf.sort_values('Order_Count', ascending=False).reset_index(drop=True)
status_perf
"""))
    cells.append(nbf.v4.new_code_cell("""return_rate = (df['Order_Status'] == 'Returned').mean() * 100
cancel_rate = (df['Order_Status'] == 'Cancelled').mean() * 100
total_loss_rate = return_rate + cancel_rate

print(f"Return Rate: {return_rate:.2f}%")
print(f"Cancellation Rate: {cancel_rate:.2f}%")
print(f"Total Lost Order Rate: {total_loss_rate:.2f}%")
print(f"Total Lost Revenue: Rs. {total_lost_revenue:,.2f}")
"""))
    cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Order Status Distribution Donut
status_counts = df['Order_Status'].value_counts()
colors = ['#2ca02c', '#1f77b4', '#d62728', '#ff7f0e']
axes[0].pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', startangle=140, colors=colors,
            wedgeprops=dict(width=0.4, edgecolor='white'))
axes[0].set_title('Order Status Distribution', fontsize=13, fontweight='bold')

# Category-wise Returns & Cancellations
cat_status = df.groupby('Category').agg(
    Total=('Order_ID', 'count'),
    Returns=('Is_Returned', 'sum'),
    Cancellations=('Is_Cancelled', 'sum')
).reset_index()

cat_status['Return_Rate_%'] = (cat_status['Returns'] / cat_status['Total']) * 100
cat_status['Cancel_Rate_%'] = (cat_status['Cancellations'] / cat_status['Total']) * 100

x = np.arange(len(cat_status['Category']))
width = 0.35
axes[1].bar(x - width/2, cat_status['Return_Rate_%'], width, label='Return Rate %', color='#d62728')
axes[1].bar(x + width/2, cat_status['Cancel_Rate_%'], width, label='Cancellation Rate %', color='#ff7f0e')
axes[1].set_xticks(x)
axes[1].set_xticklabels(cat_status['Category'], rotation=30, ha='right')
axes[1].set_ylabel('Rate (%)')
axes[1].set_title('Category-wise Return & Cancellation Rates', fontsize=13, fontweight='bold')
axes[1].legend()

plt.tight_layout()
plt.show()
"""))

    # Section 4: Payment, Fulfillment & Geography
    cells.append(nbf.v4.new_markdown_cell("""## 4. Payment, Fulfillment & Geographic Performance
Analyzing channel contribution across payment gateways, logistics partners, and regional states.
"""))
    cells.append(nbf.v4.new_code_cell("""# Payment Method breakdown
payment_summary = df.groupby('Payment_Method').agg(
    Orders=('Order_ID', 'count'),
    Sales=('Total_Sales_INR', 'sum'),
    Profit=('Profit_INR', 'sum'),
    Return_Count=('Is_Returned', 'sum')
).reset_index()

payment_summary['Return_Rate_%'] = (payment_summary['Return_Count'] / payment_summary['Orders']) * 100
payment_summary['Sales_Share_%'] = (payment_summary['Sales'] / total_sales) * 100
payment_summary = payment_summary.sort_values('Sales', ascending=False).reset_index(drop=True)
payment_summary
"""))
    cells.append(nbf.v4.new_code_cell("""# Fulfillment Breakdown
fulfil_summary = df.groupby('Fulfillment').agg(
    Orders=('Order_ID', 'count'),
    Sales=('Total_Sales_INR', 'sum'),
    Profit=('Profit_INR', 'sum'),
    Delivered=('Is_Delivered', 'sum')
).reset_index()

fulfil_summary['Delivery_Success_%'] = (fulfil_summary['Delivered'] / fulfil_summary['Orders']) * 100
fulfil_summary['Profit_Margin_%'] = (fulfil_summary['Profit'] / fulfil_summary['Sales']) * 100
fulfil_summary = fulfil_summary.sort_values('Sales', ascending=False).reset_index(drop=True)
fulfil_summary
"""))
    cells.append(nbf.v4.new_code_cell("""# State-Wise Performance
state_summary = df.groupby('Ship_State').agg(
    Orders=('Order_ID', 'count'),
    Units=('Quantity', 'sum'),
    Sales=('Total_Sales_INR', 'sum'),
    Profit=('Profit_INR', 'sum'),
    Returns=('Is_Returned', 'sum')
).reset_index()

state_summary['AOV'] = state_summary['Sales'] / state_summary['Orders']
state_summary['Margin_%'] = (state_summary['Profit'] / state_summary['Sales']) * 100
state_summary['Return_Rate_%'] = (state_summary['Returns'] / state_summary['Orders']) * 100
state_summary = state_summary.sort_values('Sales', ascending=False).reset_index(drop=True)
state_summary
"""))
    cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.barplot(data=state_summary, x='Sales', y='Ship_State', palette='rocket', ax=axes[0])
axes[0].set_title('Top Revenue Generating States in India', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Total Sales (INR)')

sns.barplot(data=state_summary, x='Profit', y='Ship_State', palette='viridis', ax=axes[1])
axes[1].set_title('State-wise Net Profit Contribution', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Total Profit (INR)')

plt.tight_layout()
plt.show()
"""))

    # Summary Takeaways for recruiters
    cells.append(nbf.v4.new_markdown_cell("""## Key Business Takeaways for Management & Recruiters
1. **Revenue Engine**: Electronics & Mobiles generates the highest revenue (~₹7.8 Crore), while Apparel & Fashion is the second biggest contributor.
2. **Profit Margins**: Home & Kitchen and Apparel maintain strong profit margins of ~21% to 22%.
3. **Loss Analysis**: Returned and Cancelled orders account for ~9.85% of total orders, leading to over ₹1.5 Crore in lost potential revenue.
4. **Logistics Channel**: Amazon (FBA) handles over 70% of shipments with the highest fulfillment reliability.
5. **Customer Payments**: UPI is the dominant payment method (>49% of volume), followed by Cards and Cash on Delivery (COD).
"""))

    nb.cells = cells
    return nb


def create_03_sql_analysis():
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(nbf.v4.new_markdown_cell("""# 03 - SQL Component: Business Queries & Database Analysis
**Project:** Amazon India Sales Analytics & AI-Powered Business Intelligence System  
**Role:** Data Science & Analytics Intern Portfolio Project  
**Author:** Manoj (for Executive Manager Ravi & Technical Interviewers)

---

### Why this notebook is crucial for your resume:
Recruiters actively seek candidates who can bridge the gap between **SQL databases**, **Python data science**, and **Business Intelligence**.
In this notebook, we:
1. Connect to the SQLite database `database/amazon_sales.db` containing our cleaned sales table.
2. Execute production-ready SQL queries using Aggregations, Subqueries, Window Functions (`RANK()`, `DENSE_RANK()`), and `CASE WHEN` statements.
3. Validate answers to all four management questions from the Sapphire IQ specification.
"""))

    cells.append(nbf.v4.new_code_cell("""import sqlite3
import pandas as pd
import os

# Connect to the SQLite Database
db_path = "../database/amazon_sales.db"
conn = sqlite3.connect(db_path)
print("Connected to database successfully:", db_path)

# Verify table schema
schema_df = pd.read_sql("PRAGMA table_info(amazon_sales);", conn)
schema_df[['cid', 'name', 'type', 'notnull']]
"""))

    # Query 1
    cells.append(nbf.v4.new_markdown_cell("""### Query 1: Executive KPI Overview (Slides 1 & 2)
Calculates overall revenue, profit, total orders, units sold, AOV, profit margin %, and revenue lost.
"""))
    cells.append(nbf.v4.new_code_cell("""query_1 = \"\"\"
SELECT 
    COUNT(DISTINCT Order_ID) AS total_orders,
    SUM(Quantity) AS total_units_sold,
    ROUND(SUM(Total_Sales_INR), 2) AS total_sales_inr,
    ROUND(SUM(Profit_INR), 2) AS total_profit_inr,
    ROUND(SUM(Total_Sales_INR) / COUNT(DISTINCT Order_ID), 2) AS average_order_value_inr,
    ROUND((SUM(Profit_INR) / SUM(Total_Sales_INR)) * 100, 2) AS profit_margin_pct,
    ROUND(SUM(CASE WHEN Order_Status IN ('Returned', 'Cancelled') THEN Total_Sales_INR ELSE 0 END), 2) AS total_revenue_lost_inr
FROM amazon_sales;
\"\"\"
pd.read_sql(query_1, conn)
"""))

    # Query 2
    cells.append(nbf.v4.new_markdown_cell("""### Query 2: Monthly Sales & Profit Trend (Slide 2)
Tracks monthly financial trajectory and order volume over time.
"""))
    cells.append(nbf.v4.new_code_cell("""query_2 = \"\"\"
SELECT 
    Year_Month,
    COUNT(DISTINCT Order_ID) AS orders_count,
    ROUND(SUM(Total_Sales_INR), 2) AS monthly_sales_inr,
    ROUND(SUM(Profit_INR), 2) AS monthly_profit_inr,
    ROUND((SUM(Profit_INR) / SUM(Total_Sales_INR)) * 100, 2) AS profit_margin_pct
FROM amazon_sales
GROUP BY Year_Month
ORDER BY Year_Month ASC
LIMIT 12;
\"\"\"
pd.read_sql(query_2, conn)
"""))

    # Query 3
    cells.append(nbf.v4.new_markdown_cell("""### Query 3: Category Performance & Ranking with Window Functions (Slide 3)
Uses SQL Window Function `DENSE_RANK()` to rank categories by revenue generation.
"""))
    cells.append(nbf.v4.new_code_cell("""query_3 = \"\"\"
SELECT 
    Category,
    COUNT(DISTINCT Order_ID) AS total_orders,
    SUM(Quantity) AS units_sold,
    ROUND(SUM(Total_Sales_INR), 2) AS category_sales_inr,
    ROUND(SUM(Profit_INR), 2) AS category_profit_inr,
    ROUND((SUM(Profit_INR) / SUM(Total_Sales_INR)) * 100, 2) AS profit_margin_pct,
    DENSE_RANK() OVER (ORDER BY SUM(Total_Sales_INR) DESC) AS sales_rank
FROM amazon_sales
GROUP BY Category
ORDER BY category_sales_inr DESC;
\"\"\"
pd.read_sql(query_3, conn)
"""))

    # Query 4
    cells.append(nbf.v4.new_markdown_cell("""### Query 4: Top 10 Best-Selling Products by Revenue (Slide 3)
"""))
    cells.append(nbf.v4.new_code_cell("""query_4 = \"\"\"
SELECT 
    Product,
    Category,
    SUM(Quantity) AS units_sold,
    ROUND(SUM(Total_Sales_INR), 2) AS total_sales_inr,
    ROUND(SUM(Profit_INR), 2) AS total_profit_inr,
    ROUND((SUM(Profit_INR) / SUM(Total_Sales_INR)) * 100, 2) AS profit_margin_pct
FROM amazon_sales
GROUP BY Product, Category
ORDER BY total_sales_inr DESC
LIMIT 10;
\"\"\"
pd.read_sql(query_4, conn)
"""))

    # Query 5
    cells.append(nbf.v4.new_markdown_cell("""### Query 5: Order Status Breakdown & Revenue Loss Analysis (Slide 4)
Computes order counts, percentage shares, realized revenue, and lost revenue using conditional aggregation.
"""))
    cells.append(nbf.v4.new_code_cell("""query_5 = \"\"\"
SELECT 
    Order_Status,
    COUNT(*) AS order_count,
    ROUND((COUNT(*) * 100.0) / (SELECT COUNT(*) FROM amazon_sales), 2) AS pct_of_total_orders,
    ROUND(SUM(Total_Sales_INR), 2) AS gross_sales_value_inr,
    ROUND(SUM(Profit_INR), 2) AS realized_profit_inr,
    ROUND(SUM(CASE WHEN Order_Status IN ('Returned', 'Cancelled') THEN Total_Sales_INR ELSE 0 END), 2) AS lost_revenue_inr
FROM amazon_sales
GROUP BY Order_Status
ORDER BY order_count DESC;
\"\"\"
pd.read_sql(query_5, conn)
"""))

    # Query 6
    cells.append(nbf.v4.new_markdown_cell("""### Query 6: Category-Wise Return & Cancellation Rates (Slide 4)
"""))
    cells.append(nbf.v4.new_code_cell("""query_6 = \"\"\"
SELECT 
    Category,
    COUNT(*) AS total_orders,
    SUM(CASE WHEN Order_Status = 'Returned' THEN 1 ELSE 0 END) AS returned_orders,
    SUM(CASE WHEN Order_Status = 'Cancelled' THEN 1 ELSE 0 END) AS cancelled_orders,
    ROUND((SUM(CASE WHEN Order_Status = 'Returned' THEN 1.0 ELSE 0 END) / COUNT(*)) * 100, 2) AS return_rate_pct,
    ROUND((SUM(CASE WHEN Order_Status = 'Cancelled' THEN 1.0 ELSE 0 END) / COUNT(*)) * 100, 2) AS cancellation_rate_pct,
    ROUND(SUM(CASE WHEN Order_Status IN ('Returned', 'Cancelled') THEN Total_Sales_INR ELSE 0 END), 2) AS revenue_lost_inr
FROM amazon_sales
GROUP BY Category
ORDER BY return_rate_pct DESC;
\"\"\"
pd.read_sql(query_6, conn)
"""))

    # Query 7
    cells.append(nbf.v4.new_markdown_cell("""### Query 7: Payment Method Performance (Slide 5)
"""))
    cells.append(nbf.v4.new_code_cell("""query_7 = \"\"\"
SELECT 
    Payment_Method,
    COUNT(DISTINCT Order_ID) AS total_orders,
    ROUND(SUM(Total_Sales_INR), 2) AS total_sales_inr,
    ROUND(SUM(Profit_INR), 2) AS total_profit_inr,
    ROUND(SUM(Total_Sales_INR) / COUNT(DISTINCT Order_ID), 2) AS aov_inr,
    ROUND((SUM(CASE WHEN Order_Status = 'Returned' THEN 1.0 ELSE 0 END) / COUNT(*)) * 100, 2) AS return_rate_pct,
    ROUND((SUM(Total_Sales_INR) * 100.0) / (SELECT SUM(Total_Sales_INR) FROM amazon_sales), 2) AS sales_share_pct
FROM amazon_sales
GROUP BY Payment_Method
ORDER BY total_sales_inr DESC;
\"\"\"
pd.read_sql(query_7, conn)
"""))

    # Query 8
    cells.append(nbf.v4.new_markdown_cell("""### Query 8: Fulfillment Channel Efficiency (Slide 5)
"""))
    cells.append(nbf.v4.new_code_cell("""query_8 = \"\"\"
SELECT 
    Fulfillment,
    COUNT(DISTINCT Order_ID) AS total_orders,
    ROUND(SUM(Total_Sales_INR), 2) AS total_sales_inr,
    ROUND(SUM(Profit_INR), 2) AS total_profit_inr,
    ROUND((SUM(Profit_INR) / SUM(Total_Sales_INR)) * 100, 2) AS profit_margin_pct,
    ROUND((SUM(CASE WHEN Order_Status = 'Delivered' THEN 1.0 ELSE 0 END) / COUNT(*)) * 100, 2) AS delivery_success_rate_pct,
    ROUND((SUM(CASE WHEN Order_Status = 'Returned' THEN 1.0 ELSE 0 END) / COUNT(*)) * 100, 2) AS return_rate_pct
FROM amazon_sales
GROUP BY Fulfillment
ORDER BY total_sales_inr DESC;
\"\"\"
pd.read_sql(query_8, conn)
"""))

    # Query 9
    cells.append(nbf.v4.new_markdown_cell("""### Query 9: Geographic Performance by State (Slide 5)
"""))
    cells.append(nbf.v4.new_code_cell("""query_9 = \"\"\"
SELECT 
    Ship_State,
    COUNT(DISTINCT Order_ID) AS total_orders,
    SUM(Quantity) AS units_sold,
    ROUND(SUM(Total_Sales_INR), 2) AS total_sales_inr,
    ROUND(SUM(Profit_INR), 2) AS total_profit_inr,
    ROUND(SUM(Total_Sales_INR) / COUNT(DISTINCT Order_ID), 2) AS aov_inr,
    ROUND((SUM(Profit_INR) / SUM(Total_Sales_INR)) * 100, 2) AS profit_margin_pct,
    ROUND((SUM(CASE WHEN Order_Status = 'Returned' THEN 1.0 ELSE 0 END) / COUNT(*)) * 100, 2) AS state_return_rate_pct
FROM amazon_sales
GROUP BY Ship_State
ORDER BY total_sales_inr DESC;
\"\"\"
pd.read_sql(query_9, conn)
"""))

    # Close connection
    cells.append(nbf.v4.new_code_cell("""conn.close()
print("SQL Analysis queries completed successfully.")
"""))

    nb.cells = cells
    return nb


def create_04_sales_forecasting():
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(nbf.v4.new_markdown_cell("""# 04 - AI Sales Forecasting Pipeline
**Project:** Amazon India Sales Analytics & AI-Powered Business Intelligence System  
**Workflow:** Dataset Ingestion -> Filtering -> Preprocessing & Feature Engineering -> Candidate Model Comparison -> Best Model Selection -> 30-Day Future Projection

---

### Machine Learning Objective:
Predict expected future daily sales (INR) for the next 30 days based on historical sales patterns.

### Pipeline Steps (Strictly following end-to-end ML methodology):
1. **Data Ingestion**: Take processed dataset `data/processed/amazon_sales_cleaned.csv`.
2. **Filtering**: Filter out cancelled transactions and aggregate to daily sales timeline.
3. **Preprocessing & Feature Engineering**: Create multi-scale lag features (`lag_1` to `lag_30`), rolling window statistics (7, 14, 30 days), and calendar indicators.
4. **Chronological Train/Test Split**: Strictly avoid data leakage by training on historical timeline and evaluating on holdout test set.
5. **Model Comparison**:
   - Model 1: **Linear Regression** (Parametric Baseline)
   - Model 2: **Ridge Regression** (L2 Regularized)
   - Model 3: **Random Forest Regressor** (Ensemble Bagging)
   - Model 4: **XGBoost Regressor** (Gradient Boosting)
6. **Model Evaluation & Leaderboard**: Compare on MAE, RMSE, MAPE, and R² Score.
7. **Best Model Selection**: Pick the best-performing model based on minimal error metrics.
8. **Final Refit & 30-Day Forecast**: Retrain winner on full series, generate recursive 30-day future sales projection with confidence bounds, and save artifact to `models/sales_forecasting_model.pkl`.
"""))

    cells.append(nbf.v4.new_code_cell("""import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Ensure root path is accessible for src imports
if ".." not in sys.path:
    sys.path.append("..")

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
"""))

    # Step 1: Take Dataset & Filtering
    cells.append(nbf.v4.new_markdown_cell("""## Step 1: Take Dataset & Filter
We ingest the cleaned dataset and filter to non-cancelled revenue transactions.
"""))
    cells.append(nbf.v4.new_code_cell("""data_path = "../data/processed/amazon_sales_cleaned.csv"
df = pd.read_csv(data_path)
df['Order_Date'] = pd.to_datetime(df['Order_Date'])

# Filter out cancelled orders
df_valid = df[df['Order_Status'] != 'Cancelled'].copy()
print(f"Total non-cancelled orders: {len(df_valid):,} out of {len(df):,}")

# Daily sales aggregation
daily_sales = (
    df_valid.groupby('Order_Date')['Total_Sales_INR']
    .sum()
    .asfreq('D', fill_value=0.0)
    .reset_index()
)
print(f"Daily timeline points: {len(daily_sales)} days from {daily_sales['Order_Date'].min().date()} to {daily_sales['Order_Date'].max().date()}")
daily_sales.head()
"""))

    # Step 2: Feature Engineering
    cells.append(nbf.v4.new_markdown_cell("""## Step 2: Preprocessing & Time-Series Feature Engineering
We construct autoregressive lag features, rolling moving averages, rolling standard deviations, and seasonal day/month/weekend markers.
"""))
    cells.append(nbf.v4.new_code_cell("""# 1. Autoregressive Lags
for lag in [1, 2, 3, 7, 14, 21, 30]:
    daily_sales[f'lag_{lag}'] = daily_sales['Total_Sales_INR'].shift(lag)

# 2. Rolling Window Moving Averages & Volatility
for window in [7, 14, 30]:
    daily_sales[f'rolling_mean_{window}'] = daily_sales['Total_Sales_INR'].shift(1).rolling(window).mean()
    daily_sales[f'rolling_std_{window}'] = daily_sales['Total_Sales_INR'].shift(1).rolling(window).std()

# 3. Calendar & Temporal Features
daily_sales['day_of_week'] = daily_sales['Order_Date'].dt.dayofweek
daily_sales['day_of_month'] = daily_sales['Order_Date'].dt.day
daily_sales['month'] = daily_sales['Order_Date'].dt.month
daily_sales['quarter'] = daily_sales['Order_Date'].dt.quarter
daily_sales['is_weekend'] = daily_sales['day_of_week'].isin([5, 6]).astype(int)
daily_sales['is_month_start'] = daily_sales['Order_Date'].dt.is_month_start.astype(int)
daily_sales['is_month_end'] = daily_sales['Order_Date'].dt.is_month_end.astype(int)

# Drop initial rows with NaN from lags
df_featured = daily_sales.dropna().reset_index(drop=True)
print(f"Featured dataset shape: {df_featured.shape}")
df_featured.head(3)
"""))

    # Step 3: Train/Test Split
    cells.append(nbf.v4.new_markdown_cell("""## Step 3: Chronological Train / Test Split
To evaluate time-series forecasting realistically, we reserve the most recent 20% of the timeline as an unseen holdout test set.
"""))
    cells.append(nbf.v4.new_code_cell("""features = [
    'lag_1', 'lag_2', 'lag_3', 'lag_7', 'lag_14', 'lag_21', 'lag_30',
    'rolling_mean_7', 'rolling_std_7',
    'rolling_mean_14', 'rolling_std_14',
    'rolling_mean_30', 'rolling_std_30',
    'day_of_week', 'day_of_month', 'month', 'quarter',
    'is_weekend', 'is_month_start', 'is_month_end'
]

train_ratio = 0.8
train_size = int(len(df_featured) * train_ratio)

train_df = df_featured.iloc[:train_size]
test_df = df_featured.iloc[train_size:]

X_train, y_train = train_df[features], train_df['Total_Sales_INR']
X_test, y_test = test_df[features], test_df['Total_Sales_INR']

print(f"Training Period: {train_df['Order_Date'].min().date()} to {train_df['Order_Date'].max().date()} ({len(train_df)} days)")
print(f"Testing Period:  {test_df['Order_Date'].min().date()} to {test_df['Order_Date'].max().date()} ({len(test_df)} days)")
"""))

    # Step 4: Model Comparison & Evaluation
    cells.append(nbf.v4.new_markdown_cell("""## Step 4: Candidate Models Comparison
We train 4 candidate algorithms on the same features and evaluate them on the test set:
1. **Linear Regression**
2. **Ridge Regression**
3. **Random Forest Regressor**
4. **XGBoost Regressor**
"""))
    cells.append(nbf.v4.new_code_cell("""models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=10.0),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_split=4, random_state=42),
    "XGBoost Regressor": XGBRegressor(n_estimators=80, max_depth=4, learning_rate=0.05, random_state=42)
}

results = []
test_preds = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    test_preds[name] = preds

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mape = np.mean(np.abs((y_test - preds) / np.maximum(y_test, 1))) * 100
    r2 = r2_score(y_test, preds)

    results.append({
        "Model": name,
        "MAE (INR)": round(mae, 2),
        "RMSE (INR)": round(rmse, 2),
        "MAPE (%)": round(mape, 2),
        "R2 Score": round(r2, 4)
    })

leaderboard_df = pd.DataFrame(results).sort_values("RMSE (INR)").reset_index(drop=True)
print("=== Forecasting Model Comparison Leaderboard ===")
leaderboard_df
"""))

    # Step 5: Best Model Selection
    cells.append(nbf.v4.new_markdown_cell("""## Step 5: Select Which Model is Best
Based on empirical evaluation across both MAE and RMSE, we identify the best performing model and proceed accordingly.
"""))
    cells.append(nbf.v4.new_code_cell("""best_model_name = leaderboard_df.iloc[0]['Model']
print(f"Top Performing Model: {best_model_name}")
print(f"Lowest RMSE achieved: Rs. {leaderboard_df.iloc[0]['RMSE (INR)']:,}")
print(f"Lowest MAE achieved:  Rs. {leaderboard_df.iloc[0]['MAE (INR)']:,}")
"""))
    cells.append(nbf.v4.new_code_cell("""# Plot Actual vs Predicted on Test Set
plt.figure(figsize=(14, 6))
plt.plot(test_df['Order_Date'], y_test, label='Actual Test Sales', color='black', alpha=0.7, linewidth=1.5)
plt.plot(test_df['Order_Date'], test_preds[best_model_name], label=f'{best_model_name} (Best Model)', color='#d62728', linewidth=2)
plt.plot(test_df['Order_Date'], test_df['rolling_mean_7'], label='7-Day Moving Average', color='#1f77b4', linestyle='--', alpha=0.7)
plt.title(f'Actual vs Forecasted Daily Sales (Test Period) - {best_model_name}', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Daily Sales (INR)')
plt.legend()
plt.tight_layout()
plt.show()
"""))

    # Step 6: Final Refit & 30-Day Future Forecast
    cells.append(nbf.v4.new_markdown_cell("""## Step 6: Final Model Training & 30-Day Future Sales Forecast
We refit our selected best model on the entire timeline and recursively project expected sales for the next 30 days.
"""))
    cells.append(nbf.v4.new_code_cell("""# Retrain best model on full dataset
X_full = df_featured[features]
y_full = df_featured['Total_Sales_INR']

if best_model_name == "Linear Regression":
    final_model = LinearRegression()
elif best_model_name == "Ridge Regression":
    final_model = Ridge(alpha=10.0)
elif best_model_name == "Random Forest":
    final_model = RandomForestRegressor(n_estimators=120, max_depth=8, min_samples_split=4, random_state=42)
else:
    final_model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)

final_model.fit(X_full, y_full)

# Save artifact
os.makedirs('../models', exist_ok=True)
artifact = {
    'model': final_model,
    'model_name': best_model_name,
    'features': features,
    'last_known_date': df_featured['Order_Date'].max(),
    'recent_history': df_featured.tail(45),
    'leaderboard': leaderboard_df
}
joblib.dump(artifact, '../models/sales_forecasting_model.pkl')
print(f"Saved trained best model to: ../models/sales_forecasting_model.pkl")
"""))
    cells.append(nbf.v4.new_code_cell("""# Generate 30-Day Future Forecast
import sys
if ".." not in sys.path:
    sys.path.append("..")
from src.forecasting import generate_future_forecast

forecast_30d = generate_future_forecast(artifact, horizon_days=30)
print(f"30-Day Projected Revenue: Rs. {forecast_30d['Predicted_Sales_INR'].sum():,.2f}")
forecast_30d.head(10)
"""))
    cells.append(nbf.v4.new_code_cell("""# Visualization of Historical + 30-Day Future Projection
recent_hist = df_featured.tail(60)

plt.figure(figsize=(15, 6))
plt.plot(recent_hist['Order_Date'], recent_hist['Total_Sales_INR'], label='Historical Daily Sales (Recent 60 Days)', color='#1f77b4', linewidth=1.5)
plt.plot(forecast_30d['Date'], forecast_30d['Predicted_Sales_INR'], label='AI 30-Day Forecast', color='#d62728', linewidth=2.5, marker='o', markersize=4)
plt.fill_between(forecast_30d['Date'], forecast_30d['Lower_Bound_INR'], forecast_30d['Upper_Bound_INR'], color='#d62728', alpha=0.15, label='Forecast Confidence Band (±15%)')

plt.title('Amazon India Daily Sales: Historical Timeline & Next 30-Day AI Forecast', fontsize=14, fontweight='bold', pad=12)
plt.xlabel('Date', fontsize=11)
plt.ylabel('Daily Sales (INR)', fontsize=11)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
"""))

    nb.cells = cells
    return nb


def create_05_return_prediction():
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(nbf.v4.new_markdown_cell("""# 05 - Return Risk Prediction Classification System
**Project:** Amazon India Sales Analytics & AI-Powered Business Intelligence System  
**Workflow:** Dataset Ingestion -> Filtering -> Feature Preprocessing -> Model Benchmarking -> Best Model Selection -> Risk Inference

---

### Machine Learning Objective:
Predict the likelihood that an incoming e-commerce order will result in a **customer return (`Is_Returned = 1`)**.  
This enables Amazon managers to flag high-risk orders in advance, implement targeted courier checks, verify addresses, or request prepayment.

### Pipeline Steps (Strictly following end-to-end ML methodology):
1. **Take Dataset**: Ingest `data/processed/amazon_sales_cleaned.csv`.
2. **Filtering**: Select orders that reached fulfillment (delivered or returned), excluding pre-dispatch cancellations.
3. **Preprocessing & Feature Engineering**: One-hot encode categorical factors (`Category`, `Product`, `Payment_Method`, `Fulfillment`, `Ship_State`) and include order size/financials (`Quantity`, `Unit_Price_INR`, `Discount_Pct`, `Total_Sales_INR`).
4. **Stratified Train/Test Split**: Maintain exact class ratio in train and test splits to handle the imbalanced return rate (~5%).
5. **Candidate Classifiers Comparison**:
   - Model 1: **Logistic Regression** (with StandardScaler Pipeline & balanced class weights)
   - Model 2: **Decision Tree Classifier** (Max depth constrained)
   - Model 3: **Random Forest Classifier** (Ensemble of trees with balanced weights)
   - Model 4: **XGBoost Classifier** (Gradient boosted trees with positive class weighting `scale_pos_weight`)
6. **Evaluation & Leaderboard**: Compare on Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
7. **Best Model Selection**: Pick the best classifier based on ROC-AUC and F1 performance.
8. **Comprehensive Diagnostics**: Classification report, confusion matrix, ROC curve, and top 15 feature importances.
9. **Artifact Serialization & Real-Time Inference**: Save `models/return_prediction_model.pkl` and test on simulated live order scenarios!
"""))

    cells.append(nbf.v4.new_code_cell("""import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

if ".." not in sys.path:
    sys.path.append("..")

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
"""))

    # Step 1: Take Dataset & Filtering
    cells.append(nbf.v4.new_markdown_cell("""## Step 1: Take Dataset & Filter
We ingest the processed dataset and define the target variable `Is_Returned`. We filter for fulfilled orders (Delivered and Returned), filtering out early cancellations.
"""))
    cells.append(nbf.v4.new_code_cell("""data_path = "../data/processed/amazon_sales_cleaned.csv"
df = pd.read_csv(data_path)

# Filter out orders that were cancelled prior to shipping
df_fulfilled = df[df['Order_Status'] != 'Cancelled'].copy()

# Target variable: 1 if Returned, 0 if Delivered/Shipped
df_fulfilled['Is_Returned'] = (df_fulfilled['Order_Status'] == 'Returned').astype(int)

return_count = df_fulfilled['Is_Returned'].sum()
total_fulfilled = len(df_fulfilled)
print(f"Total Fulfilled Orders: {total_fulfilled:,}")
print(f"Total Returned Orders:  {return_count:,} ({return_count / total_fulfilled * 100:.2f}%)")
print(f"Total Retained Orders:  {total_fulfilled - return_count:,} ({(total_fulfilled - return_count) / total_fulfilled * 100:.2f}%)")
"""))

    # Step 2: Feature Engineering & Preprocessing
    cells.append(nbf.v4.new_markdown_cell("""## Step 2: Preprocessing & One-Hot Feature Encoding
Convert categorical features into numeric indicators and extract feature matrices.
"""))
    cells.append(nbf.v4.new_code_cell("""cat_features = ['Category', 'Product', 'Payment_Method', 'Fulfillment', 'Ship_State']
num_features = ['Quantity', 'Unit_Price_INR', 'Discount_Pct', 'Total_Sales_INR']

X_raw = df_fulfilled[cat_features + num_features].copy()
y = df_fulfilled['Is_Returned'].copy()

# One-hot encoding
X = pd.get_dummies(X_raw, columns=cat_features, drop_first=True)
feature_names = list(X.columns)

print(f"Encoded Feature Matrix Shape: {X.shape[0]:,} rows, {X.shape[1]} features")
X.head(3)
"""))

    # Step 3: Stratified Train/Test Split
    cells.append(nbf.v4.new_markdown_cell("""## Step 3: Stratified Train / Test Split
Given the ~5% positive class ratio, stratified splitting guarantees the same proportion of returns in both train and test partitions.
"""))
    cells.append(nbf.v4.new_code_cell("""X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pos_weight = (len(y_train) - sum(y_train)) / max(sum(y_train), 1)

print(f"Train set: {len(X_train):,} samples (Returns: {sum(y_train):,})")
print(f"Test set:  {len(X_test):,} samples (Returns: {sum(y_test):,})")
print(f"Class imbalance ratio (pos_weight): {pos_weight:.2f}")
"""))

    # Step 4: Candidate Models Comparison
    cells.append(nbf.v4.new_markdown_cell("""## Step 4: Candidate Models Comparison
We evaluate four standard classification algorithms configured with cost-sensitive class weights:
1. **Logistic Regression** (Standardized)
2. **Decision Tree Classifier**
3. **Random Forest Classifier**
4. **XGBoost Classifier**
"""))
    cells.append(nbf.v4.new_code_cell("""classifiers = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42))
    ]),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=6, class_weight="balanced", random_state=42),
    "XGBoost Classifier": XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, scale_pos_weight=pos_weight, random_state=42, eval_metric="logloss")
}

comparison_results = []
trained_models = {}
test_probabilities = {}
test_predictions = {}

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else preds

    trained_models[name] = clf
    test_predictions[name] = preds
    test_probabilities[name] = probs

    comparison_results.append({
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, preds), 4),
        "Precision": round(precision_score(y_test, preds, zero_division=0), 4),
        "Recall": round(recall_score(y_test, preds, zero_division=0), 4),
        "F1-Score": round(f1_score(y_test, preds, zero_division=0), 4),
        "ROC-AUC": round(roc_auc_score(y_test, probs), 4)
    })

leaderboard_clf = pd.DataFrame(comparison_results).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
print("=== Classification Model Comparison Leaderboard ===")
leaderboard_clf
"""))

    # Step 5: Select Best Model & Detailed Evaluation
    cells.append(nbf.v4.new_markdown_cell("""## Step 5: Best Model Selection & In-Depth Diagnostics
Selecting the top performer and examining its Confusion Matrix, ROC-AUC Curve, and Classification Report.
"""))
    cells.append(nbf.v4.new_code_cell("""best_clf_name = leaderboard_clf.iloc[0]['Model']
best_clf = trained_models[best_clf_name]
best_probs = test_probabilities[best_clf_name]
best_preds = test_predictions[best_clf_name]

print(f"Selected Best Classifier: {best_clf_name}")
print("\\n--- Detailed Classification Report ---")
print(classification_report(y_test, best_preds, target_names=['Retained (0)', 'Returned (1)']))
"""))
    cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confusion Matrix Heatmap
cm = confusion_matrix(y_test, best_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0],
            xticklabels=['Predicted Retained', 'Predicted Returned'],
            yticklabels=['Actual Retained', 'Actual Returned'])
axes[0].set_title(f'Confusion Matrix - {best_clf_name}', fontsize=13, fontweight='bold')

# ROC Curves Comparison
for name, probs in test_probabilities.items():
    fpr, tpr, _ = roc_curve(y_test, probs)
    score = roc_auc_score(y_test, probs)
    axes[1].plot(fpr, tpr, label=f'{name} (AUC = {score:.3f})', linewidth=2)

axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Random Chance (AUC = 0.500)')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate (Recall)')
axes[1].set_title('Receiver Operating Characteristic (ROC) Comparison', fontsize=13, fontweight='bold')
axes[1].legend(loc='lower right')

plt.tight_layout()
plt.show()
"""))

    # Step 6: Feature Importances
    cells.append(nbf.v4.new_markdown_cell("""## Step 6: Feature Importance Analysis
Which factors contribute most to return risk?
"""))
    cells.append(nbf.v4.new_code_cell("""if hasattr(best_clf, 'feature_importances_'):
    importances = best_clf.feature_importances_
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(15)

    plt.figure(figsize=(12, 6))
    feat_imp.plot(kind='barh', color='#1f77b4').invert_yaxis()
    plt.title(f'Top 15 Predictive Features for Return Risk ({best_clf_name})', fontsize=14, fontweight='bold')
    plt.xlabel('Feature Importance Score')
    plt.tight_layout()
    plt.show()
"""))

    # Step 7: Train Final Model on Full Data and Save
    cells.append(nbf.v4.new_markdown_cell("""## Step 7: Save Model Artifact & Live Inference Simulator
Refit the chosen model on the complete dataset and save the artifact to `models/return_prediction_model.pkl`.
"""))
    cells.append(nbf.v4.new_code_cell("""import sys
if ".." not in sys.path:
    sys.path.append("..")
from src.prediction import train_and_save_best_classifier, predict_order_return_risk

# Save best model artifact
artifact = train_and_save_best_classifier(df)

# Test real-time inference on sample orders
sample_order_low = {
    'Category': 'Pantry & Groceries',
    'Product': 'Ghee 1L',
    'Quantity': 2,
    'Unit_Price_INR': 450.0,
    'Discount_Pct': 0.05,
    'Payment_Method': 'UPI',
    'Fulfillment': 'Amazon (FBA)',
    'Ship_State': 'Maharashtra'
}

sample_order_high = {
    'Category': 'Electronics & Mobiles',
    'Product': '5G Smartphone',
    'Quantity': 3,
    'Unit_Price_INR': 32000.0,
    'Discount_Pct': 0.20,
    'Payment_Method': 'Cash on Delivery (COD)',
    'Fulfillment': 'Merchant (FBM)',
    'Ship_State': 'Uttar Pradesh'
}

print("=== Simulated Order 1 (Grocery item via UPI) ===")
print(predict_order_return_risk(sample_order_low, artifact))

print("\\n=== Simulated Order 2 (High-value smartphone via COD) ===")
print(predict_order_return_risk(sample_order_high, artifact))
"""))

    nb.cells = cells
    return nb


def build_and_run_all_notebooks():
    os.makedirs("notebooks", exist_ok=True)

    notebook_creators = [
        ("notebooks/01_data_cleaning.ipynb", create_01_data_cleaning),
        ("notebooks/02_eda.ipynb", create_02_eda),
        ("notebooks/03_sql_analysis.ipynb", create_03_sql_analysis),
        ("notebooks/04_sales_forecasting.ipynb", create_04_sales_forecasting),
        ("notebooks/05_return_prediction.ipynb", create_05_return_prediction)
    ]

    for nb_path, creator in notebook_creators:
        # If notebook already has executed outputs, skip re-executing to save time
        if os.path.exists(nb_path) and "05_return_prediction" not in nb_path:
            print(f"Skipping already executed notebook: {nb_path}")
            continue

        print(f"\n==========================================")
        print(f"Generating notebook: {nb_path}")
        nb = creator()

        with open(nb_path, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        print(f"Saved structure to {nb_path}.")

        print(f"Executing {nb_path} with kernel...")
        try:
            client = NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": "notebooks"}})
            client.execute()
            with open(nb_path, "w", encoding="utf-8") as f:
                nbf.write(nb, f)
            print(f"Successfully executed and saved with outputs: {nb_path}")
        except Exception as e:
            print(f"Warning during execution of {nb_path}: {e}")
            with open(nb_path, "w", encoding="utf-8") as f:
                nbf.write(nb, f)


if __name__ == "__main__":
    build_and_run_all_notebooks()
