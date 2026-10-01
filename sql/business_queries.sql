-- ==============================================================================
-- AMAZON INDIA SALES ANALYTICS & BUSINESS INTELLIGENCE
-- Production SQL Business Queries for SQLite / MySQL
-- ==============================================================================

-- ------------------------------------------------------------------------------
-- QUERY 1: Executive KPI Overview (Slides 1 & 2)
-- Total Sales, Total Profit, Total Orders, Units Sold, AOV, Profit Margin %
-- ------------------------------------------------------------------------------
SELECT 
    COUNT(DISTINCT Order_ID) AS total_orders,
    SUM(Quantity) AS total_units_sold,
    ROUND(SUM(Total_Sales_INR), 2) AS total_sales_inr,
    ROUND(SUM(Profit_INR), 2) AS total_profit_inr,
    ROUND(SUM(Total_Sales_INR) / COUNT(DISTINCT Order_ID), 2) AS average_order_value_inr,
    ROUND((SUM(Profit_INR) / SUM(Total_Sales_INR)) * 100, 2) AS profit_margin_pct,
    ROUND(SUM(CASE WHEN Order_Status IN ('Returned', 'Cancelled') THEN Total_Sales_INR ELSE 0 END), 2) AS total_revenue_lost_inr
FROM amazon_sales;


-- ------------------------------------------------------------------------------
-- QUERY 2: Monthly Sales & Profit Trend (Slide 2)
-- Month-on-Month Revenue, Profit, Order Count, and Margin %
-- ------------------------------------------------------------------------------
SELECT 
    Year_Month,
    COUNT(DISTINCT Order_ID) AS orders_count,
    ROUND(SUM(Total_Sales_INR), 2) AS monthly_sales_inr,
    ROUND(SUM(Profit_INR), 2) AS monthly_profit_inr,
    ROUND((SUM(Profit_INR) / SUM(Total_Sales_INR)) * 100, 2) AS profit_margin_pct
FROM amazon_sales
GROUP BY Year_Month
ORDER BY Year_Month ASC;


-- ------------------------------------------------------------------------------
-- QUERY 3: Category Performance & Ranking (Slide 3)
-- Category-wise Sales, Profit, Units, Margin, and Sales Rank
-- ------------------------------------------------------------------------------
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


-- ------------------------------------------------------------------------------
-- QUERY 4: Top 10 Best-Selling Products by Revenue (Slide 3)
-- ------------------------------------------------------------------------------
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


-- ------------------------------------------------------------------------------
-- QUERY 5: Order Status Breakdown & Revenue Loss Analysis (Slide 4)
-- Delivered, Shipped, Returned, Cancelled with percentages and revenue lost
-- ------------------------------------------------------------------------------
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


-- ------------------------------------------------------------------------------
-- QUERY 6: Category-Wise Return & Cancellation Rates (Slide 4)
-- ------------------------------------------------------------------------------
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


-- ------------------------------------------------------------------------------
-- QUERY 7: Payment Method Performance & Behavior (Slide 5)
-- Sales, Profit, AOV, and Return Rate by Payment Channel
-- ------------------------------------------------------------------------------
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


-- ------------------------------------------------------------------------------
-- QUERY 8: Fulfillment Channel Efficiency (Slide 5)
-- Amazon (FBA) vs Seller Flex vs Merchant (FBM)
-- ------------------------------------------------------------------------------
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


-- ------------------------------------------------------------------------------
-- QUERY 9: Geographic / State-Wise Performance (Slide 5)
-- Top Revenue Generating States with Orders, Profit, and AOV
-- ------------------------------------------------------------------------------
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


-- ------------------------------------------------------------------------------
-- QUERY 10: High-Value VIP Orders Analysis
-- Orders above 90th percentile of total sales
-- ------------------------------------------------------------------------------
SELECT 
    Order_ID,
    Order_Date,
    Category,
    Product,
    Quantity,
    Total_Sales_INR,
    Profit_INR,
    Payment_Method,
    Ship_State,
    Order_Status
FROM amazon_sales
WHERE Total_Sales_INR >= 75000
ORDER BY Total_Sales_INR DESC
LIMIT 15;
