"""
Database Population Script
Populate SQLite database from processed Amazon India Sales data
"""

import sqlite3
import pandas as pd
import os

def create_and_populate_db(csv_path: str = "data/processed/amazon_sales_cleaned.csv", db_path: str = "database/amazon_sales.db"):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    df = pd.read_csv(csv_path)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Store table
    df.to_sql("amazon_sales", conn, if_exists="replace", index=False)
    
    # Create indexes for high query performance
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_order_date ON amazon_sales(Order_Date);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_category ON amazon_sales(Category);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON amazon_sales(Order_Status);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_state ON amazon_sales(Ship_State);")
    
    conn.commit()
    
    # Test query
    cursor.execute("SELECT COUNT(*) FROM amazon_sales;")
    count = cursor.fetchone()[0]
    print(f"Successfully populated SQLite database '{db_path}' with {count:,} rows in table 'amazon_sales'.")
    conn.close()

if __name__ == "__main__":
    create_and_populate_db()
