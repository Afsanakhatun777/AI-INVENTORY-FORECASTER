import pandas as pd
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_features(input_path="Data/cleaned_retail.csv", output_path="Data/features_retail.csv"):
    if not os.path.exists(input_path):
        logging.error(f"File not found: {input_path}. Did you run ingestion.py first?")
        return

    logging.info("Starting Feature Engineering...")
    
    # 1. Load Data
    df = pd.read_csv(input_path, parse_dates=['InvoiceDate'])
    
    # 2. Daily Aggregation: Group by Day and Product
    df['Date'] = df['InvoiceDate'].dt.date
    daily_data = df.groupby(['Date', 'StockCode']).agg({
        'Quantity': 'sum',
        'UnitPrice': 'mean'
    }).reset_index()
    
    # 3. Time-Based Features: Extract patterns
    daily_data['Date'] = pd.to_datetime(daily_data['Date'])
    daily_data['day_of_week'] = daily_data['Date'].dt.dayofweek
    daily_data['month'] = daily_data['Date'].dt.month
    daily_data['is_weekend'] = daily_data['day_of_week'].isin([5, 6]).astype(int)
    
    # 3. Lag Features (The AI's Memory)
    # We sort by date to ensure the "shift" grabs the correct previous day
    daily_data = daily_data.sort_values(['StockCode', 'Date'])
    
    # Create a 7-day lag (Sales from exactly 1 week ago)
    daily_data['quantity_lag_7'] = daily_data.groupby('StockCode')['Quantity'].shift(7)
    
    # 4. Rolling Average (Smooths out random spikes)
    # Average sales over the last 7 days
    daily_data['rolling_mean_7'] = daily_data.groupby('StockCode')['Quantity'].transform(
        lambda x: x.rolling(window=7).mean()
    )

    # 5. Drop the NaN rows created by the lags (the first 7 days of data)
    daily_data = daily_data.dropna()
    
    # 4. Save Features
    daily_data.to_csv(output_path, index=False)
    logging.info(f"Feature engineering complete. New shape: {daily_data.shape}")

if __name__ == "__main__":
    create_features()