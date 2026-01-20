import pandas as pd
import logging
import os

# Set up logging to track our "SaaS" health
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    """Loads the raw retail CSV."""
    if not os.path.exists(file_path):
        logging.error(f"File not found at {file_path}")
        return None
    logging.info(f"Loading data from {file_path}...") 
    # Note: Use 'unicode_escape' or 'latin1' because retail data often has weird characters
    return pd.read_csv(file_path, encoding='unicode_escape') 

def clean_data(df):
    """The core logic to clean 500k rows for inventory forecasting."""
    logging.info("Starting data cleaning...")
    
    # 1. Drop missing CustomerIDs (We can't track behavior without them)
    df = df.dropna(subset=['CustomerID'])
    
    # 2. Remove Cancellations (Invoice starts with 'C')
    # These represent returns/cancellations which mess up demand forecasting
    df = df[~df['InvoiceNo'].str.startswith('C', na=False)]
    
    # 3. Ensure Quantity and UnitPrice are positive
    # Some rows have negative prices for adjustments—we don't want those
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    
    # 4. Convert Date to actual DateTime object
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], dayfirst=True, errors='coerce')

# Drop any rows where the date couldn't be parsed
    df = df.dropna(subset=['InvoiceDate'])
    
    # 5. Drop duplicates
    df = df.drop_duplicates()
    
    logging.info(f"Cleaning complete. Remaining rows: {len(df)}")
    return df

if __name__ == "__main__":
    # This block allows you to test the script individually
    RAW_DATA_PATH = "Data/Online_Retail.csv"  # Ensure your file is here!
    PROCESSED_PATH = "Data/cleaned_retail.csv"
    
    raw_df = load_data(RAW_DATA_PATH)
    if raw_df is not None:
        cleaned_df = clean_data(raw_df)
        cleaned_df.to_csv(PROCESSED_PATH, index=False)
        logging.info(f"Saved cleaned data to {PROCESSED_PATH}")

