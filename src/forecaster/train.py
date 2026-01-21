import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model(input_path="Data/features_retail.csv", model_path="src/forecaster/model.joblib"):
    logging.info("Loading feature-engineered data...")
    df = pd.read_csv(input_path)
    
    # 1. Select Features and Target
    # We drop 'Date' and 'StockCode' because the model needs numbers, not IDs
    features = ['day_of_week', 'month', 'is_weekend', 'UnitPrice', 'quantity_lag_7', 'rolling_mean_7']
    target = 'Quantity'
    
    X = df[features]
    y = df[target]
    
    # 2. Split into Training and Testing sets (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Initialize and Train the Random Forest
    logging.info("Training Random Forest Regressor... this might take a minute.")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # 4. Evaluate
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    logging.info(f"Model Training Complete! Mean Absolute Error (MAE): {mae:.2f}")
    
    # 5. Save the brain
    joblib.dump(model, model_path)
    logging.info(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_model()