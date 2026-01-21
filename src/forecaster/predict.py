import joblib
import pandas as pd
import logging

def predict_demand(day_of_week, month, is_weekend, unit_price, lag_7, rolling_7):
    # 1. Load the trained brain
    model = joblib.load("src/forecaster/model.joblib")
    
    # 2. Create a tiny dataframe for the input
    data = pd.DataFrame([[day_of_week, month, is_weekend, unit_price, lag_7, rolling_7]], 
                        columns=['day_of_week', 'month', 'is_weekend', 'UnitPrice', 'quantity_lag_7', 'rolling_mean_7'])
    
    # 3. Predict!
    prediction = model.predict(data)[0]
    return round(prediction, 2)

if __name__ == "__main__":
    # Example: Product sells for 2.55, sold 20 last week, average is 15
    # Parameters: Monday(0), January(1), Not Weekend(0), Price(2.55), Lag7(20), Rolling7(15)
    result = predict_demand(0, 1, 0, 2.55, 20.0, 15.0)
    print(f"--- PREDICTED DEMAND: {result} units ---")