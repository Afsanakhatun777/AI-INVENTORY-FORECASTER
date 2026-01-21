from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("src/forecaster/model.joblib")

@app.get("/")
def home():
    return {"message": "Inventory Forecaster API is Online"}

@app.get("/predict")
def predict(day: int, month: int, weekend: int, price: float, lag: float, rolling: float):
    # Prepare the data
    data = pd.DataFrame([[day, month, weekend, price, lag, rolling]], 
                        columns=['day_of_week', 'month', 'is_weekend', 'UnitPrice', 'quantity_lag_7', 'rolling_mean_7'])
    
    # Predict
    prediction = model.predict(data)[0]
    
    return {
        "predicted_inventory_needed": round(prediction, 2),
        "status": "success"
    }