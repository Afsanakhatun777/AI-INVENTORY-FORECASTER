
# AI-Powered Inventory Forecaster 

A professional-grade inventory management system that uses Machine Learning to predict stock demand and automate restock reporting.

## Key Features
* Predictive Analytics: Uses a Random Forest Regressor to forecast demand based on historical trends, pricing, and seasonality.
* Bulk Analysis Engine: One-click analysis of the entire warehouse to identify understocking risks.
* Smart UX: High-contrast, color-coded dashboard for instant decision-making (Critical vs. Warning levels).
* Exportable Reports: Download AI-generated Buy Lists directly to CSV for procurement teams.
* Model Transparency: Integrated Model Health section showing MAE and Feature Importance.

## Tech Stack
* Language: Python 3.9+
* AI/ML: Scikit-Learn (Random Forest), Pandas, NumPy
* Dashboard: Streamlit
* Backend: FastAPI (for model serving)
* Environment: Virtualenv

## How to Run
1. Setup Environment:
   python -m venv venv
   source venv/Scripts/activate
   pip install -r requirements.txt

2. Train the Model:
python src/forecaster/train.py

3. Launch the Dashboard:
streamlit run src/forecaster/app.py
