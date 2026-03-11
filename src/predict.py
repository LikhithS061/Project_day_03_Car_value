"""
predict.py — Load trained model and predict car price
=====================================================
Provides a clean prediction interface for the Streamlit app.
"""

import os
import joblib
import pandas as pd
from datetime import datetime


# Path to the saved model artifact
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "car_price_model.pkl")


def load_model(path: str = MODEL_PATH):
    """Load the trained model artifact (model + feature_names + metrics)."""
    artifact = joblib.load(path)
    return artifact


def predict_price(
    year: int,
    present_price: float,
    kms_driven: int,
    fuel_type: str,
    seller_type: str,
    transmission: str,
    owner: int,
) -> float:
    """
    Predict the selling price of a used car.

    Parameters:
        year          : Manufacturing year (e.g. 2018)
        present_price : Current ex-showroom price in lakhs
        kms_driven    : Total kilometers driven
        fuel_type     : 'Petrol', 'Diesel', or 'CNG'
        seller_type   : 'Dealer' or 'Individual'
        transmission  : 'Manual' or 'Automatic'
        owner         : Number of previous owners (0, 1, 2, 3)

    Returns:
        Predicted selling price in lakhs.
    """
    artifact = load_model()
    model = artifact["model"]
    feature_names = artifact["feature_names"]

    current_year = datetime.now().year
    car_age = current_year - year

    # Build a raw feature dict matching the preprocessing steps
    data = {
        "Present_Price": present_price,
        "Kms_Driven": kms_driven,
        "Owner": owner,
        "CarAge": car_age,
        # One-hot encoded columns (drop_first=True was used)
        # Fuel_Type: base = CNG → dummies = Fuel_Type_Diesel, Fuel_Type_Petrol
        "Fuel_Type_Diesel": 1 if fuel_type == "Diesel" else 0,
        "Fuel_Type_Petrol": 1 if fuel_type == "Petrol" else 0,
        # Seller_Type: base = Dealer → dummy = Seller_Type_Individual
        "Seller_Type_Individual": 1 if seller_type == "Individual" else 0,
        # Transmission: base = Automatic → dummy = Transmission_Manual
        "Transmission_Manual": 1 if transmission == "Manual" else 0,
    }

    # Create DataFrame with the exact feature order the model expects
    input_df = pd.DataFrame([data])

    # Ensure all expected columns exist, fill missing with 0
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder to match training feature order
    input_df = input_df[feature_names]

    prediction = model.predict(input_df)[0]
    return max(prediction, 0.0)  # Price can't be negative


if __name__ == "__main__":
    # Quick test
    price = predict_price(
        year=2017,
        present_price=9.85,
        kms_driven=6900,
        fuel_type="Petrol",
        seller_type="Dealer",
        transmission="Manual",
        owner=0,
    )
    print(f"Predicted selling price: ₹{price:.2f} lakh")
