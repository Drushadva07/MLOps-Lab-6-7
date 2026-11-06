
import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field


# Create FastAPI application
app = FastAPI(
    title="House Price Prediction API",
    description="API for predicting house prices using Machine Learning",
    version="1.0"
)


# Load the trained model
model = joblib.load("models/house_price_pipeline.pkl")


# Define input format
class HouseInput(BaseModel):
    area_sqft: float = Field(gt=0)
    bedrooms: int = Field(gt=0)
    age_years: int = Field(ge=0)
    location_score: float = Field(ge=0, le=10)


# Health check endpoint
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "message": "House Price Prediction API is running"
    }


# Prediction endpoint
@app.post("/predict")
def predict_price(house: HouseInput):

    input_data = pd.DataFrame([{
        "area_sqft": house.area_sqft,
        "bedrooms": house.bedrooms,
        "age_years": house.age_years,
        "location_score": house.location_score
    }])

    prediction = model.predict(input_data)

    return {
        "predicted_price": round(float(prediction[0]), 2)
    }
