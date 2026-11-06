
import joblib
import pandas as pd


MODEL_PATH = "models/house_price_pipeline.pkl"


def predict_price(area_sqft, bedrooms, age_years, location_score):

    # Load the saved model
    model = joblib.load(MODEL_PATH)

    # Prepare input data
    input_data = pd.DataFrame([{
        "area_sqft": area_sqft,
        "bedrooms": bedrooms,
        "age_years": age_years,
        "location_score": location_score
    }])

    # Make prediction
    prediction = model.predict(input_data)

    return float(prediction[0])


if __name__ == "__main__":

    houses = [
        (850, 2, 15, 6),
        (1000, 2, 10, 7),
        (1200, 3, 8, 8),
        (1800, 3, 5, 9),
        (2200, 4, 3, 10)
    ]

    print("HOUSE PRICE PREDICTIONS")
    print("----------------------")

    for house in houses:

        price = predict_price(*house)

        print(
            f"Area: {house[0]} sqft | "
            f"Bedrooms: {house[1]} | "
            f"Predicted Price: {price:.2f}"
        )
