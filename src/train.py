
from pathlib import Path
import yaml
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data import load_data, validate_data, split_data


def train_model(config_path="configs/config.yaml"):

    # Load configuration
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Read dataset
    df = load_data(config["data"]["path"])

    # Validate dataset
    validate_data(df, config["data"]["target"])

    # Split dataset
    X_train, X_test, y_train, y_test = split_data(
        df,
        config["data"]["target"],
        config["data"]["test_size"],
        config["data"]["random_state"]
    )

    # Create complete ML pipeline
    model_pipeline = Pipeline([
        ("preprocessing", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=config["model"]["n_estimators"],
            random_state=config["model"]["random_state"]
        ))
    ])

    # Train the model
    model_pipeline.fit(X_train, y_train)

    # Evaluate on test data
    predictions = model_pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)

    # Save trained pipeline
    model_path = Path(config["output"]["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model_pipeline, model_path)

    print("Model training completed successfully!")
    print("Model saved at:", model_path)
    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))
    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2 Score:", round(r2, 4))


if __name__ == "__main__":
    train_model()
