
import json
from pathlib import Path

import joblib
import yaml
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.data import load_data, split_data


def evaluate_model(config_path="configs/config.yaml"):

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    df = load_data(config["data"]["path"])

    X_train, X_test, y_train, y_test = split_data(
        df,
        config["data"]["target"],
        config["data"]["test_size"],
        config["data"]["random_state"]
    )

    model = joblib.load(config["output"]["model_path"])

    predictions = model.predict(X_test)

    results = {
        "MAE": float(mean_absolute_error(y_test, predictions)),
        "RMSE": float(mean_squared_error(y_test, predictions) ** 0.5),
        "R2_Score": float(r2_score(y_test, predictions))
    }

    report_path = Path(config["output"]["report_path"])
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w") as file:
        json.dump(results, file, indent=4)

    print("Evaluation completed!")
    print(json.dumps(results, indent=4))
    print("Report saved at:", report_path)


if __name__ == "__main__":
    evaluate_model()
