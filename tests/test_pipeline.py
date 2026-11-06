
from pathlib import Path

from src.data import load_data, validate_data, split_data


def test_dataset_loading():
    df = load_data("data/raw/dataset.csv")

    assert not df.empty
    assert len(df) == 33


def test_data_validation():
    df = load_data("data/raw/dataset.csv")

    assert validate_data(df, "price") is True


def test_data_splitting():
    df = load_data("data/raw/dataset.csv")

    X_train, X_test, y_train, y_test = split_data(
        df, "price", test_size=0.2, random_state=42
    )

    assert len(X_train) == 26
    assert len(X_test) == 7
    assert len(y_train) == 26
    assert len(y_test) == 7


def test_saved_model_exists():
    assert Path("models/house_price_pipeline.pkl").exists()
