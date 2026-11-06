
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(file_path):
    """Load the house price dataset."""
    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError("Dataset is empty.")

    return df


def validate_data(df, target):
    """Check the dataset before training."""

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")

    if df.isnull().values.any():
        raise ValueError("Dataset contains missing values.")

    if not pd.api.types.is_numeric_dtype(df[target]):
        raise ValueError("Target column must contain numeric values.")

    return True


def split_data(df, target, test_size=0.2, random_state=42):
    """Separate input features and target, then split the data."""

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test
