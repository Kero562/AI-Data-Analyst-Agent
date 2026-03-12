"""Utilities for loading CSV files into pandas DataFrames."""

from pathlib import Path

import pandas as pd


def load_csv(file_path: str | Path) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame."""
    # Normalize the input so path checks work the same way everywhere.
    csv_path = Path(file_path)

    # Stop early if the file path is wrong.
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # This project only handles CSV files, so reject anything else.
    if csv_path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a CSV file, but got: {csv_path}")

    # Read the file into a DataFrame so the rest of the app can analyze it.
    dataframe = pd.read_csv(csv_path)
    return dataframe


def get_dataset_shape(dataframe: pd.DataFrame) -> tuple[int, int]:
    """Return the number of rows and columns in a DataFrame."""
    return dataframe.shape
