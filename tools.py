"""Small, safe dataframe analysis functions used by the LangChain agent."""

from typing import Any

import pandas as pd


ALLOWED_AGGREGATIONS = {"mean", "sum", "min", "max", "count"}


def list_columns(dataframe: pd.DataFrame) -> list[str]:
    """Return all column names in the dataset."""
    return dataframe.columns.tolist()


def show_data_preview(dataframe: pd.DataFrame, rows: int = 5) -> str:
    """Return the first few rows as a readable table."""
    preview_row_count = max(1, rows)
    return dataframe.head(preview_row_count).to_string(index=False)


def summarize_dataset(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Return basic metadata about the dataset."""
    numeric_columns = dataframe.select_dtypes(include="number").columns.tolist()
    categorical_columns = dataframe.select_dtypes(exclude="number").columns.tolist()

    return {
        "row_count": len(dataframe),
        "column_count": len(dataframe.columns),
        "columns": dataframe.columns.tolist(),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
    }


def count_missing_values(dataframe: pd.DataFrame) -> dict[str, int]:
    """Return the number of missing values in each column."""
    missing_counts = dataframe.isna().sum()
    return {column: int(count) for column, count in missing_counts.items()}


def group_and_aggregate(
    dataframe: pd.DataFrame,
    group_by_column: str,
    value_column: str,
    agg_func: str = "mean",
) -> pd.DataFrame:
    """Group the dataset and apply one simple aggregation."""
    _validate_column_exists(dataframe, group_by_column)
    _validate_column_exists(dataframe, value_column)
    _validate_numeric_column(dataframe, value_column)

    if agg_func not in ALLOWED_AGGREGATIONS:
        allowed = ", ".join(sorted(ALLOWED_AGGREGATIONS))
        raise ValueError(f"Unsupported aggregation '{agg_func}'. Use one of: {allowed}")

    grouped = (
        dataframe.groupby(group_by_column)[value_column]
        .agg(agg_func)
        .reset_index()
        .sort_values(by=value_column, ascending=False)
    )
    return grouped


def top_n_rows(
    dataframe: pd.DataFrame,
    sort_by: str,
    n: int = 5,
    ascending: bool = False,
) -> pd.DataFrame:
    """Return the top rows after sorting by one column."""
    _validate_column_exists(dataframe, sort_by)
    row_count = max(1, n)

    sorted_dataframe = dataframe.sort_values(by=sort_by, ascending=ascending)
    return sorted_dataframe.head(row_count)


def basic_correlation_check(
    dataframe: pd.DataFrame,
    first_column: str,
    second_column: str,
) -> float:
    """Return the Pearson correlation between two numeric columns."""
    _validate_column_exists(dataframe, first_column)
    _validate_column_exists(dataframe, second_column)
    _validate_numeric_column(dataframe, first_column)
    _validate_numeric_column(dataframe, second_column)

    correlation = dataframe[first_column].corr(dataframe[second_column])
    if pd.isna(correlation):
        raise ValueError(
            f"Could not calculate correlation for '{first_column}' and '{second_column}'."
        )
    return float(correlation)


def _validate_column_exists(dataframe: pd.DataFrame, column_name: str) -> None:
    """Raise an error if the requested column is not present."""
    if column_name not in dataframe.columns:
        available_columns = ", ".join(dataframe.columns)
        raise ValueError(
            f"Column '{column_name}' was not found. Available columns: {available_columns}"
        )


def _validate_numeric_column(dataframe: pd.DataFrame, column_name: str) -> None:
    """Raise an error if the requested column is not numeric."""
    if not pd.api.types.is_numeric_dtype(dataframe[column_name]):
        raise ValueError(f"Column '{column_name}' must be numeric for this operation.")
