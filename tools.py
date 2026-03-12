"""Small, safe dataframe analysis functions used by the LangChain agent."""

from typing import Any

from langchain_core.tools import BaseTool, tool
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


def create_dataframe_tools(dataframe: pd.DataFrame) -> list[BaseTool]:
    """Build LangChain tools around the current DataFrame."""

    @tool("list_columns")
    def list_columns_tool() -> str:
        """Return all column names in the loaded dataset."""
        return ", ".join(list_columns(dataframe))

    @tool("show_data_preview")
    def show_data_preview_tool(rows: int = 5) -> str:
        """Show the first few rows from the loaded dataset."""
        return show_data_preview(dataframe, rows)

    @tool("summarize_dataset")
    def summarize_dataset_tool() -> str:
        """Summarize the dataset, including shape and column types."""
        summary = summarize_dataset(dataframe)
        return _format_dataset_summary(summary)

    @tool("count_missing_values")
    def count_missing_values_tool() -> str:
        """Count missing values in each column."""
        missing_values = count_missing_values(dataframe)
        return _format_missing_values(missing_values)

    @tool("group_and_aggregate")
    def group_and_aggregate_tool(
        group_by_column: str,
        value_column: str,
        agg_func: str = "mean",
    ) -> str:
        """Group rows by one column and aggregate a numeric column."""
        result = group_and_aggregate(dataframe, group_by_column, value_column, agg_func)
        return result.to_string(index=False)

    @tool("top_n_rows")
    def top_n_rows_tool(
        sort_by: str,
        n: int = 5,
        ascending: bool = False,
    ) -> str:
        """Sort the dataset by one column and return the top rows."""
        result = top_n_rows(dataframe, sort_by, n, ascending)
        return result.to_string(index=False)

    @tool("basic_correlation_check")
    def basic_correlation_check_tool(first_column: str, second_column: str) -> str:
        """Calculate a simple Pearson correlation between two numeric columns."""
        correlation = basic_correlation_check(dataframe, first_column, second_column)
        return _format_correlation(first_column, second_column, correlation)

    return [
        list_columns_tool,
        show_data_preview_tool,
        summarize_dataset_tool,
        count_missing_values_tool,
        group_and_aggregate_tool,
        top_n_rows_tool,
        basic_correlation_check_tool,
    ]


def _format_dataset_summary(summary: dict[str, Any]) -> str:
    """Convert dataset metadata into a readable multi-line string."""
    lines = [
        f"Rows: {summary['row_count']}",
        f"Columns: {summary['column_count']}",
        f"All columns: {', '.join(summary['columns'])}",
        f"Numeric columns: {', '.join(summary['numeric_columns'])}",
        f"Categorical columns: {', '.join(summary['categorical_columns'])}",
    ]
    return "\n".join(lines)


def _format_missing_values(missing_values: dict[str, int]) -> str:
    """Convert missing-value counts into a readable multi-line string."""
    lines = [f"{column}: {count}" for column, count in missing_values.items()]
    return "\n".join(lines)


def _format_correlation(
    first_column: str,
    second_column: str,
    correlation: float,
) -> str:
    """Convert a numeric correlation value into a readable sentence."""
    return (
        f"Correlation between {first_column} and {second_column}: "
        f"{correlation:.3f}"
    )


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
