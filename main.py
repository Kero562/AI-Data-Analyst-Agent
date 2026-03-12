from pathlib import Path

from dotenv import load_dotenv

from data_loader import get_dataset_shape, load_csv


SAMPLE_CSV_PATH = Path("sample_data") / "sales_data.csv"


def print_welcome() -> None:
    """Show a simple startup message."""
    print("=" * 60)
    print("AI Data Analyst Agent")
    print("Step 3: plain pandas CSV loading")
    print("=" * 60)


def main() -> None:
    """Load environment variables, open the sample CSV, and print basic info."""
    # Load values from .env if the file exists.
    load_dotenv()
    print_welcome()

    # Start with the sample dataset so the project is easy to run locally.
    dataframe = load_csv(SAMPLE_CSV_PATH)
    row_count, column_count = get_dataset_shape(dataframe)

    print(f"Loaded file: {SAMPLE_CSV_PATH}")
    print(f"Rows: {row_count}")
    print(f"Columns: {column_count}")
    print("Column names:")

    # Print the headers on one line for a quick first look at the dataset.
    print(", ".join(dataframe.columns))


if __name__ == "__main__":
    main()
