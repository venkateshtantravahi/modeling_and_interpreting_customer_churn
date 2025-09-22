import glob
import logging
import os

import pandas as pd
from pandera import Check, Column, DataFrameSchema
from pandera.errors import SchemaErrors

# Configure logging for better visibility in a CI/CD pipeline
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define the directory where the raw data
RAW_DIR = "data/raw"

# --- Define the schema based on the fields ---
# This schema is the single source of truth for your data quality.
# It uses pandera's checks to ensure data integrity.
SCHEMA = DataFrameSchema(
    {
        "CustomerID": Column(int),
        "Age": Column(int, Check.ge(0), Check.le(120), coerce=True),
        "Gender": Column(str, nullable=True),
        "Tenure": Column(int, Check.ge(0), coerce=True),
        "Usage Frequency": Column(int, Check.ge(0), coerce=True),
        "Support Calls": Column(int, Check.ge(0), coerce=True),
        "Payment Delay": Column(int, Check.ge(0), coerce=True),
        "Subscription Type": Column(str),
        "Contract Length": Column(str),
        "Total Spend": Column(int, Check.ge(0), coerce=True, nullable=True),
        "Last Interaction": Column(int),
        "Churn": Column(object, Check.isin([0, 1, "Yes", "No", True, False]), nullable=True),
    },
    # Use lazy=True to collect all validation errors before raising an exception
    # which provides a more comprehensive report in the CI logs.
    lazy=True,
)


def _find_first_csv_path() -> str:
    """Finds the path to the first CSV file in the raw data directory."""
    paths = glob.glob(os.path.join(RAW_DIR, "*.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No CSV files found in {RAW_DIR}. "
            "Provide Kaggle secrets or place a sample CSV file locally."
        )
    return paths[0]


def _load_and_clean_data(csv_path: str) -> pd.DataFrame:
    """Loads the CSV and performs basic cleaning before validation."""
    logging.info(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Convert 'Total Spend' column to numeric, coercing errors to NaN.
    # This is a common data quality issue to address.
    df["Total Spend"] = pd.to_numeric(df["Total Spend"], errors="coerce")

    logging.info(f"Loaded DataFrame with shape: {df.shape}")
    return df


def _run_sanity_checks(df: pd.DataFrame):
    """
    Runs custom sanity checks on the DataFrame beyond schema validation.
    These checks are specific to the dataset's business logic.
    """
    logging.info("Running custom sanity checks...")

    # Check for a reasonable number of rows
    assert df.shape[0] > 100, f"Dataset too small; expected > 100 rows, but got {df.shape[0]}."

    # Check for unique CustomerID
    assert df["CustomerID"].is_unique, "CustomerID column must be unique."

    # Check for a realistic churn rate
    churn_rate = (
        df["Churn"]
        # .replace({"Yes": 1, "No": 0, True: 1, False: 0})
        .astype(float)
        .mean()
    )
    assert 0.01 <= churn_rate <= 0.99, (
        f"Churn rate out of bounds: {churn_rate}. Expected between 0.01 and 0.99."
    )
    logging.info("Custom sanity checks passed.")


def main():
    """Main function to orchestrate the data quality checks."""
    try:
        csv_path = _find_first_csv_path()
        df = _load_and_clean_data(csv_path)

        # Validate the DataFrame against the defined schema
        logging.info("Validating data against schema...")
        SCHEMA.validate(df, lazy=True)

        _run_sanity_checks(df)

        logging.info("All data quality checks passed successfully!")

    except FileNotFoundError as e:
        logging.error(f"Aborted: {e}")
        exit(1)
    except SchemaErrors as e:
        logging.error("Schema validation failed.")
        logging.error(e.failure_cases)  # Shows a detailed table of failures
        logging.error(e.message)  # A summary message
        exit(1)
    except AssertionError as e:
        logging.error(f"Sanity check failed: {e}")
        exit(1)
    except Exception as e:
        logging.error(f" An unexpected error occurred: {e}")
        exit(1)


if __name__ == "__main__":
    main()
