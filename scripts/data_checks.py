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
        "customer_id": Column(str, nullable=False),
        "Name": Column(str, nullable=False),
        "age": Column(int, Check.le(120), Check.ge(0), coerce=True),
        "gender": Column(str, Check.isin(["M", "F"]), coerce=True, nullable=True),
        "security_no": Column(str, nullable=False),
        "region_category": Column(str, nullable=True),
        "membership_category": Column(str, nullable=False),
        "joining_date": Column(str, nullable=False),
        "joined_through_referral": Column(str, Check.isin(["Yes", "No", "?"]), nullable=True),
        "referral_id": Column(str, nullable=True),
        "preferred_offer_types": Column(str, nullable=False),
        "medium_of_operation": Column(str, nullable=True),
        "internet_option": Column(str, nullable=False),
        "last_visit_time": Column(str, nullable=False),
        "days_since_last_login": Column(int, Check.ge(0), coerce=True, nullable=True),
        "avg_time_spent": Column(float, coerce=True, nullable=True),
        "avg_transaction_value": Column(float, coerce=True, nullable=True),
        "avg_frequency_login_days": Column(float, Check.ge(0), coerce=True, nullable=True),
        "points_in_wallet": Column(float, Check.ge(0), coerce=True, nullable=True),
        "used_special_discount": Column(str, Check.isin(["Yes", "No"]), nullable=False),
        "offer_application_preference": Column(str, Check.isin(["Yes", "No"]), nullable=False),
        "past_complaint": Column(str, Check.isin(["Yes", "No"]), nullable=False),
        "complaint_status": Column(str, nullable=False),
        "feedback": Column(str, nullable=False),
        "churn_risk_score": Column(int, Check.ge(0), Check.le(5), coerce=True, nullable=True),
    }
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

    # Clean up the 'churn_risk_score' column, which may have negative values.
    df["churn_risk_score"] = df["churn_risk_score"].replace({-1: None})

    # In Some Cases, avg_time_spent might have negative values, which should be handled.
    df["avg_time_spent"] = df["avg_time_spent"].apply(lambda x: x if x >= 0 else None)

    # The 'avg_frequency_login_days' column contains an 'Error' string, which needs to be handled.
    df["avg_frequency_login_days"] = pd.to_numeric(df["avg_frequency_login_days"], errors="coerce")

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
    assert df["customer_id"].is_unique, "CustomerID column must be unique."

    # Check for a realistic average churn rate
    avg_churn_score = df["churn_risk_score"].mean()
    assert 1.0 <= avg_churn_score <= 4.0, (
        f"Average churn risk score out of bounds: {avg_churn_score}. Expected between 1.0 and 4.0."
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
