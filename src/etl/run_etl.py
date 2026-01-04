import logging
from src.etl.extract import extract_parquet as extract
from src.etl.transform import transform
from src.etl.load import load_parquet as load
from src.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting ETL pipeline...")

    try:
        # Extract
        df_raw = extract(settings.RAW_DATA_PATH)
        logger.info(f"Raw data shape: {df_raw.shape}")

        # Transform
        df_processed = transform(df_raw)
        logger.info(f"Processed data shape: {df_processed.shape}")

        # Load
        load(df_processed, settings.PROCESSED_DATA_PATH)
        logger.info("ETL pipeline completed successfully.")

    except Exception as e:
        logger.error(f"ETL Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
