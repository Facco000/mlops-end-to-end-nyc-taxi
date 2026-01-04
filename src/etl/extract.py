from pathlib import Path
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def extract_parquet(path: Path) -> pd.DataFrame:
    """
    Read a Parquet file into a pandas DataFrame.

    Args:
        path (Path): Path to the Parquet file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    logger.info(f"Reading raw data from {path}")
    try:
        df = pd.read_parquet(path)
        logger.info(f"Extracted {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"Failed to extract data from {path}: {e}")
        raise

