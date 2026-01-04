from pathlib import Path
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def load_parquet(df: pd.DataFrame, path: Path) -> None:
    """
    Save DataFrame to Parquet.

    Args:
        df (pd.DataFrame): DataFrame to save.
        path (Path): Destination path.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, engine="pyarrow", index=False)
        logger.info(f"Saved processed data to {path} ({len(df)} rows)")
    except Exception as e:
        logger.error(f"Failed to save data to {path}: {e}")
        raise

