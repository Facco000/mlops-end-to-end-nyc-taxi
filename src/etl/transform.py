import pandas as pd
import logging
from src.config import settings
import numpy as np

logger = logging.getLogger(__name__)


def parse_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Convert datetime columns to datetime objects."""
    df = df.copy()
    df["pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")
    df["dropoff_datetime"] = pd.to_datetime(
        df["tpep_dropoff_datetime"], errors="coerce"
    )
    return df


def compute_trip_duration(df: pd.DataFrame) -> pd.DataFrame:
    """Compute trip duration in seconds."""
    df = df.copy()
    df["trip_duration_seconds"] = (
        df["dropoff_datetime"] - df["pickup_datetime"]
    ).dt.total_seconds()
    return df


def filter_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Filter rows with invalid data (duration, distance, coordinates)."""
    df = df.copy()
    initial_rows = len(df)

    # Basic validity checks
    mask = (
        (df["trip_duration_seconds"] > 60)
        & (df["trip_duration_seconds"] < 3 * 60 * 60)
        & (df["trip_distance"] > 0)
        & (df["fare_amount"] > 0)
        & (df["pickup_latitude"].between(40.5, 41.0))
        & (df["pickup_longitude"].between(-74.5, -73.5))
        & (df["dropoff_latitude"].between(40.5, 41.0))
        & (df["dropoff_longitude"].between(-74.5, -73.5))
    )

    df = df[mask]

    dropped_count = initial_rows - len(df)
    dropped_pct = dropped_count / initial_rows if initial_rows > 0 else 0

    logger.info(f"Filtered rows: {dropped_count} ({dropped_pct:.2%})")

    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add temporal features like hour, day_of_week, is_weekend."""
    df = df.copy()
    df["hour"] = df["pickup_datetime"].dt.hour
    df["day_of_week"] = df["pickup_datetime"].dt.weekday
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    return df


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select only relevant features for the model."""
    # Ensure target and other necessary columns are included
    cols_to_keep = settings.FEATURES + [settings.TARGET, "pickup_datetime"]
    # Check if columns exist
    available_cols = [c for c in cols_to_keep if c in df.columns]

    missing = set(cols_to_keep) - set(available_cols)
    if missing:
        logger.warning(f"Missing columns in dataframe: {missing}")

    return df[available_cols]


def add_haversine_distance(df: pd.DataFrame) -> pd.DataFrame:
    """Compute haversine distance in meters between pickup and dropoff."""
    df = df.copy()

    R = 6371000  # Earth radius in meters

    lat1 = np.radians(df["pickup_latitude"])
    lon1 = np.radians(df["pickup_longitude"])
    lat2 = np.radians(df["dropoff_latitude"])
    lon2 = np.radians(df["dropoff_longitude"])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    df["haversine_m"] = R * c
    return df


def transform(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Starting transformation pipeline...")

    df = parse_datetime(df)
    df = compute_trip_duration(df)
    df = filter_invalid_rows(df)
    df = add_haversine_distance(df)

    df = add_temporal_features(df)

    df = select_features(df)

    df = df.reset_index(drop=True)
    df["trip_id"] = df.index.astype("int64")

    logger.info(f"Transformation complete. Result shape: {df.shape}")
    return df
