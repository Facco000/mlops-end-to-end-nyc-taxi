import pandas as pd
import pytest


from src.etl.transform import (
    parse_datetime,
    compute_trip_duration,
    filter_invalid_rows,
    add_temporal_features,
    add_haversine_distance,
)


@pytest.fixture
def sample_df():
    data = {
        "tpep_pickup_datetime": ["2015-01-01 00:00:00", "2015-01-01 00:05:00"],
        "tpep_dropoff_datetime": [
            "2015-01-01 00:10:00",
            "2015-01-01 00:00:00",
        ],  # 2nd is invalid (neg duration)
        "trip_distance": [2.5, 0],
        "fare_amount": [10.0, -5.0],
        "pickup_latitude": [40.7128, 42.0],  # NYC approx
        "pickup_longitude": [-74.0060, -70.0],
        "dropoff_latitude": [40.7128, 42.1],
        "dropoff_longitude": [-74.0060, -70.1],
        "passenger_count": [1, 1],
    }
    return pd.DataFrame(data)


def test_add_haversine_distance(sample_df):
    df = add_haversine_distance(sample_df)
    assert "haversine_m" in df.columns
    # Row 0 has same pickup/dropoff coords in sample_df (fixed above)
    assert df.loc[0, "haversine_m"] < 1.0  # Should be ~0


def test_parse_datetime(sample_df):
    df = parse_datetime(sample_df)
    assert "pickup_datetime" in df.columns
    assert "dropoff_datetime" in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["pickup_datetime"])


def test_compute_trip_duration(sample_df):
    df = parse_datetime(sample_df)
    df = compute_trip_duration(df)
    assert "trip_duration_seconds" in df.columns
    # First row: 10 min = 600s
    assert df.loc[0, "trip_duration_seconds"] == 600


def test_filter_invalid_rows(sample_df):
    df = parse_datetime(sample_df)
    df = compute_trip_duration(df)

    # 2nd row has neg duration and invalid coords, should be filtered
    df_filtered = filter_invalid_rows(df)

    assert len(df_filtered) == 1
    assert df_filtered.iloc[0]["trip_duration_seconds"] == 600


def test_add_temporal_features(sample_df):
    df = parse_datetime(sample_df)
    df = add_temporal_features(df)

    assert "hour" in df.columns
    assert "day_of_week" in df.columns
    assert "is_weekend" in df.columns

    # 2015-01-01 is Thursday (3)
    assert df.iloc[0]["day_of_week"] == 3
    assert df.iloc[0]["is_weekend"] == 0
