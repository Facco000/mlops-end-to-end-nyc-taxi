from feast import FileSource
from pathlib import Path

DATA_PATH = (
    Path(__file__)
    .resolve()
    .parents[4]  # risale fino alla root del progetto
    / "data/processed/taxi_ml_2015_01.parquet"
)

taxi_trip_source = FileSource(
    name="taxi_trip_source",
    path=str(DATA_PATH),
    event_timestamp_column="pickup_datetime",
)
