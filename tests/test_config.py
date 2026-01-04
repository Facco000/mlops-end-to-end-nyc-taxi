from src.config import settings

def test_settings_load():
    assert settings.RAW_DATA_FILE == "yellow_tripdata_2015-01.parquet"
    assert settings.TARGET == "trip_duration_seconds"
    assert len(settings.FEATURES) > 0
    assert "trip_distance" in settings.FEATURES
