from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Project structure
    PROJECT_ROOT: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    
    # Data Paths
    DATA_DIR: Path = Field(default_factory=lambda: Path("data"))
    RAW_DATA_FILE: str = "yellow_tripdata_2015-01.parquet"
    PROCESSED_DATA_FILE: str = "taxi_ml_2015_01.parquet"
    
    @property
    def RAW_DATA_PATH(self) -> Path:
        return self.PROJECT_ROOT / self.DATA_DIR / "raw" / self.RAW_DATA_FILE
        
    @property
    def PROCESSED_DATA_PATH(self) -> Path:
        return self.PROJECT_ROOT / self.DATA_DIR / "processed" / self.PROCESSED_DATA_FILE

    # Model Parameters
    TARGET: str = "trip_duration_seconds"
    FEATURES: list[str] = [
        "haversine_m",
        "trip_distance",
        "fare_amount",
        "passenger_count",
        "hour",
        "day_of_week",
        "is_weekend",
        "pickup_latitude",
        "pickup_longitude",
        "dropoff_latitude",
        "dropoff_longitude",
    ]
    
    # MLflow
    MLFLOW_EXPERIMENT_NAME: str = "taxi_trip_duration_xgboost"
    
    # Feast
    FEAST_REPO_PATH: str = "feature_repo/taxi_features"
    FEAST_FEATURES: list[str] = [
        "taxi_trip_features:haversine_m",
        "taxi_trip_features:trip_distance",
        "taxi_trip_features:fare_amount",
        "taxi_trip_features:passenger_count",
        "taxi_trip_features:hour",
        "taxi_trip_features:day_of_week",
        "taxi_trip_features:is_weekend",
        "taxi_trip_features:pickup_latitude",
        "taxi_trip_features:pickup_longitude",
        "taxi_trip_features:dropoff_latitude",
        "taxi_trip_features:dropoff_longitude",
    ]

    # XGBoost Params (default values)
    XGB_PARAMS: dict = {
        "objective": "reg:squarederror",
        "n_estimators": 500,  # Increased default n_estimators
        "max_depth": 6,
        "learning_rate": 0.05,  # Reduced default learning rate
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "early_stopping_rounds": 50,  # Added early stopping
    }

    # Hyperparameter search space for tuning
    XGB_TUNING_SPACE: dict = {
        "n_estimators": [100, 300, 500, 800, 1000],
        "max_depth": [3, 5, 7, 9],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.7, 0.8, 0.9],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
        "gamma": [0, 0.1, 0.2, 0.4],
        "min_child_weight": [1, 3, 5, 7],
    }

    # Tuning configuration
    TUNING_CONFIG: dict = {
        "n_iter": 5,           # Reduced from 20 for speed
        "cv_folds": 2,         # Reduced from 3 for speed
        "random_state": 42,
        "tuning_data_fraction": 0.1,  # Use 10% of data for tuning
    }

settings = Settings()
