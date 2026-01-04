import logging
import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.sklearn
from feast import FeatureStore
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from typing import Tuple, Dict, Any, Optional

from src.config import settings

logger = logging.getLogger(__name__)

def load_data(path: str) -> pd.DataFrame:
    """Load data from parquet and enrich with features from Feast."""
    logger.info(f"Loading entity data from {path}")
    
    # Load only entity columns needed for Feast
    entity_df = pd.read_parquet(
        path,
        columns=["trip_id", "pickup_datetime", settings.TARGET]
    )
    
    logger.info("Fetching historical features from Feast...")
    store = FeatureStore(repo_path=settings.FEAST_REPO_PATH)
    
    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=settings.FEAST_FEATURES,
    ).to_df()
    
    return training_df

def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Prepare features and target, split into train/val."""
    
    # Sort by time for temporal split
    if "pickup_datetime" in df.columns:
        df = df.sort_values("pickup_datetime")
    
    X = df[settings.FEATURES]
    y = df[settings.TARGET]
    
    # Temporal split 80/20
    split_idx = int(len(df) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    
    logger.info(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")
    return X_train, y_train, X_val, y_val

def train_model(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    X_val: pd.DataFrame, 
    y_val: pd.Series,
    params: Dict[str, Any]
) -> xgb.XGBRegressor:
    """Train XGBoost model with early stopping."""
    logger.info("Training XGBoost model...")
    
    # Extract early stopping rounds if present
    early_stopping_rounds = params.pop("early_stopping_rounds", None)
    
    model = xgb.XGBRegressor(**params)
    
    if early_stopping_rounds:
        logger.info(f"Training with early stopping (rounds={early_stopping_rounds})...")
        model.set_params(early_stopping_rounds=early_stopping_rounds)
        model.fit(
            X_train, 
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
    else:
        model.fit(X_train, y_train)
        
    return model

def hyperparameter_tuning(
    X_train: pd.DataFrame, 
    y_train: pd.Series
) -> Dict[str, Any]:
    """Find best hyperparameters using RandomizedSearchCV on a subset of data."""
    logger.info("Starting hyperparameter tuning...")
    
    # Subsample data for faster tuning
    fraction = settings.TUNING_CONFIG.get("tuning_data_fraction", 1.0)
    if fraction < 1.0:
        logger.info(f"Subsampling {fraction*100}% of data for tuning...")
        sample_idx = X_train.sample(frac=fraction, random_state=settings.TUNING_CONFIG["random_state"]).index
        X_tune = X_train.loc[sample_idx]
        y_tune = y_train.loc[sample_idx]
    else:
        X_tune, y_tune = X_train, y_train

    estimator = xgb.XGBRegressor(objective="reg:squarederror", random_state=settings.TUNING_CONFIG["random_state"])
    
    random_search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=settings.XGB_TUNING_SPACE,
        n_iter=settings.TUNING_CONFIG["n_iter"],
        cv=settings.TUNING_CONFIG["cv_folds"],
        scoring="neg_root_mean_squared_error",
        verbose=1,
        random_state=settings.TUNING_CONFIG["random_state"],
        n_jobs=-1
    )
    
    random_search.fit(X_tune, y_tune)
    
    logger.info(f"Best parameters found: {random_search.best_params_}")
    logger.info(f"Best RMSE (CV): {-random_search.best_score_:.2f}")
    
    return random_search.best_params_

def evaluate_model(model: xgb.XGBRegressor, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, float]:
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_val)
    rmse = root_mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    
    logger.info(f"RMSE: {rmse:.2f}")
    logger.info(f"MAE: {mae:.2f}")
    
    return {"rmse": rmse, "mae": mae}

def run_training(tune: bool = False):
    """Main training pipeline."""
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)
    
    with mlflow.start_run():
        # Load and prepare
        df = load_data(settings.PROCESSED_DATA_PATH)
        X_train, y_train, X_val, y_val = prepare_data(df)
        
        # Hyperparameter Tuning
        params = settings.XGB_PARAMS.copy()
        if tune:
            best_params = hyperparameter_tuning(X_train, y_train)
            params.update(best_params)
            mlflow.log_params({f"tuned_{k}": v for k, v in best_params.items()})
        
        # Train
        model = train_model(X_train, y_train, X_val, y_val, params)
        
        # Evaluate
        metrics = evaluate_model(model, X_val, y_val)
        
        # Logging
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="model")
        
        logger.info("Model logged successfully to MLflow.")