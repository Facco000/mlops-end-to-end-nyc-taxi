#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import mlflow
import xgboost as xgb

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import settings

sns.set(style="whitegrid")
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
MAP_DIR = PROJECT_ROOT / "reports" / "maps"
FIG_DIR.mkdir(parents=True, exist_ok=True)
MAP_DIR.mkdir(parents=True, exist_ok=True)

def generate_reports():
    # Carica dataset
    data_path = settings.PROCESSED_DATA_PATH
    if not data_path.exists():
        print(f"Error: Processed data not found at {data_path}")
        return

    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)

    # Try to load the latest model from MLflow
    print("Fetching latest model from MLflow...")
    client = mlflow.tracking.MlflowClient()
    try:
        experiment = client.get_experiment_by_name(settings.MLFLOW_EXPERIMENT_NAME)
        runs = client.search_runs(experiment.experiment_id, order_by=["attributes.start_time DESC"], max_results=1)
        if not runs:
            print("No MLflow runs found. Skipping model-based plots.")
            model = None
        else:
            run_id = runs[0].info.run_id
            model_uri = f"runs:/{run_id}/model"
            model = mlflow.sklearn.load_model(model_uri)
            print(f"Loaded model from run {run_id}")
    except Exception as e:
        print(f"Could not load model from MLflow: {e}")
        model = None

    # ---------- 1) Feature importance ----------
    if model is not None:
        try:
            # XGBRegressor might need get_score if plot_importance isn't used
            fi = model.get_booster().get_score(importance_type='gain')
            fi_df = pd.DataFrame(fi.items(), columns=['feature', 'importance']).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(10,6))
            sns.barplot(x="importance", y="feature", data=fi_df.head(15))
            plt.title("Feature Importance (XGBoost - Gain)")
            plt.tight_layout()
            plt.savefig(FIG_DIR / "feature_importance.png", dpi=200)
            plt.close()
            print(f"Saved: {FIG_DIR / 'feature_importance.png'}")
        except Exception as e:
            print(f"Error generating feature importance: {e}")

    # ---------- 2) Predicted vs Actual & Residuals ----------
    if model is not None:
        try:
            sample = df.sample(n=min(5000, len(df)), random_state=42)
            X_sample = sample[settings.FEATURES]
            y_sample = sample[settings.TARGET]
            
            # Convert to DMatrix for consistency if needed, but XGBRegressor.predict works on DF
            preds = model.predict(X_sample)
            
            # Predicted vs Actual
            plt.figure(figsize=(8,8))
            max_val = max(y_sample.max(), preds.max())
            plt.scatter(y_sample, preds, alpha=0.3, s=10)
            plt.plot([0, max_val], [0, max_val], 'r--', lw=2)
            plt.xlabel("Actual Duration (s)")
            plt.ylabel("Predicted Duration (s)")
            plt.title("Predicted vs Actual Trip Duration")
            plt.tight_layout()
            plt.savefig(FIG_DIR / "predicted_vs_actual.png", dpi=200)
            plt.close()
            
            # Residuals
            residuals = y_sample - preds
            plt.figure(figsize=(10,6))
            sns.histplot(residuals, kde=True, bins=50)
            plt.axvline(x=0, color='r', linestyle='--')
            plt.title("Residuals Distribution (Actual - Predicted)")
            plt.tight_layout()
            plt.savefig(FIG_DIR / "residuals_distribution.png", dpi=200)
            plt.close()
            print("Saved predicted vs actual and residuals plots.")
        except Exception as e:
            print(f"Error generating model performance plots: {e}")

    # ---------- 3) Heatmap hour x day_of_week ----------
    if "hour" in df.columns and "day_of_week" in df.columns:
        pivot = df.pivot_table(index="hour", columns="day_of_week", values=settings.TARGET, aggfunc="median")
        plt.figure(figsize=(12,7))
        sns.heatmap(pivot, cmap="YlGnBu", annot=False)
        plt.title("Median Trip Duration by Hour and Day of Week")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "heatmap_hour_day.png", dpi=200)
        plt.close()
        print(f"Saved: {FIG_DIR / 'heatmap_hour_day.png'}")

    # ---------- 4) Pickup sample map (folium) ----------
    try:
        import folium
        print("Generating interactive map...")
        sample_map = df.sample(n=min(1000, len(df)), random_state=42)
        m = folium.Map(location=[40.7128, -74.0060], zoom_start=11, tiles="cartodbpositron")
        for _, r in sample_map.iterrows():
            folium.CircleMarker(
                location=[r.pickup_latitude, r.pickup_longitude],
                radius=1,
                color="blue",
                fill=True,
                fill_opacity=0.4
            ).add_to(m)
        map_path = MAP_DIR / "pickup_sample_map.html"
        m.save(str(map_path))
        print(f"Saved: {map_path}")
    except ImportError:
        print("Folium not installed, skipping map generation.")
    except Exception as e:
        print(f"Error generating map: {e}")

if __name__ == "__main__":
    generate_reports()
