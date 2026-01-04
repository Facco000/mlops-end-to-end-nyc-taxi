# End-to-End MLOps Pipeline – NYC Taxi Trip Duration

## Project Overview

Questo progetto dimostra una **pipeline completa di Machine Learning**, dal caricamento dei dati grezzi fino a componenti pronti per la produzione, seguendo le best practice di MLOps.

L'obiettivo è predire la **durata delle corse dei taxi a New York** usando dati storici, con un focus su:

- Riproducibilità
- Qualità dei dati
- Separazione tra training e serving
- Design orientato alla produzione

---

## Tech Stack

- **Python 3.11**
- **pandas, numpy**
- **scikit-learn**
- **XGBoost**
- **MLflow** (experiment tracking & model registry)
- **Feast** (feature store)
- **FastAPI** (model serving)
- **Docker / docker-compose** (opzionale)

---

## Dataset

NYC Taxi Trip Records (open dataset pubblico):  
[NYC Taxi Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

- Sottoinsieme dei **Yellow Taxi 2015** per mantenere gli esperimenti leggeri
- File principali:

```
data/raw/
  yellow_tripdata_2015-01.csv
  yellow_tripdata_2015-01.parquet
data/processed/
  taxi_ml_2015_01.parquet
feature_repo/taxi_features/data/
  driver_stats.parquet
```

---

## ML Task

**Supervised regression**:

- Target: `trip_duration_seconds`  
- Metriche principali: **RMSE**, **MAE**
- Ottimizzazione: **Hyperparameter Tuning** (condizionale), **Early Stopping**

---

## Project Structure

```
mlops-end-to-end-taxi/
│
├── data/
│   ├── raw/                    # dati originali, immutabili
│   └── processed/              # dati puliti e pronti per ML
│
├── notebooks/
│   └── 01_eda.ipynb           # EDA esplorativa
│
├── src/
│   ├── etl/
│   │   ├── extract.py
│   │   ├── transform.py
│   │   ├── load.py
│   │   ├── run_etl.py
│   │   ├── convert_csv_to_parquet.py
│   │   └── __init__.py
│   │
│   ├── config.py
│   ├── utils.py
│   └── __init__.py
│
├── feature_repo/taxi_features/  # Feast feature repo
│   ├── feature_definitions.py
│   ├── taxi_trip_source.py
│   ├── driver_stats.parquet
│   ├── test_workflow.py
│   └── feature_store.yaml
│
├── training/
│   ├── train.py               # entry point
│   ├── model_training.py      # training logic (XGBoost + Feast)
│   └── __init__.py
│
├── mlruns/                     # MLflow experiments (auto-generato)
├── notebooks/                  # Notebook di EDA
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Pipeline Steps

1. **Exploratory Data Analysis (EDA)**  
   - Analisi preliminare, valori mancanti, distribuzione target, correlazioni

2. **ETL & Data Cleaning**
   - Estrazione CSV/Parquet
   - Trasformazioni: gestione outlier, tipologie, feature temporali

3. **Feature Engineering**
   - Calcolo di hour, day_of_week, is_weekend
   - Creazione di feature geografiche e derivate

4. **Feature Store Integration (Feast)**
   - Organizza e serve le feature in modo coerente tra training e serving

5. **Model Training & Optimization (XGBoost + MLflow)**
   - **Tuning**: Supporto per `RandomizedSearchCV` per ottimizzare gli iperparametri.
   - **Early Stopping**: Integrazione per prevenire l'overfitting.
   - **Tracking**: Log di parametri (base e tuned), metriche e modelli in MLflow.
   - Run esempio:
     ```
     RMSE: 150.22
     MAE: 51.50
     ```


## Pipeline Architecture

```mermaid
flowchart TB
    subgraph Data["📦 Data Layer"]
        A[NYC Taxi Raw Data<br/>CSV / Parquet]
        B[ETL Pipeline<br/>Extract · Transform · Load]
        C[Processed Dataset<br/>Parquet]
        A --> B --> C
    end

    subgraph Features["📊 Feature Layer"]
        D[Feature Engineering]
        E[Feature Store<br/>Feast]
        C --> D --> E
    end

    subgraph Training["🤖 Training Layer"]
        F[XGBoost Training]
        G[Hyperparameter Tuning<br/>RandomizedSearchCV]
        H[Experiment Tracking<br/>MLflow]
        E --> F
        F --> G
        G --> H
    end

    subgraph Serving["🚀 Serving Layer"]
        I[FastAPI Inference Service]
        J[Model Registry<br/>MLflow]
        K[Monitoring<br/>Metrics & Drift]
        H --> J --> I --> K
    end

    %% Shared features between training and serving
    E -.-> I
```

---

## How to Run

### 1. Setup Environment

```bash
make install
```

### 2. Run ETL Pipeline

```bash
make run_etl
```

This command extracts raw data, transforms it using the logic in `src/etl`, and saves the processed parquet file to `data/processed/`.

### 3. Feature Store

```bash
cd feature_repo/taxi_features
feast apply
```

### 4. Model Training
```bash
make train
```

This runs the XGBoost training pipeline with default (optimized) settings.

### 5. Hyperparameter Tuning

```bash
make train-tune
```

Runs a `RandomizedSearchCV` phase on a subset of the data (per velocità) per trovare i parametri migliori, poi addestra il modello finale.

### 5. Running Tests

```bash
make test
```

### 6. Code Quality

```bash
make lint
make format
```

### 7. MLflow UI

```bash
mlflow ui
```

### 8. Generating Reports

```bash
make reports
```

This generates visualization plots and an interactive map in the `reports/` folder.

### 9. Running with Docker

If you prefer using Docker, you can start the environment with:

```bash
docker-compose up --build
```

This will:
- Spin up an **MLflow** server on `http://localhost:5001`.
- Provide an `app` container where you can run commands:
  ```bash
  docker-compose exec app make run_etl
  docker-compose exec app make train
  ```

---

## Results

Run finale XGBoost:

```
RMSE: 148.39
MAE: 50.92
```

Modello loggato in MLflow e riproducibile

## Visual reports

Ho incluso alcuni grafici chiave e una mappa interattiva nella cartella `reports/`.

### Figure
![Feature importance](reports/figures/feature_importance.png)
*Figura 1 — Feature importance (XGBoost).*

![Predicted vs Actual](reports/figures/predicted_vs_actual.png)
*Figura 2 — Predette vs Reali (sample).*

![Residuals distribution](reports/figures/residuals_distribution.png)
*Figura 3 — Distribuzione dei residui (Actual - Predicted).*

![Heatmap hour/day](reports/figures/heatmap_hour_day.png)
*Figura 4 — Durata media per ora × giorno della settimana.*

### Mappa interattiva
La mappa delle pickup (sample) è salvata come HTML:
- `reports/maps/pickup_sample_map.html` (apri nel browser)



---

## Notes / Next Steps

Pipeline completamente end-to-end e pronta per il portfolio

Miglioramenti futuri:
- Modelli avanzati (LightGBM, CatBoost, reti neurali)
- API FastAPI per serving
- Drift detection e monitoraggio metriche

---

## References

- [NYC Taxi & Limousine Commission Dataset](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- [MLflow Docs](https://mlflow.org/docs/latest/index.html)
- [Feast Docs](https://docs.feast.dev/)