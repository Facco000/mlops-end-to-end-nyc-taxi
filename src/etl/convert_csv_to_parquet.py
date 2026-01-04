from pathlib import Path
import pandas as pd


def csv_to_parquet(csv_path: Path, parquet_path: Path) -> None:
    df = pd.read_csv(csv_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(parquet_path, engine="pyarrow", index=False)


if __name__ == "__main__":
    csv_file = Path(
        "/Users/facco/Documents/Coding/DATA_SCIENCE/mlops-end-to-end-taxi/data/raw/yellow_tripdata_2015-01.csv"
    )
    parquet_file = Path(
        "/Users/facco/Documents/Coding/DATA_SCIENCE/mlops-end-to-end-taxi/data/raw/yellow_tripdata_2015-01.parquet"
    )

    csv_to_parquet(csv_file, parquet_file)
