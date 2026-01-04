import logging
import click
from training.model_training import run_training


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


@click.command()
@click.option("--tune", is_flag=True, help="Whether to run hyperparameter tuning.")
def main(tune):
    run_training(tune=tune)


if __name__ == "__main__":
    main()

# You can now run tuning with:
# export PYTHONPATH=$PYTHONPATH:. && python3 training/train.py --tune
