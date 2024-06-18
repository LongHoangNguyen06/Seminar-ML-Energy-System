import os
import sys
import traceback

import pandas as pd

import wandb
from pipeline.config import CONF, get_config
from pipeline.models import training

os.environ["WANDB_TIMEOUT"] = "300"
PROJECT_NAME = "Seminar ML for Renewable Energy System"
ENTITY_NAME = "Seminar ML for Renewable Energy System"
SWEEP_NAME = "Multitask supply only 1h, 24h. ReLU output and more layers."


def exception_handling_train(df):
    with wandb.init(
        project=PROJECT_NAME,
        entity=ENTITY_NAME,
    ) as run:
        train_id = run.name  # Using the WandB's name train_id
        config = run.config  # Retrieve the configuration for this run

        hyperparameters = get_config()
        hyperparameters.model.num_layers = config["num_layers"]
        hyperparameters.model.num_heads = config["num_heads"]
        hyperparameters.model.dropout = config["dropout"]
        hyperparameters.model.lag = config["lag"]
        hyperparameters.model.weather_future = config["weather_future"]
        hyperparameters.model.dim_feedforward_factor = config["dim_feedforward_factor"]

        hyperparameters.train.batch_size = config["batch_size"]
        hyperparameters.train.lr = config["lr"]
        hyperparameters.train.min_lr = config["min_lr"]
        try:
            training.train_loop(
                hyperparameters,
                df,
                train_id=train_id,
                patience=hyperparameters.train.patience,
            )
        except Exception as e:
            print(e)
            print("An error occurred during training.")
            # traceback
            traceback.print_exc()


def hyper_parameter_optimize(sweep_id=None):
    df = pd.read_csv(os.path.join(CONF.data.preprocessed_data_dir, "df.csv"))
    if sweep_id is None:
        sweep_id = wandb.sweep(
            {
                "project": PROJECT_NAME,
                "name": SWEEP_NAME,
                "method": "bayes",  # Adjust search method as needed (grid, random)
                "metric": {
                    "goal": "minimize",  # Specify optimization goal (minimize/maximize)
                    "name": "best_val_loss",  # Replace with the metric you want to optimize
                },
                "parameters": {
                    "num_layers": {"values": list(range(2, 11, 2))},
                    "num_heads": {"values": list(range(2, 9, 2))},
                    "dropout": {"min": 0.0, "max": 0.4},
                    "lag": {"min": 12, "max": 48},
                    "weather_future": {"min": 12, "max": 24},
                    "dim_feedforward_factor": {"values": [1.0, 2.0, 4.0]},
                    "batch_size": {"values": [512, 256, 128]},
                    "lr": {"min": 1e-4, "max": 1e-1},
                    "min_lr": {"min": 1e-8, "max": 1e-5},
                },
            }
        )

    wandb.agent(
        sweep_id,
        lambda: exception_handling_train(df),
        count=CONF.train.hyperparameters_iters,
        project=PROJECT_NAME,
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sweep_id = sys.argv[1]
        print(f"Starting new agent for sweep {sweep_id}")
        hyper_parameter_optimize(sweep_id)
    else:
        hyper_parameter_optimize()
