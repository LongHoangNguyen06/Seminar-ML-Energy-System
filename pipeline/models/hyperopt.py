import os
import sys
import traceback

import pandas as pd

import wandb
from pipeline import utils
from pipeline.config import CONF, get_config
from pipeline.models import training

os.environ["WANDB_TIMEOUT"] = "300"


def exception_handling_train(df):
    hyperparameters = get_config()
    run_name = utils.current_time_str()
    with wandb.init(
        project=hyperparameters.wandb.project_name,
        name=run_name,
    ) as run:
        train_id = run_name  # Using the WandB's name train_id
        config = run.config  # Retrieve the configuration for this run

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
                "project": CONF.wandb.project_name,
                "name": CONF.wandb.sweep_name,
                "method": "bayes",  # Adjust search method as needed (grid, random)
                "metric": {
                    "goal": "minimize",  # Specify optimization goal (minimize/maximize)
                    "name": "best_val_loss",  # Replace with the metric you want to optimize
                },
                "parameters": {
                    "num_layers": {"values": [1, 2]},
                    "num_heads": {"values": [1, 2]},
                    "dropout": {"min": 0.0, "max": 0.1},
                    "lag": {"min": 1, "max": 24},
                    "weather_future": {"min": 12, "max": 24},
                    "dim_feedforward_factor": {
                        "values": [0.5, 1.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
                    },
                    "batch_size": {"values": [512]},
                    "lr": {"min": 1e-4, "max": 1e-3},
                    "min_lr": {"min": 1e-8, "max": 1e-5},
                },
            }
        )

    wandb.agent(
        sweep_id,
        lambda: exception_handling_train(df),
        count=CONF.train.hyperparameters_iters,
        project=CONF.wandb.project_name,
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sweep_id = sys.argv[1]
        print(f"Starting new agent for sweep {sweep_id}")
        hyper_parameter_optimize(sweep_id)
    else:
        hyper_parameter_optimize()
