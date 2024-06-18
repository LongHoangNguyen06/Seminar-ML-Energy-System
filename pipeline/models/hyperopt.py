import os
import random
import sys

import numpy as np
import pandas as pd
import torch

import wandb
from pipeline.config import CONF, get_config
from pipeline.models import training

os.environ["WANDB_TIMEOUT"] = "300"


def hyper_parameter_optimize(sweep_id=None):
    df = pd.read_csv(os.path.join(CONF.data.preprocessed_data_dir, "df.csv"))
    if sweep_id is None:
        sweep_id = wandb.sweep(
            {
                "project": "Seminar ML for Renewable Energy System",
                "name": "Multitask supply only 1h, 24h",
                "method": "bayes",  # Adjust search method as needed (grid, random)
                "metric": {
                    "goal": "minimize",  # Specify optimization goal (minimize/maximize)
                    "name": "best_val_loss",  # Replace with the metric you want to optimize
                },
                "parameters": {
                    "num_layers": {"min": 1, "max": 12},
                    "num_heads": {"min": 2, "max": 23},
                    "dropout": {"min": 0.0, "max": 0.4},
                    "lag": {"min": 12, "max": 168},
                    "weather_future": {"min": 12, "max": 24},
                    "dim_feedforward_factor": {"values": [0.5, 1.0, 2.0, 4.0]},
                    "batch_size": {"values": [1024, 512, 256, 128]},
                    "lr": {"min": 1e-4, "max": 1e-1},
                    "min_lr": {"min": 1e-8, "max": 1e-5},
                },
            }
        )

    def exception_handling_train():
        with wandb.init(
            project="Seminar ML for Renewable Energy System",
            entity="Seminar ML for Renewable Energy System",
        ) as run:
            train_id = run.id  # Using the WandB run ID as train_id
            config = run.config  # Retrieve the configuration for this run

            hyperparameters = get_config()
            hyperparameters.model.num_layers = config["num_layers"]
            hyperparameters.model.num_heads = config["num_heads"]
            hyperparameters.model.dropout = config["dropout"]
            hyperparameters.model.lag = config["lag"]
            hyperparameters.model.weather_future = config["weather_future"]
            hyperparameters.model.dim_feedforward_factor = config[
                "dim_feedforward_factor"
            ]

            hyperparameters.train.batch_size = config["batch_size"]
            hyperparameters.train.lr = config["lr"]
            hyperparameters.train.min_lr = config["min_lr"]

            randomseed = 42
            random.seed(randomseed)
            np.random.seed(randomseed)
            torch.manual_seed(randomseed)
            torch.cuda.manual_seed(randomseed)
            torch.cuda.manual_seed_all(randomseed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            training.train_loop(hyperparameters, df, train_id=train_id)

    wandb.agent(
        sweep_id, exception_handling_train, count=CONF.train.hyperparameters_iters
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sweep_id = sys.argv[1]
        hyper_parameter_optimize(sweep_id)
    else:
        hyper_parameter_optimize()
