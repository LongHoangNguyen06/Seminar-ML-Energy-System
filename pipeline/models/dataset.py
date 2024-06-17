import pandas as pd
import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, df: pd.DataFrame, hyperparameters):
        self.dataframe = df
        self.features = hyperparameters.model.features
        self.targets = hyperparameters.model.targets
        self.lag = hyperparameters.model.lag
        self.horizons = hyperparameters.model.horizons
        self.weather_future = (
            hyperparameters.model.weather_future
        )  # new parameter for future weather data

        # Determine the maximum horizon to ensure all targets can be accessed
        self.max_horizon = max(self.horizons) + self.weather_future
        # Calculate total samples considering the lag and the maximum forecast horizon
        self.total_samples = len(df) - (self.lag + self.max_horizon)

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # Adjust start index to accommodate the lag
        idx += self.lag

        # Collect inputs using the lag
        input_features = torch.tensor(
            self.dataframe.loc[
                [idx - lag for lag in range(self.lag)], self.features
            ].values.astype(float)
        )  # Shape: [n_batches, n_lag, n_features]

        # Collect lagged targets as additional features
        input_targets = torch.tensor(
            self.dataframe.loc[
                [idx - lag for lag in range(0, self.lag)], self.targets
            ].values.astype(float)
        )  # Shape: [n_batches, n_lag, num_targets]

        # Concatenate input features and lagged targets along the feature dimension
        inputs = torch.cat((input_features, input_targets), dim=-1)

        # Collect targets for each horizon and each feature
        targets = torch.tensor(
            self.dataframe.loc[
                [idx + horizon for horizon in self.horizons], self.targets
            ].values.astype(float)
        )  # Shape will be [n_batches, n_horizons, n_outputs]
        return inputs.to(torch.float32), targets.to(torch.float32)
