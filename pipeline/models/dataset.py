from functools import cache

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, df: pd.DataFrame, hyperparameters, features=None):
        self.dataframe = df
        self.features = hyperparameters.model.features if features is None else features
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
        self.lags_array = np.array([lag for lag in range(self.lag)])
        self.horizon_array = np.array([horizon for horizon in self.horizons])
        self.data = df.to_numpy()
        self.feature_indices = np.array([df.columns.get_loc(f) for f in self.features])
        self.target_indices = np.array([df.columns.get_loc(t) for t in self.targets])

    def __len__(self):
        return self.total_samples

    @cache
    def __getitem__(self, idx):
        assert isinstance(idx, int), "Index must be an integer"
        # Adjust start index to accommodate the lag
        idx += self.lag

        # Collect inputs using the lag
        input_features = torch.tensor(
            self.data[np.ix_(idx - self.lags_array, self.feature_indices)].astype(float)
        )  # Shape: [n_lag, n_features]

        # Collect lagged targets as additional features
        input_targets = torch.tensor(
            self.data[np.ix_(idx - self.lags_array, self.target_indices)].astype(float)
        )  # Shape: [n_lag, num_targets]

        # Concatenate input features and lagged targets along the feature dimension
        inputs = torch.cat((input_features, input_targets), dim=-1)

        # Collect targets for each horizon and each feature
        targets = torch.tensor(
            self.dataframe.loc[idx + self.horizon_array, self.targets].values.astype(
                float
            )
        )  # Shape will be [n_horizons, n_outputs]
        return inputs.to(torch.float32), targets.to(torch.float32)

    def get_feature_names(self):
        # Adjust start index to accommodate the lag
        rets = []
        # Collect inputs using the lag
        for lag in range(self.lag):
            for feature in self.features:
                rets.append(f"{feature}_lag_{lag}")

        # Collect lagged targets as additional features
        for lag in range(self.lag):
            for target in self.targets:
                rets.append(f"{target}_lag_{lag}")

        return rets
