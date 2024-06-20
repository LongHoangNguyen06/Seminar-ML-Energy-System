from functools import cache

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, hyperparameters, features=None):
        self.dataframe = dataframe
        self.features = hyperparameters.model.features if features is None else features
        self.targets = hyperparameters.model.targets
        self.weather_features = hyperparameters.model.weather_features
        self.lag = hyperparameters.model.lag
        self.horizons = hyperparameters.model.horizons

        # Determine the maximum horizon to ensure all targets can be accessed
        self.max_horizon = max(self.horizons)
        # Calculate total samples considering the lag and the maximum forecast horizon
        self.total_samples = len(dataframe) - (self.lag + self.max_horizon)
        self.lags_array = np.array([lag for lag in range(self.lag)])
        self.horizon_array = np.array([horizon for horizon in self.horizons])
        self.weather_horizon_array = np.array([i for i in range(0, 24)])
        self.data = dataframe.to_numpy()
        self.feature_indices = np.array(
            [dataframe.columns.get_loc(f) for f in self.features]
        )
        self.target_indices = np.array(
            [dataframe.columns.get_loc(t) for t in self.targets]
        )
        self.weather_indices = np.array(
            [dataframe.columns.get_loc(w) for w in self.weather_features]
        )

    def __len__(self):
        return self.total_samples

    @cache
    def __getitem__(self, idx):
        assert isinstance(idx, int), "Index must be an integer"
        # Adjust start index to accommodate the lag
        idx = self.get_dataframe_index(idx)

        # Collect inputs using the lag
        lagged_features = self.get_lagged_features(idx)

        # Collect lagged targets
        lagged_targets = self.get_lagged_targets(idx)

        # Adding hour of day, day of year
        time_features = self.get_time_features(idx)
        time_features = time_features.expand(lagged_features.size(0), -1)

        # Adding future weather data
        forecast = self.get_forecast_features(idx)

        # Concatenate everything together
        past = torch.cat([lagged_features, lagged_targets, time_features], dim=-1).to(
            torch.float32
        )
        y = self.get_targets(idx).to(torch.float32)
        # Return the input and output tensors
        return (past, forecast), y

    def get_feature_names(self):
        # Adjust start index to accommodate the lag
        rets = []

        # Lagged features
        for lag in range(self.lag):
            for feature in self.features:
                rets.append(f"{feature}_lag_{lag}")

        # Lagged targets
        for lag in range(self.lag):
            for target in self.targets:
                rets.append(f"{target}_lag_{lag}")

        # Times
        rets.append("hour_of_day")
        rets.append("day_of_year")

        # Forecast
        for h in self.weather_horizon_array:
            for feature in self.weather_features:
                rets.append(f"{feature}_future_{h}h")

        return rets

    def get_dataframe_index(self, idx):
        return idx + self.lag

    def get_lagged_features(self, idx):
        return torch.tensor(
            self.data[np.ix_(idx - self.lags_array, self.feature_indices)].astype(float)
        )  # Shape: [n_lag, n_features]

    def get_targets(self, idx):
        return torch.tensor(
            self.data[np.ix_(idx + self.horizon_array, self.target_indices)].astype(
                float
            )
        )  # Shape: [n_lag, n_targets]

    def get_lagged_targets(self, idx):
        return torch.tensor(
            self.data[np.ix_(idx - self.lags_array, self.target_indices)].astype(float)
        )  # Shape: [n_lag, n_targets]

    def get_time_features(self, idx):
        # Extract hour of day and day of year features
        datetime_index = self.dataframe["Date from"][idx]
        hour_of_day = datetime_index.hour / 24.0
        day_of_year = datetime_index.dayofyear / 365.0
        return torch.tensor([hour_of_day, day_of_year]).to(torch.float32).unsqueeze(0)

    def get_forecast_features(self, idx):
        # mask the data that should be ignored
        datetime_index = self.dataframe["Date from"][idx]
        hour_of_day = datetime_index.hour
        n_next = 24 - hour_of_day
        # Extract future weather forecast data
        future_weather_data = self.data[
            np.ix_(idx + self.weather_horizon_array[:n_next], self.weather_indices)
        ].astype(float)
        output = torch.tensor(list(future_weather_data)).to(torch.float32)
        return output


class TimeSeriesDataLoader:
    def __init__(self, dataset, batch_size, dataframe):
        self.dataset = dataset
        self.batch_size = batch_size

        # Group indices by their hour of the day
        self.hour_groups = {
            hour: [] for hour in dataframe["Date from"].dt.hour.unique()
        }
        for dataset_idx in range(len(self.dataset)):
            hour = (
                dataframe["Date from"]
                .iloc[dataset.get_dataframe_index(dataset_idx)]
                .hour
            )
            self.hour_groups[hour].append(dataset_idx)

        # Shuffle each group
        for hour in self.hour_groups:
            np.random.shuffle(self.hour_groups[hour])

        # Create index_batches
        self.index_batches = []
        for hour in self.hour_groups:
            for i in range(0, len(self.hour_groups[hour]), self.batch_size):
                self.index_batches.append(
                    self.hour_groups[hour][i : i + self.batch_size]  # noqa: E203
                )

        # Shuffle the index_batches
        np.random.shuffle(self.index_batches)

    def __iter__(self):
        for index_batch in self.index_batches:
            yield self._get_batch(index_batch)

    def __len__(self):
        return len(self.index_batches)

    def _get_batch(self, batch_indices):
        batch = [self.dataset[idx] for idx in batch_indices]
        past = torch.stack([item[0][0] for item in batch])
        forecast = torch.stack([item[0][1] for item in batch])
        targets = torch.stack([item[1] for item in batch])
        return (past, forecast), targets
