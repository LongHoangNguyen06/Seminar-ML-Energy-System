import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.models.transformer import PositionalEncoding


class SingleTimeSeriesTransformer(nn.Module):
    def __init__(self, hyperparameters):
        super().__init__()
        d_model = hyperparameters.model.num_features * hyperparameters.model.num_heads
        self.output_horizons = hyperparameters.model.horizons

        # Linear transformation to project input features to a higher dimensional space
        self.feature_to_embedding = nn.Linear(
            in_features=hyperparameters.model.num_features,
            out_features=d_model,
        )

        self.positional_encoder = PositionalEncoding(
            d_model=d_model, dropout=hyperparameters.model.dropout
        )

        # Transformer Encoder Layer
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=hyperparameters.model.num_heads,
            dim_feedforward=int(d_model * hyperparameters.model.dim_feedforward_factor),
            dropout=hyperparameters.model.dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=hyperparameters.model.num_heads
        )

        # Output layer for each horizon and feature
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, src):
        src = src.permute(
            1, 0, 2
        )  # Permute to (sequence_length, batch_size, num_features)

        src = self.feature_to_embedding(
            src
        )  # Map features to the higher dimensional space

        # Apply positional encoding
        src = self.positional_encoder(src)

        # Apply transformer encoder
        transformed = self.transformer_encoder(src)

        # Use all tokens for forecasting
        pooled_transformed = transformed.mean(dim=0)

        # Forecast
        output = self.fc_out(pooled_transformed)

        # Apply ReLU if not in training mode
        if not self.training:
            output = F.relu(output)

        return output


class HorizonPredictor(nn.Module):
    def __init__(self, hyperparameters):
        super().__init__()
        self.models = nn.ModuleList(
            [
                SingleTimeSeriesTransformer(hyperparameters)
                for _ in range(len(hyperparameters.model.horizons))
            ]
        )

    def forward(self, src):
        outputs = [model(src) for model in self.models]
        return torch.cat(outputs, dim=1)


class TargetPredictor(nn.Module):
    def __init__(self, hyperparameters):
        super().__init__()
        self.models = nn.ModuleList(
            [
                SingleTimeSeriesTransformer(hyperparameters)
                for _ in range(len(hyperparameters.model.targets))
            ]
        )

    def forward(self, src):
        outputs = [model(src) for model in self.models]
        return torch.stack(outputs, dim=2)


class HorizonTargetPredictor(nn.Module):
    def __init__(self, hyperparameters):
        super().__init__()
        self.models = nn.ModuleList(
            [
                SingleTimeSeriesTransformer(
                    hyperparameters, horizon_idx=h, target_idx=t
                )
                for h in range(len(hyperparameters.model.horizons))
                for t in range(len(hyperparameters.model.targets))
            ]
        )
        self.n_horzions = len(hyperparameters.model.horizons)
        self.n_targets = len(hyperparameters.model.targets)

    def forward(self, src):
        outputs = [model(src) for model in self.models]
        return torch.stack(outputs, dim=2).view(
            src.size(0), self.n_horzions, self.n_targets
        )
