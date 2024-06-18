import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class TimeSeriesTransformer(nn.Module):
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
            dim_feedforward=d_model * hyperparameters.model.dim_feedforward_factor,
            dropout=hyperparameters.model.dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=hyperparameters.model.num_heads
        )

        # Output layer for each horizon and feature
        self.fc_out = nn.ModuleList(
            [
                nn.Linear(d_model, hyperparameters.model.num_targets)
                for _ in self.output_horizons
            ]
        )

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
        return nn.ReLU()(
            torch.stack([fc(pooled_transformed) for fc in self.fc_out], dim=1)
        )
