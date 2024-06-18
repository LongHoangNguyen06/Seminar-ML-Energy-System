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
        super(TimeSeriesTransformer, self).__init__()
        num_layers = hyperparameters.model.num_layers
        num_heads = hyperparameters.model.num_heads
        forward_expansion = hyperparameters.model.forward_expansion
        dropout = hyperparameters.model.dropout
        output_horizons = hyperparameters.model.horizons
        d_model = hyperparameters.model.num_features * forward_expansion
        self.output_horizons = output_horizons

        # Linear transformation to project input features to a higher dimensional space
        self.feature_to_embedding = nn.Linear(
            in_features=hyperparameters.model.num_features,
            out_features=d_model,
            bias=False,
        )

        self.positional_encoder = PositionalEncoding(d_model=d_model, dropout=dropout)

        # Transformer Encoder Layer
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        )

        # Output layer for each horizon and feature
        self.fc_out = nn.ModuleList(
            [
                nn.Linear(d_model, hyperparameters.model.num_targets)
                for _ in output_horizons
            ]
        )

    def forward(self, src):
        src = src.permute(
            1, 0, 2
        )  # Permute to (sequence_length, batch_size, num_features)
        src = self.feature_to_embedding(
            src
        )  # Map features to the higher dimensional space
        src = self.positional_encoder(src)
        transformed = self.transformer_encoder(src)

        # Use all tokens for forecasting, potentially averaging their representations
        # or using another method to combine information across all tokens
        outputs = [
            fc(transformed.mean(dim=0)) for fc in self.fc_out
        ]  # Example: mean pooling

        outputs = [fc(outputs) for fc in self.fc_out]
        return torch.stack(outputs, dim=1)
