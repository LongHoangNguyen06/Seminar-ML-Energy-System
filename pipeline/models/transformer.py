import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_model(hyperparameters):
    if hyperparameters.model.architecture == "MultiTaskTransformer":
        print("Building MultiTaskTransformer model")
        return MultiTaskTransformer(hyperparameters=hyperparameters)
    elif hyperparameters.model.architecture == "HorizonTransformer":
        print("Building HorizonTransformer model")
        return HorizonTransformer(hyperparameters=hyperparameters)
    elif hyperparameters.model.architecture == "TargetTransformer":
        print("Building TargetTransformer model")
        return TargetTransformer(hyperparameters=hyperparameters)
    elif hyperparameters.model.architecture == "HorizonTargetTransformer":
        print("Building HorizonTargetTransformer model")
        return HorizonTargetTransformer(hyperparameters=hyperparameters)
    else:
        raise ValueError("Invalid model architecture")


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term)[:, 0:-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(self, hyperparameters, n_outputs):
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
        self.fc_out = nn.Linear(d_model, n_outputs)

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


class MultiTaskTransformer(nn.Module):
    def __init__(self, hyperparameters):
        super().__init__()
        self.n_horzions = len(hyperparameters.model.horizons)
        self.n_targets = len(hyperparameters.model.targets)
        self.model = Transformer(
            hyperparameters, n_outputs=self.n_targets * self.n_horzions
        )

    def forward(self, src):
        outputs = self.model(src)
        return outputs.view(src.size(0), self.n_horzions, self.n_targets)


class HorizonTransformer(nn.Module):
    def __init__(self, hyperparameters):
        super().__init__()
        self.n_horzions = len(hyperparameters.model.horizons)
        self.n_targets = len(hyperparameters.model.targets)
        self.models = nn.ModuleList(
            [
                Transformer(hyperparameters, n_outputs=self.n_targets)
                for _ in range(len(hyperparameters.model.horizons))
            ]
        )

    def forward(self, src):
        outputs = [model(src) for model in self.models]
        return torch.cat(outputs, dim=1).view(
            src.size(0), self.n_horzions, self.n_targets
        )


class TargetTransformer(nn.Module):
    def __init__(self, hyperparameters):
        super().__init__()
        self.n_horzions = len(hyperparameters.model.horizons)
        self.n_targets = len(hyperparameters.model.targets)
        self.models = nn.ModuleList(
            [
                Transformer(hyperparameters, n_outputs=self.n_horzions)
                for _ in range(len(hyperparameters.model.targets))
            ]
        )

    def forward(self, src):
        outputs = [model(src) for model in self.models]
        return torch.stack(outputs, dim=2).view(
            src.size(0), self.n_horzions, self.n_targets
        )


class HorizonTargetTransformer(nn.Module):
    def __init__(self, hyperparameters):
        super().__init__()
        self.models = nn.ModuleList(
            [
                Transformer(hyperparameters, n_outputs=1)
                for _ in range(len(hyperparameters.model.horizons))
                for _ in range(len(hyperparameters.model.targets))
            ]
        )
        self.n_horzions = len(hyperparameters.model.horizons)
        self.n_targets = len(hyperparameters.model.targets)

    def forward(self, src):
        outputs = [model(src) for model in self.models]
        return torch.stack(outputs, dim=2).view(
            src.size(0), self.n_horzions, self.n_targets
        )
