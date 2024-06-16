import torch.nn as nn

class PytorchEncoderOnlyMultiHorizonTransformer(nn.Module):
    def __init__(self, num_features=1, num_layers=1, num_heads=8, forward_expansion=4, dropout=0.1, output_horizons=[1, 24]):
        """Initializes the Transformer model for multi-horizon time series forecasting.

        Args:
            num_features (int): The number of features in the input data. Default is 1 for univariate time series.
            num_layers (int): The number of layers in the Transformer encoder.
            num_heads (int): The number of heads in the multi-head attention mechanism.
            forward_expansion (int): The factor by which to expand dimensions in the feedforward network.
            dropout (float): The dropout rate to use for regularization in the Transformer.
            output_horizons (list): A list of integers representing the forecast horizons (in steps) to predict.
        """
        super(PytorchEncoderOnlyMultiHorizonTransformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=num_features, 
            nhead=num_heads, 
            dim_feedforward=num_features * forward_expansion, 
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(num_features, len(output_horizons))  # Output layer to predict multiple horizons

    def forward(self, src):
        """Defines the forward pass of the model.

        Args:
            src (Tensor): The input tensor of shape (batch_size, sequence_length, num_features).
                          `sequence_length` is the length of the input time series.

        Returns:
            Tensor: A tensor of shape (batch_size, len(output_horizons)).
                    Each element in the tensor corresponds to a prediction for one of the horizons specified in `output_horizons`.
        """
        # Permute the input to match the expected shape (sequence_length, batch_size, num_features) for the Transformer
        src = src.permute(1, 0, 2)
        
        # Encode the input sequence
        transformed = self.transformer_encoder(src)
        
        # Select the last time step's output from the transformed sequence for making the forecast
        # This step assumes that the model should use the representation of the final time step in the input sequence
        # to make predictions for all specified horizons.
        output = self.fc_out(transformed[-1])
        
        return output
