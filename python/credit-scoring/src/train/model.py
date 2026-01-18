from ast import Return
from typing import List, Optional

import torch
import torch.functional as F
import torch.nn as nn


class creditScoringModel(nn.Module):
    """
    Multi-layer perceptron for credit scoring
    """

    def __init__(
        self,
        num_features: int,
        hidden_layers: List[int],
        dropout_rate: float = 0.20,
        use_batch_norm: bool = True,
        activation_funct: str = "ReLU",
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.hidden_layers_config = hidden_layers
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.activation_fn_name = activation_funct

        layers = []
        input_size = num_features

        # network dinamyc
        for i, layer_size in enumerate(hidden_layers):
            # lineal layer
            layers.append(nn.Linear(input_size, layer_size))
            # batch
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(layer_size))
            # activation
            if activation_funct == "ReLU":
                layers.append(nn.ReLU())
            elif activation_funct == "LeakyReLU":
                layers.append(nn.LeakyReLU())
            elif activation_funct == "GELU":
                layers.append(nn.GELU())
            # dropout
            layers.append(nn.Dropout(dropout_rate))

            # outputsize to inputsize
            input_size = layer_size
        # output layer
        layers.append(nn.Linear(input_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(x)
            prob_good = torch.sigmoid(logits)
            prob_bad = 1 - prob_good
            return torch.cat([prob_good, prob_bad], dim=1)

    def predict(self, x: torch.Tensor, threshold: float = 0.5):
        with torch.no_grad():
            logits = self.forward(x)
            predict = torch.sigmoid(logits)
            return (predict > threshold).float()

    def get_model_info(self) -> dict:
        return {
            "model_type": "CreditScoringModel",
            "num_features": self.num_features,
            "dropuot_rate": self.dropout_rate,
            "use_batch_norm": self.use_batch_norm,
            "activation_fnc": self.activation_fn_name,
            "architecture": {
                "input_layer": self.num_features,
                "hidden_layers": self.hidden_layers_config,
                "output_layer": 1,
            },
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
        }
