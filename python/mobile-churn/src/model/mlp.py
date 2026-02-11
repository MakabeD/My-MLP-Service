import torch
from torch import nn
from torch.nn.modules.container import Sequential


# print(torch.__version__)
# print(torch.cuda.is_available())
class ChurnMLP(nn.Module):
    def __init__(self, num_features: int, model_config: dict) -> None:
        super().__init__()
        model_config = model_config["model_config"]
        self.num_features = num_features
        self.hidden_layers = model_config["architecture"]["hidden_layers"]
        self.dropout_rate = model_config["architecture"]["dropout_rate"]
        self.use_batch_norm = model_config["architecture"]["use_batch_norm"]
        self.activation_fn = model_config["architecture"]["activation_fn"]
        input_size = num_features
        layers = []
        for output_size in self.hidden_layers:
            layers.append(nn.Linear(input_size, output_size))
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(output_size))
            if self.activation_fn == "ReLU":
                layers.append(nn.ReLU())
            elif self.activation_fn == "GELU":
                layers.append(nn.GELU())
            elif self.activation_fn == "LeakyReLU":
                layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(self.dropout_rate))
            input_size = output_size
        layers.append(nn.Linear(input_size, 1))
        self.network = Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def predict(self, x: torch.Tensor, threshold: float = 0.5):
        with torch.no_grad():
            logits = self.forward(x)
            predict = torch.sigmoid(logits)
            return (predict > threshold).float()

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(x)
            prob_good = torch.sigmoid(logits)
            prob_bad = 1 - prob_good
            return torch.cat([prob_good, prob_bad], dim=1)

    def get_model_info(self) -> dict:
        return {
            "model_type": "Mobile Churn",
            "num_features": self.num_features,
            "dropout_rate": self.dropout_rate,
            "use_batch_norm": self.use_batch_norm,
            "activation_fn": self.activation_fn,
            "architecture": {
                "input_layer": self.num_features,
                "hidden_layers": self.hidden_layers,
                "output_layer": 1,
            },
            "num_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
        }
