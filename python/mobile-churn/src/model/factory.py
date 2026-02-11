import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from mlp import ChurnMLP
from utils.config import Config


class ModelFactory:
    @staticmethod
    def build(num_features, config: Config):
        model = ChurnMLP(num_features, config.model)
        return model

    @staticmethod
    def load(num_features, config: Config, path: str):
        model = ChurnMLP(num_features, config.model)
        model.load_state_dict(torch.load(path))
        model.eval()
        return model


if __name__ == "__main__":
    from utils.config import Config, parse_args

    dir_index = parse_args()
    config = Config(dir_index.config)
    model = ModelFactory().build(1, config)
    print(model.get_model_info())
