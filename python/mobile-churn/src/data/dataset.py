import random

import numpy as np
import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data.preprocess import Preprocess
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed=42):
    """Set all seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For GPU
    # This ensures that even the convolution algorithms in CuDNN are deterministic
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    print(f"Seeds set to {seed}")


class DatasetManager:
    def __init__(self, preprocess, seed=42):
        self.preprocess = preprocess
        self.seed = seed
        # Placeholders for tensors and datasets
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.set_all_seeds()

    def run(self):
        self.to_tensor()
        self.to_dataset()

    def set_all_seeds(self):
        set_seed(self.seed)

    def get_loaders(self, batch_size=32):
        """Creates DataLoaders for all splits."""
        # generator ensures the DataLoader shuffle is also deterministic
        g = torch.Generator()
        g.manual_seed(self.seed)
        if self.train_ds is None or self.val_ds is None or self.test_ds is None:
            raise ValueError("Datasets are empty. Run DatasetManager.run() first!")
        train_loader = DataLoader(
            self.train_ds,
            batch_size=batch_size,
            shuffle=True,
            generator=g,
        )

        val_loader = DataLoader(self.val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(self.test_ds, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

    def _prepare_tensor(self, x, y):
        """Helper to convert DF/Series to float32 Tensors."""
        # Convert X to numpy then to tensor
        x_tensor = torch.tensor(x.to_numpy(), dtype=torch.float32)
        # Convert Y to numpy, then tensor, then reshape to (N, 1)
        y_tensor = torch.tensor(y.to_numpy(), dtype=torch.float32).unsqueeze(1)
        return x_tensor, y_tensor

    def to_tensor(self):
        """Convert all preprocessed splits into PyTorch Tensors."""
        # Validation: Ensure preprocess has the data
        if self.preprocess.x_train is None:
            raise ValueError(
                "Preprocess splits are empty. Run preprocessing.run() first!"
            )

        # Process each split using the helper
        self.x_train, self.y_train = self._prepare_tensor(
            self.preprocess.x_train, self.preprocess.y_train
        )
        self.x_val, self.y_val = self._prepare_tensor(
            self.preprocess.x_val, self.preprocess.y_val
        )
        self.x_test, self.y_test = self._prepare_tensor(
            self.preprocess.x_test, self.preprocess.y_test
        )

        print("All splits successfully converted to Tensors.")

    def to_dataset(self):
        """Create TensorDatasets for training, validation, and testing."""
        # It's better to call them _ds or similar to avoid confusion with the class name
        self.train_ds = TensorDataset(self.x_train, self.y_train)
        self.val_ds = TensorDataset(self.x_val, self.y_val)
        self.test_ds = TensorDataset(self.x_test, self.y_test)
        print("TensorDatasets created.")


if __name__ == "__main__":
    """Entry point: load config and execute preprocessing pipeline with serialization."""

    from utils.config import Config, parse_args

    config_index = parse_args().config
    config = Config(config_index)
    data = config.data
    data_source = data["data_source"]
    dataset_info = data["dataset_info"]
    x = Preprocess(data_source, dataset_info)
    x.run()
    y = DatasetManager(x)
    y.to_tensor()
    y.to_dataset()
