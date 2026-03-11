import os
import sys
from typing import Dict, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import mlflow
import mlflow.pytorch
import torch
from utils.config import Config, parse_args

from src.data.dataset import DatasetManager
from src.data.preprocess import Preprocess
from src.model.mlp import ChurnMLP

# Device Configuration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")


# Early Stopping


class EarlyStopping:
    """Early stopping to prevent overfitting during training."""

    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        """
        Args:
            patience: Number of epochs with no improvement after which training stops
            min_delta: Minimum change in validation loss to qualify as improvement
        """
        self.patience = patience
        self.delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> None:
        """Check if early stopping criteria is met."""
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# Utility Functions


def compute_metrics(TP: int, TN: int, FP: int, FN: int) -> Dict[str, float]:
    """
    Compute classification metrics from confusion matrix.

    Args:
        TP: True Positives
        TN: True Negatives
        FP: False Positives
        FN: False Negatives

    Returns:
        Dictionary with accuracy, precision, recall, specificity, and f1_score
    """
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    specificity = TN / (TN + FP + 1e-8)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": f1_score,
    }


# Main Trainer Class


class Trainer:
    """Mobile churn prediction model trainer with MLflow experiment tracking."""

    def run_train(self):
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(
            run_name=f"v{self.config_index}_{self.mlflow_cfg['mlflow_config']['run_name']}"
        ):
            self._log_experiment_config()
            self.train_loop()
            self.test_loop()
            self.save_model()
            print("[COMPLETE] Training and evaluation finished!")

    def __init__(
        self, config_index: int, experiment_name: str = "mobile_churn_prediction"
    ):
        """
        Initialize trainer with configuration.

        Args:
            config_index: Configuration version index (0-5 for v1-v5)
        """
        print(f"\n{'=' * 70}")
        print(f"[TRAINER] Initializing with config index: {config_index}")
        print(f"{'=' * 70}\n")
        self.experiment_name = experiment_name
        self.config_index = config_index
        self.config = Config(dir_index=config_index)
        self._load_configurations()
        self._prepare_data()
        self._build_model()
        self._setup_training_components()
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

    # Configuration Loading

    def _load_configurations(self) -> None:
        """Load and parse all configuration parameters from YAML files."""
        training_params = self.config.training["training_params"]
        self.epochs = training_params["epochs"]
        self.batch_size = training_params["batch_size"]
        self.optimizer_config = training_params["optimizer"]
        self.scheduler_config = training_params["scheduler"]
        self.early_stop_config = training_params["early_stopping"]
        self.loss_config = training_params["loss_function"]
        self.model_config = self.config.model
        self.data_config = self.config.data
        self.mlflow_cfg = self.config.mlflow
        print("[CONFIG] All configurations loaded successfully")

    # Data Preparation

    def _prepare_data(self) -> None:
        """Load, preprocess, and prepare data for training."""
        print("[DATA] Preparing dataset...")
        self.preprocessor = Preprocess(
            self.data_config["data_source"], self.data_config["dataset_info"]
        )
        self.preprocessor.run_and_save()

        self.data_manager = DatasetManager(self.preprocessor)
        self.data_manager.run()

        self.train_loader, self.val_loader, self.test_loader = (
            self.data_manager.get_loaders(batch_size=self.batch_size)
        )
        self.num_features = self.data_manager.x_train.shape[1]

        # Analyze class distribution
        y = self.data_manager.y_train
        n_positive = int(y.sum().item())
        n_negative = int(y.shape[0] - n_positive)
        total = n_positive + n_negative
        self.class_balance = n_positive / total

        print(f"[DATA] Training set size: {total}")
        print(
            f"[DATA] Class distribution - Positive: {n_positive} ({100 * n_positive / total:.1f}%), "
            f"Negative: {n_negative} ({100 * n_negative / total:.1f}%)"
        )

    # Model Architecture

    def _build_model(self) -> None:
        """Instantiate and initialize model architecture."""
        print("[MODEL] Building ChurnMLP model...")
        self.model = ChurnMLP(
            num_features=self.num_features, model_config=self.model_config
        ).to(device)
        model_info = self.model.get_model_info()
        print(f"[MODEL] Architecture: {model_info['architecture']}")
        print(f"[MODEL] Total parameters: {model_info['num_parameters']:,}")

    # Training Components Setup

    def _setup_training_components(self) -> None:
        """Initialize optimizer, scheduler, loss function, and early stopping."""
        self.criterion = self._create_loss_function()
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.early_stopping = EarlyStopping(
            patience=self.early_stop_config["patience"],
            min_delta=self.early_stop_config["delta"],
        )
        print("[TRAIN] All training components initialized\n")

    def _create_loss_function(self) -> torch.nn.Module:
        """
        Create loss function with optional positive weight for imbalanced data.

        Returns:
            BCEWithLogitsLoss with or without positive weight
        """
        if self.loss_config.get("use_pos_weight", False):
            y = self.data_manager.y_train
            n_positive = y.sum().item()
            n_negative = y.shape[0] - n_positive
            pos_weight = n_negative / (n_positive + 1e-8)
            print(
                f"[LOSS] Using weighted BCEWithLogitsLoss (pos_weight={pos_weight:.4f})"
            )
            return torch.nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(pos_weight, dtype=torch.float32).to(device)
            )
        print("[LOSS] Using standard BCEWithLogitsLoss")
        return torch.nn.BCEWithLogitsLoss()

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """
        Create optimizer based on configuration.

        Returns:
            Optimizer instance (Adam or SGD)
        """
        cfg = self.optimizer_config
        if cfg["name"] == "Adam":
            print(
                f"[OPTIMIZER] Adam - LR: {cfg['learning_rate']}, Weight Decay: {cfg['weight_decay']}"
            )
            return torch.optim.Adam(
                self.model.parameters(),
                lr=cfg["learning_rate"],
                weight_decay=cfg["weight_decay"],
            )
        elif cfg["name"] == "SGD":
            print(
                f"[OPTIMIZER] SGD - LR: {cfg['learning_rate']}, Momentum: {cfg.get('momentum', 0.9)}"
            )
            return torch.optim.SGD(
                self.model.parameters(),
                lr=cfg["learning_rate"],
                weight_decay=cfg["weight_decay"],
                momentum=cfg.get("momentum", 0.9),
            )
        else:
            raise ValueError(f"Unsupported optimizer: {cfg['name']}")

    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """
        Create learning rate scheduler based on configuration.

        Returns:
            Scheduler instance or None if not configured
        """
        cfg = self.scheduler_config
        if cfg["name"] == "ReduceLROnPlateau":
            print(
                f"[SCHEDULER] ReduceLROnPlateau - Patience: {cfg['patience']}, Factor: {cfg['factor']}"
            )
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=cfg["factor"],
                patience=cfg["patience"],
            )
        return None

    # MLflow Experiment Tracking

    def _log_experiment_config(self) -> None:
        """Log all experiment configuration parameters to MLflow."""
        print("[MLFLOW] Logging experiment configuration...")

        # Model parameters

        arch = self.model_config["model_config"]["architecture"]
        mlflow.log_param("model.hidden_layers", str(arch["hidden_layers"]))
        mlflow.log_param("model.dropout_rate", arch["dropout_rate"])
        mlflow.log_param("model.use_batch_norm", arch["use_batch_norm"])
        mlflow.log_param("model.activation_fn", arch["activation_fn"])

        # Training parameters
        tp = self.config.training["training_params"]
        mlflow.log_param("training.epochs", tp["epochs"])
        mlflow.log_param("training.batch_size", tp["batch_size"])

        # Optimizer parameters
        mlflow.log_param("optimizer.name", self.optimizer_config["name"])
        mlflow.log_param(
            "optimizer.learning_rate", self.optimizer_config["learning_rate"]
        )
        mlflow.log_param(
            "optimizer.weight_decay", self.optimizer_config["weight_decay"]
        )

        # Scheduler parameters
        mlflow.log_param("scheduler.name", self.scheduler_config["name"])
        mlflow.log_param("scheduler.patience", self.scheduler_config["patience"])
        mlflow.log_param("scheduler.factor", self.scheduler_config["factor"])

        # Early stopping parameters
        mlflow.log_param("early_stopping.patience", self.early_stop_config["patience"])
        mlflow.log_param("early_stopping.delta", self.early_stop_config["delta"])

        # Loss function parameters
        mlflow.log_param(
            "loss.use_pos_weight", self.loss_config.get("use_pos_weight", False)
        )

        # Data parameters
        mlflow.log_param("data.class_balance_ratio", self.class_balance)

    # Training Loop

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Execute one training epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        train_loss = 0.0
        TP = TN = FP = FN = 0

        for x_batch, y_batch in self.train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(x_batch)
            loss = self.criterion(logits, y_batch)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

            # Compute predictions and confusion matrix
            with torch.no_grad():
                pred = (torch.sigmoid(logits) >= 0.5).float()
                TP += ((pred == 1) & (y_batch == 1)).sum().item()
                TN += ((pred == 0) & (y_batch == 0)).sum().item()
                FP += ((pred == 1) & (y_batch == 0)).sum().item()
                FN += ((pred == 0) & (y_batch == 1)).sum().item()

        metrics = compute_metrics(TP, TN, FP, FN)
        metrics["loss"] = train_loss
        return metrics

    def _validate_epoch(self) -> Dict[str, float]:
        """
        Execute validation on entire validation set.

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        val_loss = 0.0
        TP = TN = FP = FN = 0

        with torch.no_grad():
            for x_val, y_val in self.val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)

                logits = self.model(x_val)
                val_loss += self.criterion(logits, y_val).item()

                pred = (torch.sigmoid(logits) >= 0.5).float()
                TP += ((pred == 1) & (y_val == 1)).sum().item()
                TN += ((pred == 0) & (y_val == 0)).sum().item()
                FP += ((pred == 1) & (y_val == 0)).sum().item()
                FN += ((pred == 0) & (y_val == 1)).sum().item()

        metrics = compute_metrics(TP, TN, FP, FN)
        metrics["loss"] = val_loss
        return metrics

    def train_loop(self) -> None:
        """Main training loop with validation and early stopping."""
        print(f"[TRAIN] Starting training for {self.epochs} epochs...\n")

        # Training loop
        for epoch in range(1, self.epochs + 1):
            train_metrics = self._train_epoch(epoch)
            val_metrics = self._validate_epoch()

            # Store history
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["train_acc"].append(train_metrics["accuracy"])
            self.history["val_acc"].append(val_metrics["accuracy"])

            # Log to MLflow
            mlflow.log_metric("train_loss", train_metrics["loss"], step=epoch)
            mlflow.log_metric("train_accuracy", train_metrics["accuracy"], step=epoch)
            mlflow.log_metric("train_precision", train_metrics["precision"], step=epoch)
            mlflow.log_metric("train_recall", train_metrics["recall"], step=epoch)
            mlflow.log_metric("train_f1", train_metrics["f1_score"], step=epoch)

            mlflow.log_metric("val_loss", val_metrics["loss"], step=epoch)
            mlflow.log_metric("val_accuracy", val_metrics["accuracy"], step=epoch)
            mlflow.log_metric("val_precision", val_metrics["precision"], step=epoch)
            mlflow.log_metric("val_recall", val_metrics["recall"], step=epoch)
            mlflow.log_metric("val_f1", val_metrics["f1_score"], step=epoch)

            # Print progress
            if epoch % 5 == 0 or epoch == 1:
                print(
                    f"[EPOCH {epoch:3d}/{self.epochs}] "
                    f"Train Loss: {train_metrics['loss']:.4f} | "
                    f"Train Acc: {train_metrics['accuracy']:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f} | "
                    f"Val Acc: {val_metrics['accuracy']:.4f}"
                )

            # Scheduler step and early stopping
            if self.scheduler:
                self.scheduler.step(val_metrics["loss"])

            self.early_stopping(val_metrics["loss"])
            if self.early_stopping.early_stop:
                print(f"\n[EARLY STOP] Triggered at epoch {epoch}")
                break

            print(f"\n[TRAIN] Training completed after {epoch} epochs\n")

    # Evaluation

    def evaluate(
        self, dataset_loader: torch.utils.data.DataLoader, dataset_name: str = "Test"
    ) -> Dict[str, float]:
        """
        Evaluate model on provided dataset.

        Args:
            dataset_loader: DataLoader for the dataset
            dataset_name: Name of the dataset (for logging)

        Returns:
            Dictionary with evaluation metrics and confusion matrix values
        """
        self.model.eval()
        TP = TN = FP = FN = 0

        with torch.no_grad():
            for x, y in dataset_loader:
                x = x.to(device)
                y = y.to(device)

                logits = self.model(x)
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()

                TP += ((preds == 1) & (y == 1)).sum().item()
                TN += ((preds == 0) & (y == 0)).sum().item()
                FP += ((preds == 1) & (y == 0)).sum().item()
                FN += ((preds == 0) & (y == 1)).sum().item()

        metrics = compute_metrics(TP, TN, FP, FN)
        metrics["TP"] = TP
        metrics["TN"] = TN
        metrics["FP"] = FP
        metrics["FN"] = FN

        # Log to MLflow
        mlflow.log_metric(f"{dataset_name}_accuracy", metrics["accuracy"])
        mlflow.log_metric(f"{dataset_name}_precision", metrics["precision"])
        mlflow.log_metric(f"{dataset_name}_recall", metrics["recall"])
        mlflow.log_metric(f"{dataset_name}_specificity", metrics["specificity"])
        mlflow.log_metric(f"{dataset_name}_f1_score", metrics["f1_score"])

        return metrics

    def test_loop(self) -> None:
        """Evaluate model on test set and display results."""
        print(f"\n{'=' * 70}")
        print("[TEST] Evaluating on test set...")
        print(f"{'=' * 70}\n")

        metrics = self.evaluate(self.test_loader, dataset_name="test")

        # Display results
        print(f"{'=' * 50}")
        print("TEST SET EVALUATION RESULTS")
        print(f"{'=' * 50}")
        print(f"Accuracy:    {metrics['accuracy']:.4f}")
        print(
            f"Precision:   {metrics['precision']:.4f}  (Correctness of positive predictions)"
        )
        print(
            f"Recall:      {metrics['recall']:.4f}  (Coverage of actual positive cases)"
        )
        print(
            f"Specificity: {metrics['specificity']:.4f}  (Correct identification of negatives)"
        )
        print(
            f"F1-Score:    {metrics['f1_score']:.4f}  (Harmonic mean of precision and recall)"
        )
        print("-" * 50)
        print("CONFUSION MATRIX:")
        print(f"                Actual CHURN    Actual LOYAL")
        print(
            f"Predicted CHURN:  {int(metrics['TP']):5d} (TP)   {int(metrics['FP']):5d} (FP)"
        )
        print(
            f"Predicted LOYAL:  {int(metrics['FN']):5d} (FN)   {int(metrics['TN']):5d} (TN)"
        )
        print(f"{'=' * 50}\n")

    # Model Persistence

    def save_model(self, path: Optional[str] = None) -> None:
        """
        Save model weights to disk and log to MLflow.

        Args:
            path: Path to save model. If None, uses model_name from config
        """
        if path is None:
            base_path = "artifacts/models"
            os.makedirs(base_path, exist_ok=True)
            model_name = self.model_config["model_config"].get(
                "model_version", "default_name"
            ) + self.model_config["model_config"].get(
                "model_name", "mobile_churn_model.pt"
            )
            path = os.path.join(base_path, model_name)

        torch.save(self.model.state_dict(), path)
        print(f"[SAVE] Model saved to: {path}")

        # Log model artifact to MLflow
        mlflow.pytorch.log_model(self.model, "churn_model")
        print(f"[MLFLOW] Model logged to MLflow")


# Main Execution


def main():
    args = parse_args()
    trainer = Trainer(config_index=args.config)
    trainer.run_train()


if __name__ == "__main__":
    main()
