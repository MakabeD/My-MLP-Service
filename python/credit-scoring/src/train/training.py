import os
import sys

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.processing.main import DataProcessing
from src.train.model import creditScoringModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yaml_default = "./config/training/train_credit-scoring_001a.yaml"


class EarlyStopping:
    def __init__(self, patience=10, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return

        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

            if self.counter >= self.patience:
                self.early_stop = True


class train:
    def __init__(self, yaml_path) -> None:
        
        # yaml loads
        (
            dataset_path,
            preprocess_filename,
            model_name,
            architecture,
            _test_size,
            _random_state,
            batch_size_config,
            _epochs,
            optimizer_config,
            loss_config,
            scheduler_config,
            early_stopping_config,
            metrics_to_compute_,
            mlflow_cfg,
        ) = self.getConfig(yaml_=yaml_path)
        # global seed
        self.set_global_seed(_random_state)

        # data
        self.dataProcss = DataProcessing(
            path=dataset_path,
            cache_file=preprocess_filename,
            test_size=_test_size,
            val_size=_test_size,
            random_state=_random_state,
        )

        self.x_train, self.y_train = self.dataProcss.get_train()
        self.x_val, self.y_val = self.dataProcss.get_val()
        self.x_test, self.y_test = self.dataProcss.get_test()
        # architecture
        self.input_size = self.x_train.shape[1]
        self.hidden_layers = architecture["hidden_layers"]
        self.use_batch_norm = architecture["use_batch_norm"]
        self.activation_fn = architecture["activation_fn"]
        self.dropout_rate = architecture["dropout_rate"]

        self.model = creditScoringModel(
            self.input_size,
            self.hidden_layers,
            self.dropout_rate,
            self.use_batch_norm,
            self.activation_fn,
        ).to(device)

        # loaders
        self.train_loader, self.val_loader, self.test_loader = self.posDataProcss(
            batch_size_config`
        )
        # inits
        mlflow.set_experiment(mlflow_cfg["mlflow_proyect_name"])
        self.criterion = self.set_loss_config(
            loss_config=loss_config, dataset_path=dataset_path
        )
        self.optimizer = self.set_optimizer_config(optimizer_config)
        self.scheduler = self.set_scheduler(scheduler_config, self.optimizer)
        self.early_stopping = self.set_early_stopping(early_stopping_config)

        # train loop
        with mlflow.start_run(run_name=mlflow_cfg["mlflow_run_name"]):
            self.train_loop(epochs=_epochs)
            self.mlflow_params_registry(
                optimizer_config, loss_config, scheduler_config, early_stopping_config
            )
            self.test_loop(
                metrics_to_compute=metrics_to_compute_, mlflow_cfg=mlflow_cfg
            )
            print("METRICS:", self.metrics)
            print("TYPE:", type(self.metrics))

            self.mlflow_metrics_registry(self.metrics, mlflow_cfg)
            self.save_model(model_name)
    def set_global_seed(self, seed:int): 
        import numpy as np
        import torch
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.deterministic = True
        
    def mlflow_params_registry(
        self, optimizer, use_pos_weight, scheduler, early_stopping
    ):
        # optimizer loop
        for name, value in optimizer.items():
            mlflow.log_param("optimizer_" + name, value)
        # pos weight
        mlflow.log_param("pos_weight", use_pos_weight["use_pos_weight"])
        # scheduler loop
        for name, value in scheduler.items():
            mlflow.log_param("scheduler_" + name, value)
        # early_stopping
        for name, value in early_stopping.items():
            mlflow.log_param("early_stopping_" + name, value)
        print("mlflow params")

    def set_early_stopping(self, config):
        patience = config["patience"]
        delta = config["delta"]
        early_stoping = EarlyStopping(patience, delta)
        return early_stoping

    def set_scheduler(self, config, optimizer):
        if config["name"] == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                patience=config["patience"],
                factor=config["factor"],
                mode="min",
            )
            return scheduler
        return None

    def set_loss_config(self, loss_config, dataset_path):
        if loss_config["use_pos_weight"]:
            y = pd.read_csv(dataset_path)
            y = y["Risk"].map({"good": 1, "bad": 0})
            trues = y.sum()
            falses = len(y) - trues
            pos_weight = falses / trues
            pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32).to(
                device
            )
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

            return loss_fn
        loss_fn = torch.nn.BCEWithLogitsLoss()
        return loss_fn

    def set_optimizer_config(self, optimizer_config):
        opt_name = optimizer_config["name"]
        if opt_name == "Adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=optimizer_config["learning_rate"],
                weight_decay=optimizer_config["weight_decay"],
            )
        elif opt_name == "SGD":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=optimizer_config["learning_rate"],
                weight_decay=optimizer_config["weight_decay"],
                momentum=0.9,
            )
        else:
            raise ValueError(f"Optimizer no soportado: {opt_name}")
        return optimizer

    def save_model(self, name):
        torch.save(self.model.state_dict(), "./models/" + name)
        print("Modelo guardado")

    def getConfig(self, yaml_: str = yaml_default):
        with open(yaml_, "r") as f:
            config = yaml.safe_load(f)
            dataset_path = config["data_source"]["data_path"]["dataset_path"]
            preprocess_filename = config["data_source"]["data_path"][
                "preprocessor_filename"
            ]
            model_name = config["model_config"]["model_name"]
            architecture = config["model_config"]["architecture"]
            test_size = config["training_params"]["test_size"]
            random_state = config["training_params"]["random_state"]
            batch_size = config["training_params"]["batch_size"]
            epochs = config["training_params"]["epochs"]
            optimizer = config["training_params"]["optimizer"]
            loss_config = config["training_params"]["loss_function"]
            scheduler_config = config["training_params"]["scheduler"]
            early_stopping_config = config["training_params"]["early_stopping"]
            metrics_to_compute = config["evaluation_params"]["metrics"]
            mlflow_cfg = config["mlflow_config"]
        return (
            dataset_path,
            preprocess_filename,
            model_name,
            architecture,
            test_size,
            random_state,
            batch_size,
            epochs,
            optimizer,
            loss_config,
            scheduler_config,
            early_stopping_config,
            metrics_to_compute,
            mlflow_cfg,
        )

    def posDataProcss(
        self, batch_size_config
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        x_train_t = self.x_train
        y_train_t = self.y_train
        x_val_t = self.x_val
        y_val_t = self.y_val
        x_test = self.x_test
        y_test = self.y_test
        # -> tensor
        x_train_t = x_train_t.astype(float)
        x_train_t = torch.tensor(x_train_t.values, dtype=torch.float32)
        y_train_t = torch.tensor(y_train_t, dtype=torch.float32).unsqueeze(1)

        x_val_t = x_val_t.astype(float)
        x_val_t = torch.tensor(x_val_t.values, dtype=torch.float32)
        y_val_t = torch.tensor(y_val_t.values, dtype=torch.float32).unsqueeze(1)

        x_test = x_test.astype(float)
        x_test = torch.tensor(x_test.values, dtype=torch.float32)
        y_test = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
        # tensor-> dataset
        train_ds = TensorDataset(x_train_t, y_train_t)
        val_ds = TensorDataset(x_val_t, y_val_t)
        tests_ds = TensorDataset(x_test, y_test)

        # Data loader
        train_loader = DataLoader(train_ds, batch_size=batch_size_config, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size_config, shuffle=False)
        test_loader = DataLoader(tests_ds, batch_size=batch_size_config, shuffle=False)
        return train_loader, val_loader, test_loader

    def train_loop(self, epochs: int = 20):
        # train
        for epoch in range(epochs):
            self.model.train()
            for x_batch, y_batch in self.train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                self.optimizer.zero_grad()
                output = self.model(x_batch)
                loss = self.criterion(output, y_batch)
                loss.backward()
                self.optimizer.step()
            # eval
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in self.val_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)

                    outputs = self.model(X_batch)
                    val_loss += self.criterion(outputs, y_batch).item()
            print(f"Epoch {epoch} | Val Loss: {val_loss:.4f}")
            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                print(" Early stopping activado")
                break

    def compute_metrics(self, TP, TN, FP, FN, metrics_to_compute):
        def accuracy_score(TP, TN, FP, FN):
            return (TP + TN) / (TP + TN + FP + FN + 1e-8)

        def precision_score(TP, TN, FP, FN):
            return TP / (TP + FP + 1e-8)

        def recall_score(TP, TN, FP, FN):
            return TP / (TP + FN + 1e-8)

        def FOneScore_(p, r):
            return 2 * (p * r) / (p + r + 1e-8)

        def specificity_score(TP, TN, FP, FN):
            return TN / (TN + FP + 1e-8)

        metrics_fncts = {
            "accuracy": accuracy_score,
            "precision": precision_score,
            "recall": recall_score,
            "f1_score": FOneScore_,
            "specificity": specificity_score,
        }
        results = {}
        for name in metrics_to_compute:
            if name == "f1_score":
                results[name] = FOneScore_(
                    precision_score(TP, TN, FP, FN), recall_score(TP, TN, FP, FN)
                )
                continue
            results[name] = metrics_fncts[name](TP, TN, FP, FN)
        return results

    def show_metrics(self, metrics):
        print("--------Evaluation Metrics--------")
        for name, value in metrics.items():
            print(f"{name:12}: {value:.4f}")

    def mlflow_metrics_registry(self, metrics, mlflow_cfg):
        for tag in mlflow_cfg["mlflow_tags"]:
            mlflow.set_tag(tag, True)
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
        print("mlflow metrics")

    def test_loop(self, metrics_to_compute, mlflow_cfg):
        TP, TN, FP, FN = 0, 0, 0, 0

        self.model.eval()
        xx, yy = 0, 0
        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.to(device)
                y = y.to(device)
                logit = self.model(x)
                logit = torch.sigmoid(logit)
                pred = (logit >= 0.5).float()

                # getting metrics
                TP += ((pred == 1) & (y == 1)).sum().item()
                TN += ((pred == 0) & (y == 0)).sum().item()
                FP += ((pred == 1) & (y == 0)).sum().item()
                FN += ((pred == 0) & (y == 1)).sum().item()
                for i in range(len(y)):
                    print(
                        f"Prob: {logit[i].item():.3f} | "
                        f"Pred: {int(pred[i].item())} | "
                        f"Real: {int(y[i].item())}"
                    )
                    yy += 1
                    if pred[i].item() == y[i].item():
                        xx += 1
            print("Total: ", yy, "||", "Buenos: ", xx)
            print(xx / yy)
            self.metrics = self.compute_metrics(TP, TN, FP, FN, metrics_to_compute)
            self.show_metrics(self.metrics)

from src.processing.yaml_process import parse_args as pa
if __name__ == "__main__":
    
    x = train(pa().config)
    #print(x.model.get_model_info())


"""

Execute training file:

python src/train/training.py --config ./config/training/train_credit-scoring_001a.yaml
python src/train/training.py --config ./config/training/train_credit-scoring_001b.yaml
python src/train/training.py --config ./config/training/train_credit-scoring_001c.yaml
python src/train/training.py --config ./config/training/train_credit-scoring_001d.yaml


"executing workflow test;ignore"
"""
