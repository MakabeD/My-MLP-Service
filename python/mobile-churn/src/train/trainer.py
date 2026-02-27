import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from utils.config import Config
from src.model.mlp import ChurnMLP 
from src.data.preprocess import Preprocess
from src.data.dataset import DatasetManager
import torch
device = "cuda" if torch.cuda.is_available() else "cpu" 

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.delta = min_delta
        self.counter = 0
        self.best_loss = None
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

class Trainer:
    def __init__(self):
        configs=self.configs = Config(dir_index=0) #TODO: dont forget do not hardcode the index, make it dynamic by using argparse
        # FIXME: this is a temporary solution, we should refactor the code to avoid hardcoding the index

        # FIXME: try hard to improve the acuracy of the model by tuning the hyperparameters, changing the architecture, etc. 
        # currently the model is not performing well, we need to improve it. Current acuracy is around 0.54, we need to get it to at least 0.7 
        self.epochs :int= configs.training["training_params"]["epochs"] 
        self.model_config = configs.model
        self.data_config = configs.data
        self.loss_config = configs.training['training_params']['loss_function']
        self.optimizer_config = configs.training['training_params']['optimizer']
        self.preprocessor = Preprocess(configs.data["data_source"], configs.data["dataset_info"])
        self.preprocessor.run_and_save()
        self.data_manager= DatasetManager(self.preprocessor)
        self.data_manager.run()
        #data loaders
        self.train_loader, self.val_loader, self.test_loader = self.data_manager.get_loaders()
        self.num_features = self.data_manager.x_train.shape[1]
        #model 
        self.model = ChurnMLP(num_features=self.num_features, model_config=self.model_config).to(device)
        self.criterion=self.set_criterion()
        self.optimizer = self.set_optimizer()
        self.scheduler= self.set_scheduler()
        self.early_stopping=self.set_early_stopping()

    def set_criterion(self):
        if self.loss_config['use_pos_weight']:
            trues=self.preprocessor.y.sum()
            falses=len(self.preprocessor.y)-trues
            pos_weight = falses / trues 
            print(f"Dataset has {trues} positive samples and {falses} negative samples.")
            print(f"Using pos_weight={pos_weight:.4f} for BCEWithLogitsLoss")
            return torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32).to(device))
        return torch.nn.BCEWithLogitsLoss() 

    def set_early_stopping(self):
        config = self.configs.training['training_params']['early_stopping']
        return EarlyStopping(patience=config['patience'], min_delta=config['delta'])

    def set_optimizer(self):
        config=self.optimizer_config
        if config["name"] == "Adam":
            return torch.optim.Adam(self.model.parameters(), lr=config["learning_rate"], weight_decay=config['weight_decay'])
        elif config["name"] == "SGD":
            return torch.optim.SGD(self.model.parameters(), lr=config["learning_rate"],weight_decay=config['weight_decay'], momentum=config.get("momentum", 0.9))
        else:
            raise ValueError(f"Unsupported optimizer: {config['name']}")

    def set_scheduler(self):
        config = self.configs.training['training_params']['scheduler']
        if config['name']=='ReduceLROnPlateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=config['factor'], patience=config['patience'])
        else:
            raise ValueError(f"Unsupported scheduler: {config['name']}")

    def validate_loop(self):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in self.val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                val_outputs = self.model(x_val)
                val_loss += self.criterion(val_outputs, y_val).item()
        return val_loss 

    def test_loop(self, metrics_to_compute=None, mlflow_cfg=None):
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
            print(xx / yy if yy > 0 else 0)

    def train_loop(self):
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs} - Training...")
            for x_batch, y_batch in self.train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                # forward
                self.optimizer.zero_grad()
                outputs=self.model(x_batch)
                loss = self.criterion(outputs, y_batch)
                # backward
                loss.backward()
                self.optimizer.step()
            # validation
            val_loss = self.validate_loop()
            print(f"Epoch {epoch+1}/{self.epochs} - Validation Loss: {val_loss:.4f}")
            self.scheduler.step(val_loss)
            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                print("Early stopping triggered.")
                break



if __name__ == "__main__":
    trainer_instance = Trainer()
    trainer_instance.train_loop()
    trainer_instance.test_loop()
