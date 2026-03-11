import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from utils.config_path import config_dir_list

from src.train.trainer import Trainer


def run():
    amount = len(config_dir_list("./configs"))
    for i in range(amount):
        trainer = Trainer(i, "Experiment: 0.1")
        trainer.run_train()
    print(
        "\n \n",
        "[Success] All configs runs finished successfully. Please refer to the mlflow user interface for the results.",
    )


if __name__ == "__main__":
    run()
