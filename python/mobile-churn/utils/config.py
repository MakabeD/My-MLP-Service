import argparse

import yaml
from config_path import config_dir_list

CONFIGS_PATH = "./configs"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load Config pipeline for mobile-churun"
    )
    parser.add_argument(
        "--config",
        type=int,
        required=True,
        help="rutal al archivo",
    )
    return parser.parse_args()


class Config:
    def __init__(self, dir_index):
        config_dir_name = self.get_configs_dirname_byIndex(dir_index)
        self.load_config(config_dir_name)

    def get_configs_dirname_byIndex(self, dir_index) -> str:
        dir_name_list = config_dir_list(CONFIGS_PATH)
        try:
            index_dir_name = dir_name_list[dir_index]
        except IndexError:
            print(dir_name_list)
            print([i for i in range(len(dir_name_list))])
            raise IndexError(
                f"--Config argument index is out of range: {len(dir_name_list)}"
            )
        return index_dir_name

    def load_mlflow_config(self, dir_name: str):
        with open("./configs/" + dir_name + "/mlflow.yaml", "r") as f:
            config = yaml.safe_load(f)
        return config

    def load_model_config(self, dir_name: str):
        with open("./configs/" + dir_name + "/model.yaml", "r") as f:
            config = yaml.safe_load(f)
        return config

    def load_training_config(self, dir_name: str):
        with open("./configs/" + dir_name + "/training.yaml", "r") as f:
            config = yaml.safe_load(f)
            return config

    def load_config(self, dir_name):
        self.model = self.load_model_config(dir_name)
        self.mlflow = self.load_mlflow_config(dir_name)
        self.training = self.load_training_config(dir_name)


if __name__ == "__main__":
    dir_index = parse_args().config
    config = Config(dir_index)
    for i, y in config.mlflow.items():
        print(i, (y.values()))
