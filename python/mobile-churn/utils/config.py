import argparse

import yaml
from config_path import config_dir_list, print_dir_list

CONFIGS_PATH = "./configs"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load Config pipeline for mobile-churun"
    )
    parser.add_argument(
        "--config",
        type=int,
        required=True,
        help=("Index of the config folder to load.\nExample:\n  --config 0\n"),
    )
    return parser.parse_args()


class Config:
    def __init__(self, dir_index):
        print(f"[INFO] Loading configuration using index: {dir_index}")
        config_dir_name = self.get_configs_dirname_byIndex(dir_index)
        self.load_config(config_dir_name)
        print(f"[SUCCESS] Configuration '{config_dir_name}' loaded successfully\n")

    def get_configs_dirname_byIndex(self, dir_index) -> str:
        dir_name_list = config_dir_list(CONFIGS_PATH)

        print("[INFO] Available configuration directories:")
        print_dir_list(dir_name_list)

        try:
            index_dir_name = dir_name_list[dir_index]
        except IndexError:
            if len(dir_name_list) <= 0:
                print(
                    "\n[ERROR] There is no directory '^[v][0-9]+$' in the ./utils folder."
                )
            else:
                print("\n[ERROR] Invalid --config index provided.")
                print(
                    f"[ERROR] Index {dir_index} is out of range "
                    f"(valid range: 0 to {len(dir_name_list) - 1})"
                )
            raise IndexError(
                f"--config argument index is out of range: {len(dir_name_list)}"
            )
        print(f"[INFO] Selected config directory: {index_dir_name}\n")
        return index_dir_name

    def load_mlflow_config(self, dir_name: str):
        print("[INFO] Loading mlflow.yaml")
        with open("./configs/" + dir_name + "/mlflow.yaml", "r") as f:
            config = yaml.safe_load(f)
        return config

    def load_model_config(self, dir_name: str):
        print("[INFO] Loading model.yaml")
        with open("./configs/" + dir_name + "/model.yaml", "r") as f:
            config = yaml.safe_load(f)
        return config

    def load_training_config(self, dir_name: str):
        print("[INFO] Loading training.yaml")
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
