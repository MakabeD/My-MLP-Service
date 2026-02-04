import yaml
import argparse
from config_path import list_config_dir
CONFIGS_PATH='./configs'
def parse_args():
    parser=argparse.ArgumentParser(description="Load Config pipeline for mobile-churun")
    parser.add_argument(
        "--config",
        type=int,
        required=True,
        help="rutal al archivo",
    )
    return parser.parse_args()

def initialize():
    dir_index = parse_args().config
    dir_name_list=list_config_dir(CONFIGS_PATH)
    try:
        index_dir_name=dir_name_list[dir_index]
    except IndexError:
        print(dir_name_list)
        print([i for i in range(len(dir_name_list))])
        raise IndexError(f"--Config argument index is out of range: {len(dir_name_list)}")

    print(index_dir_name)


def get_model():
    with open("./configs/v1/model.yaml", "r")as f:
        config = yaml.safe_load(f)
    return(config)

if __name__ =="__main__":
    initialize()