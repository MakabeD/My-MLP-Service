import yaml
import argparse
def parse_args():
    parser=argparse.ArgumentParser(description="Load Config pipeline for mobile-churun")
    parser.add_argument(
        "--config",
        type=int,
        required=True,
        help="rutal al archivo",
    )
    return parser.parse_args()

def open_test():
    with open("./configs/v1/model.yaml", "r")as f:
        config = yaml.safe_load(f)

    print(config)

if __name__ =="__main__":
    print(parse_args())
    ##open_test()