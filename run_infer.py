import argparse

from src.entities.reggs import RegGS
from src.utils.io_utils import load_config
from src.utils.utils import setup_seed


def get_args():
    parser = argparse.ArgumentParser(
        description='Arguments to compute the mesh')
    parser.add_argument('config_path', type=str,
                        help='Path to the configuration yaml file')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    config = load_config(args.config_path)

    setup_seed(config["seed"])
    reggs = RegGS(config)
    reggs.run()
