import yaml
from pathlib import Path

paths_dir = Path(__file__).parent.parent.parent.parent / \
    "configs" / "paths.yaml"

SUPPORTED_DEVICES = {"hpvictus", "collab", "h200"}


def get_paths(device):
    if device not in SUPPORTED_DEVICES:
        raise ValueError(f"the device must be one of the {SUPPORTED_DEVICES}")

    with open(paths_dir) as f:
        config = yaml.safe_load(f)

        data_root = Path(config[device + "_paths"]["data_root"])
        checkpoint_path = Path(config[device + "_paths"]["checkpoint_dir"])
    return data_root, checkpoint_path
