from pathlib import Path
import json

from kiloc.datasets import BCDataPointDataset
from kiloc.utils.config import get_paths

data_root, _ = get_paths(device="h200")

bg_stats_path = Path("configs/bcdata_background_median.json")
with bg_stats_path.open("r", encoding="utf-8") as f:
    bg_stats = json.load(f)

background_rgb = tuple(bg_stats["median_rgb_uint8_rounded"])  # (195, 188, 182)

train_ds = BCDataPointDataset(
    data_root=data_root,
    points_csv=Path("bcdata_point_tables/train_points.csv"),
    crop_size=96,
    resize_to=128,  # or 48 / None depending on experiment
    background_rgb=background_rgb,
    image_transform=None,
    input_normalization="imagenet",
)