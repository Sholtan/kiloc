'''
Checks the coordinate convention in annotations
'''


import h5py
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pathlib


from kiloc.utils.config import get_paths


dataroot, _ = get_paths('hpvictus')

img_path = next((dataroot / "images/train").glob("*.png"))

stem = img_path.stem

h5_pos = dataroot / "annotations/train/positive" / (f"{stem}" + ".h5")

with h5py.File(h5_pos, 'r') as f:
    coords = f["coordinates"][:]


img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
fix, ax = plt.subplots()

ax.imshow(img)

ax.scatter(coords[:5, 0], coords[:5, 1], c="red", s=10)
plt.savefig("scripts/check_convention.png")
