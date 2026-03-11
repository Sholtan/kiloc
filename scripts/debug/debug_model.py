from kiloc.model.kiloc_net import KiLocNet
from kiloc.utils.debug import print_info


from kiloc.datasets import BCDataDataset
from kiloc.utils.config import get_paths
from kiloc.target_generation.heatmaps import LocHeatmap


from torch.utils.data import DataLoader
import torch
from pathlib import Path

# Instantiate
model = KiLocNet(pretrained=False)
model.eval()

# Forward pass on random input
x = torch.randn(2, 3, 640, 640)
with torch.no_grad():
    out = model(x)

assert out.shape == (2, 2, 160, 160), out.shape
print_info(out)

# Check output is not degenerate
assert not torch.isnan(out).any(), "NaN in output"
assert not torch.isinf(out).any(), "Inf in output"

# Forward pass on a real batch from the DataLoader
root_dir, _ = get_paths("hpvictus")
heatmap_generator = LocHeatmap(out_hw=(160, 160), in_hw=(
    640, 640), sigma=3.0, dtype=torch.float32)
dataset = BCDataDataset(root=root_dir, split='train', target_transform=heatmap_generator,
                        image_transform=None, joint_transform=None)
loader = DataLoader(dataset, batch_size=4, num_workers=0)

imgs, heatmaps = next(iter(loader))
with torch.no_grad():
    out = model(imgs)

assert out.shape == (4, 2, 160, 160)

# Backward pass check
model.train()
out = model(imgs)
loss = out.mean()
loss.backward()
print("Backward pass OK")
print("\nModel debug finished")
