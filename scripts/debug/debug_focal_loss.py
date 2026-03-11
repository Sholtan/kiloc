import torch

from kiloc.losses.losses import sigmoid_focal_loss
from kiloc.utils.debug import print_info

datashape = (7, 2, 160, 160)
a = torch.randn(*datashape)
b = torch.rand(*datashape)


number_of_cells = 50

focalloss = sigmoid_focal_loss(a, b, number_of_cells)
print_info(focalloss, "focalloss")

assert focalloss.numel() == 1, focalloss.numel()
print(f"debug focal loss is done")
