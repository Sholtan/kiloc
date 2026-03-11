import torch

from kiloc.losses.losses import sigmoid_weighted_mse_loss
from kiloc.utils.debug import print_info

datashape = (7, 2, 160, 160)
a = torch.randn(*datashape)
b = torch.rand(*datashape)


loss = sigmoid_weighted_mse_loss(a, b)

print_info(loss, 'loss')

assert loss.numel() == 1, loss.numel()
print(f"debug weighted mse done")
