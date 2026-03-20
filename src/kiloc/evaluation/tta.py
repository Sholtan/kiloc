import torch

def _forward(x, idx):   # x: (B, C, H, W)
    if idx == 0: return x
    if idx == 1: return torch.rot90(x, 1, [-2,-1])
    if idx == 2: return torch.rot90(x, 2, [-2,-1])
    if idx == 3: return torch.rot90(x, 3, [-2,-1])
    if idx == 4: return torch.flip(x, [-1])
    if idx == 5: return torch.rot90(torch.flip(x, [-1]), 1, [-2,-1])
    if idx == 6: return torch.rot90(torch.flip(x, [-1]), 2, [-2,-1])
    if idx == 7: return torch.rot90(torch.flip(x, [-1]), 3, [-2,-1])

def _inverse(x, idx):   # x: (B, 2, H, W)
    if idx == 0: return x
    if idx == 1: return torch.rot90(x, 3, [-2,-1])
    if idx == 2: return torch.rot90(x, 2, [-2,-1])
    if idx == 3: return torch.rot90(x, 1, [-2,-1])
    if idx == 4: return torch.flip(x, [-1])
    if idx == 5: return torch.flip(torch.rot90(x, 3, [-2,-1]), [-1])
    if idx == 6: return torch.flip(torch.rot90(x, 2, [-2,-1]), [-1])
    if idx == 7: return torch.flip(torch.rot90(x, 1, [-2,-1]), [-1])

@torch.no_grad()
def tta_forward(model, img_batch):  # returns (B, 2, H, W) averaged sigmoid heatmap
    acc = None
    for idx in range(8):
        hm = torch.sigmoid(model(_forward(img_batch, idx)))
        hm = _inverse(hm, idx)
        acc = hm if acc is None else acc + hm
    return acc / 8.0