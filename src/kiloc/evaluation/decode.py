"""
Extraction of cell centers from the heatmap
"""
import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

def _as_pair(x, name: str) -> tuple[float, float]:
    if isinstance(x, (tuple, list)):
        if len(x) != 2:
            raise ValueError(f"{name} must be a scalar or length-2 tuple/list, got {x}")
        return float(x[0]), float(x[1])
    return float(x), float(x)


@torch.no_grad()
def heatmaps_to_points_batch(
    heatmaps: torch.Tensor,
    kernel_size: int,
    threshold: float | tuple[float, float],
    output_hw: tuple = (640, 640),
    merge_radius: float = 1.5,
    refine: bool = True,
):
    """
        Recover points centers from batched 2-channel heatmaps

        Parameters
        ----------
        heatmaps:
            Tensor with shape (B, 2, H, W) and values in range [0., 1.]
        kernel_size:
            Odd number kernel for local-maxima suppression
        threshold:
            Peak threshold in heatmap space
        output_hw:
            rescales points from heatmap coordinates
            to output image coordinates
        merge_radius:
            Merge peaks closer than thios radius
        refine:
            Use weighted local centroid refinement around each detected peak
        Returns
        ----------
        
    """
    if heatmaps.ndim != 4 or heatmaps.shape[1] != 2:
        raise ValueError(
            f"Expected heatmaps with shape (B, 2, H, W), got {heatmaps.shape}")
    if kernel_size % 2 == 0 or kernel_size < 1:
        raise ValueError(f"kernel size must be an odd positive integer")

    out_pos = []    # list of coordinates for positive cells
    out_neg = []    # list of coordinates for negative cells

    # height and width of the heatmaps
    hm_h, hm_w = int(heatmaps.shape[-2]), int(heatmaps.shape[-1])

    for b in range(heatmaps.shape[0]):   # loop over batch samples
        pos_pts, neg_pts = heatmaps_to_points(
            heatmaps[b], kernel_size=kernel_size, threshold=threshold,
            merge_radius=merge_radius, refine=refine
        )

        if output_hw is not None:
            pos_pts = _rescale_points(
                pos_pts, src_hw=(hm_h, hm_w), dst_hw=output_hw)
            neg_pts = _rescale_points(
                neg_pts, src_hw=(hm_h, hm_w), dst_hw=output_hw)

        out_pos.append(pos_pts)
        out_neg.append(neg_pts)

    return out_pos, out_neg


def heatmaps_to_points(
        heatmap2d: torch.Tensor, 
        kernel_size: int, 
        threshold: float | tuple[float, float], 
        merge_radius:float =1.5, 
        refine: bool = True
):
    """
    Recover points from a single sample heatmap with shape (2, H, W).
    Returns (pos_pts, neg_pts) as np arrays of shape (N, 2) in (x,y) order. 
    """
    if heatmap2d.ndim != 3 or heatmap2d.shape[0] != 2:
        raise ValueError(
            f"Expected heatmap2d with shape (2, H, W), got {heatmap2d.shape}")

    thr_pos, thr_neg = _as_pair(threshold, "threshold")

    pos_heatmap = heatmap2d[0]
    neg_heatmap = heatmap2d[1]

    pos_pts = _channel_to_points(
        pos_heatmap, kernel_size=kernel_size, threshold=thr_pos,
        merge_radius=merge_radius, refine=refine
    )
    neg_pts = _channel_to_points(
        neg_heatmap, kernel_size=kernel_size, threshold=thr_neg,
        merge_radius=merge_radius, refine=refine
    )

    return pos_pts, neg_pts


def _channel_to_points(hm, kernel_size, threshold, merge_radius, refine):
    """
    Recover points from single channel with shape (H, W).

    Returns
    ----------
    NDArray[np.float32], shape (N, 2)

    """
    mask = _local_maxima(hm, kernel_size) & (hm >= threshold)
    ys, xs = torch.where(mask)  # torch.where returns in (y, x) order

    if xs.numel() < 1:
        return np.zeros((0, 2), dtype=np.float32)

    scores = hm[ys, xs]
    pts = torch.stack((xs, ys), dim=1).to(
        torch.float32)    # shape (N,2) in (x, y) order

    if merge_radius is not None and merge_radius > 0.:
        pts, scores = _merge_close_points(
            pts, scores, merge_radius=merge_radius)

    if refine and pts.numel() > 0:
        pts = _refine_points(hm, pts)

    return pts.cpu().numpy()


def _local_maxima(hm, kernel_size):
    """
    Finds the local maximums for a single heatmap channel.
    Returns
    ----------
    The (H,W) np array with boolean values
    """
    pad = kernel_size // 2
    h = hm.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    pooled = F.max_pool2d(h, kernel_size=kernel_size, stride=1, padding=pad)
    return pooled[0, 0] == hm


def _merge_close_points(pts, scores, merge_radius):
    if pts.numel() == 0:
        return pts, scores

    order = scores.argsort(descending=True)
    pts = pts[order]
    scores = scores[order]

    keep_pts = []
    keep_scores = []

    suppressed = torch.zeros(len(pts), dtype=torch.bool, device=pts.device)
    r2 = merge_radius * merge_radius

    for i in range(len(pts)):
        if suppressed[i]:
            continue

        keep_pts.append(pts[i])
        keep_scores.append(scores[i])

        dx = pts[:, 0] - pts[i, 0]
        dy = pts[:, 1] - pts[i, 1]

        suppressed |= (dx * dx + dy * dy) <= r2

    return torch.stack(keep_pts, dim=0), torch.stack(keep_scores, dim=0)


def _refine_points(hm, pts, beta=4.0, radius=3, eps=1e-8):
    refined = torch.empty_like(pts)
    H, W = hm.shape

    for i, p in enumerate(pts):
        x0 = int(round(float(p[0].item())))
        y0 = int(round(float(p[1].item())))

        x1 = max(0, x0 - radius)
        x2 = min(W, x0 + radius + 1)
        y1 = max(0, y0 - radius)
        y2 = min(H, y0 + radius + 1)

        patch = hm[y1:y2, x1:x2].clamp_min(0.0)
        w = patch.pow(beta)
        s = w.sum()
        if s.item() < eps:
            refined[i, 0] = float(x0)
            refined[i, 1] = float(y0)
            continue

        ys = torch.arange(y1, y2, device=hm.device,
                          dtype=torch.float32).view(-1, 1)
        xs = torch.arange(x1, x2, device=hm.device,
                          dtype=torch.float32).view(1, -1)
        refined[i, 0] = (w * xs).sum() / s
        refined[i, 1] = (w * ys).sum() / s

    return refined


def _rescale_points(pts, src_hw, dst_hw):
    if pts.size == 0:
        return pts

    src_h, src_w = src_hw
    dst_h, dst_w = dst_hw
    sx = float(dst_w) / float(src_w)
    sy = float(dst_h) / float(src_h)

    out = pts.astype(np.float32, copy=True)
    out[:, 0] *= sx
    out[:, 1] *= sy
    return out
