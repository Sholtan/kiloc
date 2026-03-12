import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

from numpy.typing import NDArray

from kiloc.utils.debug import print_info


def save_image_heatmaps(img_tensor, heatmap, save_path):
    """
    plots image and both its heatmaps in 1 row
    Arguments
    -------
    img_tensor: torch.Tensor (3, H, W) float32
    heatmap: (2, H', W') float32
    """
    fig, ax = plt.subplots(1, 3, figsize=(12, 10))

    ax[0].imshow(img_tensor.permute(1, 2, 0))
    ax[0].set_title("original image")

    ax[1].imshow(heatmap[0])
    ax[1].set_title("positive's heatmap")

    ax[2].imshow(heatmap[1])
    ax[2].set_title("negative's heatmap")

    plt.savefig(save_path)


def plot_image(img_tensor):
    """
    plots image
    Arguments
    -------
    img_tensor: torch.Tensor (3, H, W) float32
    """
    plt.imshow(img_tensor.permute(1, 2, 0))
    plt.show()


def plot_heatmap(heatmap):
    hm = heatmap[0].detach().cpu().numpy()
    im = plt.imshow(hm, cmap="hot", vmin=0.0, vmax=1.0)
    plt.colorbar(im, label="value")
    plt.show()


def plot_overlay_heatmap(image, pred_heatmap, gt_heatmap, alpha=0.4, interpolation=cv2.INTER_LINEAR, color_map=cv2.COLORMAP_BONE):
    """
    Overlay a heatmap on top of an image.

    Parameters
    ----------
    image : torch.Tensor | np.ndarray
        Image tensor/array, expected shape ``(3, 640, 640)`` (CHW) or ``(640, 640, 3)`` (HWC).
    heatmap : torch.Tensor | np.ndarray
        Heatmap tensor/array with shape ``(2, 160, 160)`` or ``(160, 160)``.
    alpha : float, default=0.4
        Heatmap blending strength.
    interpolation : int, default=cv2.INTER_LINEAR
        OpenCV resize interpolation mode.
    """
    # Accept torch tensors and numpy arrays.
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    if hasattr(pred_heatmap, "detach"):
        pred_heatmap = pred_heatmap.detach().cpu().numpy()
    if hasattr(gt_heatmap, "detach"):
        gt_heatmap = gt_heatmap.detach().cpu().numpy()

    image = np.asarray(image)
    pred_heatmap = np.asarray(pred_heatmap)
    gt_heatmap = np.asarray(gt_heatmap)

    if image.ndim != 3:
        raise ValueError(
            f"image must be 3D (CHW or HWC), got shape {image.shape}")
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))  # CHW -> HWC
    elif image.shape[2] != 3:
        raise ValueError(
            f"image must have 3 channels, got shape {image.shape}")

    # Convert image to uint8 (OpenCV-friendly).
    if np.issubdtype(image.dtype, np.floating):
        if image.max() <= 1.0 + 1e-6:
            img = np.clip(image, 0.0, 1.0) * 255.0
        else:
            img = np.clip(image, 0.0, 255.0)
        img = img.astype(np.uint8)
    else:
        img = np.clip(image, 0, 255).astype(np.uint8)

    pred_hm_pos = pred_heatmap[0]
    pred_hm_pos = pred_hm_pos.astype(np.float32)
    pred_hm_pos = cv2.resize(
        pred_hm_pos, (img.shape[1], img.shape[0]), interpolation=interpolation)
    pred_hm_pos = (pred_hm_pos - pred_hm_pos.min()) / \
        (pred_hm_pos.max() - pred_hm_pos.min() + 1e-8)
    pred_heatmap_color_pos = cv2.applyColorMap(
        (pred_hm_pos * 255).astype(np.uint8), color_map)
    pred_overlay_pos = cv2.addWeighted(
        img, 1.0 - alpha, pred_heatmap_color_pos, alpha, 0)

    pred_hm_neg = pred_heatmap[1]
    pred_hm_neg = pred_hm_neg.astype(np.float32)
    pred_hm_neg = cv2.resize(
        pred_hm_neg, (img.shape[1], img.shape[0]), interpolation=interpolation)
    pred_hm_neg = (pred_hm_neg - pred_hm_neg.min()) / \
        (pred_hm_neg.max() - pred_hm_neg.min() + 1e-8)
    pred_heatmap_color_neg = cv2.applyColorMap(
        (pred_hm_neg * 255).astype(np.uint8), color_map)
    pred_overlay_neg = cv2.addWeighted(
        img, 1.0 - alpha, pred_heatmap_color_neg, alpha, 0)


# **************************************************************************************************************
    gt_hm_pos = gt_heatmap[0]
    gt_hm_pos = gt_hm_pos.astype(np.float32)
    gt_hm_pos = cv2.resize(
        gt_hm_pos, (img.shape[1], img.shape[0]), interpolation=interpolation)
    gt_hm_pos = (gt_hm_pos - gt_hm_pos.min()) / \
        (gt_hm_pos.max() - gt_hm_pos.min() + 1e-8)
    gt_heatmap_color_pos = cv2.applyColorMap(
        (gt_hm_pos * 255).astype(np.uint8), color_map)
    gt_overlay_pos = cv2.addWeighted(
        img, 1.0 - alpha, gt_heatmap_color_pos, alpha, 0)

    gt_hm_neg = gt_heatmap[1]
    gt_hm_neg = gt_hm_neg.astype(np.float32)
    gt_hm_neg = cv2.resize(
        gt_hm_neg, (img.shape[1], img.shape[0]), interpolation=interpolation)
    gt_hm_neg = (gt_hm_neg - gt_hm_neg.min()) / \
        (gt_hm_neg.max() - gt_hm_neg.min() + 1e-8)
    gt_heatmap_color_neg = cv2.applyColorMap(
        (gt_hm_neg * 255).astype(np.uint8), color_map)
    gt_overlay_neg = cv2.addWeighted(
        img, 1.0 - alpha, gt_heatmap_color_neg, alpha, 0)
# **************************************************************************************************************

    fig, axes = plt.subplots(2, 3, figsize=(12, 10))

    axes[0, 0].imshow(image)
    axes[0, 0].axis("off")
    axes[1, 0].imshow(image)
    axes[1, 0].axis("off")

    axes[0, 1].imshow(gt_overlay_pos)
    axes[0, 1].axis("off")
    axes[1, 1].imshow(gt_overlay_neg)
    axes[1, 1].axis("off")

    axes[0, 2].imshow(pred_overlay_pos)
    axes[0, 2].axis("off")
    axes[1, 2].imshow(pred_overlay_neg)
    axes[1, 2].axis("off")

    axes[0, 0].set_title("Input")
    axes[0, 1].set_title("Ground Truth")
    axes[0, 2].set_title("Prediction")

    axes[0, 0].set_ylabel("Positive")
    axes[1, 0].set_ylabel("Negative")


def plot_points(image: torch.Tensor, points_gt: NDArray, pos_pred: NDArray):
    '''

    '''
    print_info(image, 'image')
    print_info(points_gt, 'points_gt')

    plt.imshow(image[0].permute(1, 2, 0))
    plt.scatter(points_gt[0][:, 0], points_gt[0][:, 1], c='blue')
    plt.scatter(pos_pred[0][:, 0], pos_pred[0][:, 1], c='red')

    plt.show()
