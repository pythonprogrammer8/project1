import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric


def pixel_accuracy(pred, target):
    return (pred == target).sum() / max(target.size, 1)


def miou(pred, target, num_classes):
    ious = []
    for c in range(num_classes):
        p = pred == c
        t = target == c
        inter = np.logical_and(p, t).sum()
        union = np.logical_or(p, t).sum()
        if union > 0:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


def dice_coef(pred, target, num_classes, eps=1e-6):
    vals = []
    for c in range(num_classes):
        p = pred == c
        t = target == c
        vals.append((2 * np.logical_and(p, t).sum() + eps) / (p.sum() + t.sum() + eps))
    return float(np.mean(vals))


def chamfer_distance(points_a, points_b):
    a = np.asarray(points_a, dtype=np.float32)
    b = np.asarray(points_b, dtype=np.float32)
    if len(a) == 0 or len(b) == 0:
        return np.inf
    diff = a[:, None, :] - b[None, :, :]
    dist = np.sum(diff * diff, axis=-1)
    return float(np.mean(np.min(dist, axis=1)) + np.mean(np.min(dist, axis=0)))


def iou3d(occ_pred, occ_true, threshold=0.5):
    p = occ_pred > threshold
    t = occ_true > threshold
    inter = np.logical_and(p, t).sum()
    union = np.logical_or(p, t).sum()
    return float(inter / max(union, 1))


def ssim_rgb(img_a, img_b):
    return float(ssim_metric(img_a, img_b, channel_axis=-1, data_range=1.0))


def psnr_rgb(img_a, img_b):
    return float(psnr_metric(img_b, img_a, data_range=1.0))
