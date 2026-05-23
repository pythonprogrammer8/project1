import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(logits, target, eps=1e-6):
    num_classes = logits.shape[1]
    probs = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(target.long(), num_classes).permute(0, 3, 1, 2).float()
    inter = (probs * one_hot).sum(dim=(0, 2, 3))
    den = probs.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
    return 1.0 - ((2 * inter + eps) / (den + eps)).mean()


def euclidean_distance_loss(pred, target):
    return torch.mean((pred - target) ** 2)


def reconstruction_ce_loss(sem_vox_logits, sem_vox_target):
    return F.cross_entropy(sem_vox_logits, sem_vox_target.long())


def photometric_loss(rendered, target):
    return torch.mean((rendered - target) ** 2)


def total_loss(outputs, batch, lambdas):
    seg_ce = F.cross_entropy(outputs["seg_logits"], batch["mask"].long())
    d_loss = dice_loss(outputs["seg_logits"], batch["mask"])
    ed = euclidean_distance_loss(outputs["occ"], batch["occ"].float())
    recon = reconstruction_ce_loss(outputs["sem_vox"], batch["sem_vox"].long())
    total = seg_ce + lambdas["dice"] * d_loss + lambdas["ed"] * ed + lambdas["recon"] * recon
    return total, {"seg_ce": seg_ce.item(), "dice": d_loss.item(), "ed": ed.item(), "recon": recon.item()}
