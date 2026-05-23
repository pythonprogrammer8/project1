"""
PhotoRecon-GS hook.

For production-quality Gaussian Splatting, use the official 3D Gaussian Splatting renderer
with COLMAP poses. This file provides paper-aligned initialization utilities:
semantic-guided Gaussian points, covariance placeholders, spherical harmonic coefficients,
and photometric optimization hooks.
"""
import numpy as np


def voxel_to_gaussians(occ, sem, threshold=0.5):
    coords = np.argwhere(occ > threshold)
    if len(coords) == 0:
        return {}
    xyz = coords.astype(np.float32) / max(occ.shape) * 2.0 - 1.0
    labels = sem[coords[:, 0], coords[:, 1], coords[:, 2]]
    cov = np.tile(np.eye(3, dtype=np.float32)[None] * 0.01, (len(xyz), 1, 1))
    opacity = np.ones((len(xyz), 1), dtype=np.float32) * 0.8
    sh = np.zeros((len(xyz), 16, 3), dtype=np.float32)  # third-order SH placeholder
    return {"xyz": xyz, "semantic": labels, "covariance": cov, "opacity": opacity, "sh": sh}


def save_ply(path, xyz):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for p in xyz:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")
