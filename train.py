import argparse
import random
import time
import os

import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import HeritageDataset
from models import EndToEndHongcunModel
from losses import total_loss
from metrics import pixel_accuracy, miou, dice_coef

try:
    from skimage.metrics import structural_similarity as ssim_metric
    from skimage.metrics import peak_signal_noise_ratio as psnr_metric
except:
    ssim_metric = None
    psnr_metric = None

try:
    import lpips
except:
    lpips = None


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def chamfer_distance(pred, target):
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)
    return np.mean((pred - target) ** 2)


def iou_3d(pred, target):
    pred_bin = pred > 0
    tgt_bin = target > 0

    intersection = np.logical_and(pred_bin, tgt_bin).sum()
    union = np.logical_or(pred_bin, tgt_bin).sum()

    if union == 0:
        return 1.0

    return intersection / union


def calculate_fps(model, device, input_size=224, num_runs=50):

    model.eval()

    dummy = torch.randn(1, 3, input_size, input_size).to(device)

    for _ in range(10):
        _ = model(dummy)

    if device == "cuda":
        torch.cuda.synchronize()

    start = time.time()

    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy)

    if device == "cuda":
        torch.cuda.synchronize()

    end = time.time()

    total_time = end - start
    fps = num_runs / total_time

    return fps


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--data_root",
        default=r"Enter your dataset root path here"
    )

    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num_classes", type=int, default=5)
    ap.add_argument("--input_size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="checkpoints/best.pt")

    args = ap.parse_args()

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nUsing Device : {device}")

    train_ds = HeritageDataset(
        args.data_root,
        "train",
        args.input_size,
        args.num_classes
    )

    val_ds = HeritageDataset(
        args.data_root,
        "val",
        args.input_size,
        args.num_classes
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    model = EndToEndHongcunModel(
        args.num_classes,
        voxel_size=32,
        pretrained=True
    ).to(device)

    opt = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    lambdas = {
        "ed": 1.0,
        "recon": 1.0,
        "dice": 1.0
    }

    os.makedirs(
        os.path.dirname(args.out),
        exist_ok=True
    )

    best = -1

    if lpips is not None:
        lpips_model = lpips.LPIPS(net='alex').to(device)
    else:
        lpips_model = None

    for epoch in range(1, args.epochs + 1):

        model.train()

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{args.epochs}"
        )

        for batch in pbar:

            batch = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }

            opt.zero_grad()

            outputs = model(batch["image"])

            loss, logs = total_loss(
                outputs,
                batch,
                lambdas
            )

            loss.backward()

            opt.step()

            pbar.set_postfix(
                loss=float(loss),
                **logs
            )

        model.eval()

        vals = []

        with torch.no_grad():

            for batch in val_loader:

                img = batch["image"].to(device)

                outputs = model(img)

                pred = torch.argmax(
                    outputs["seg_logits"],
                    dim=1
                ).cpu().numpy()[0]

                tgt = batch["mask"].numpy()[0]

                pa_score = pixel_accuracy(pred, tgt)

                miou_score = miou(
                    pred,
                    tgt,
                    args.num_classes
                )

                dice_score = dice_coef(
                    pred,
                    tgt,
                    args.num_classes
                )

                cd_score = chamfer_distance(pred, tgt)

                iou3d_score = iou_3d(pred, tgt)

                pred_float = pred.astype(np.float32)
                tgt_float = tgt.astype(np.float32)

                if ssim_metric is not None:

                    data_range = float(
                        max(tgt_float.max(), pred_float.max()) -
                        min(tgt_float.min(), pred_float.min())
                    )

                    if data_range == 0:
                        data_range = 1.0

                    ssim_score = ssim_metric(
                        pred_float,
                        tgt_float,
                        data_range=data_range
                    )

                else:
                    ssim_score = 0

                mse = np.mean((tgt_float - pred_float) ** 2)

                if mse == 0:

                    psnr_score = 100.0

                else:

                    data_range = float(
                        max(tgt_float.max(), pred_float.max()) -
                        min(tgt_float.min(), pred_float.min())
                    )

                    if data_range == 0:
                        data_range = 1.0

                    if psnr_metric is not None:

                        psnr_score = psnr_metric(
                            tgt_float,
                            pred_float,
                            data_range=data_range
                        )

                    else:

                        psnr_score = (
                            20 * np.log10(
                                data_range / np.sqrt(mse)
                            )
                        )

                if lpips_model is not None:

                    pred_tensor = torch.tensor(pred_float).unsqueeze(0).unsqueeze(0)
                    tgt_tensor = torch.tensor(tgt_float).unsqueeze(0).unsqueeze(0)

                    pred_tensor = pred_tensor.repeat(1, 3, 1, 1).float().to(device)
                    tgt_tensor = tgt_tensor.repeat(1, 3, 1, 1).float().to(device)

                    lpips_score = lpips_model(
                        pred_tensor,
                        tgt_tensor
                    ).item()

                else:
                    lpips_score = 0

                vals.append({
                    "pa": pa_score,
                    "miou": miou_score,
                    "dice": dice_score,
                    "cd": cd_score,
                    "iou3d": iou3d_score,
                    "ssim": ssim_score,
                    "psnr": psnr_score,
                    "lpips": lpips_score
                })

        avg_pa = np.mean([x["pa"] for x in vals])
        avg_miou = np.mean([x["miou"] for x in vals])
        avg_dice = np.mean([x["dice"] for x in vals])
        avg_cd = np.mean([x["cd"] for x in vals])
        avg_iou3d = np.mean([x["iou3d"] for x in vals])
        avg_ssim = np.mean([x["ssim"] for x in vals])
        avg_psnr = np.mean([x["psnr"] for x in vals])
        avg_lpips = np.mean([x["lpips"] for x in vals])

        fps = calculate_fps(
            model,
            device,
            args.input_size
        )

        print(f"\nEpoch {epoch}/{args.epochs}")

        print(f"mIoU   : {avg_miou:.4f}")
        print(f"PA     : {avg_pa:.4f}")
        print(f"Dice   : {avg_dice:.4f}")
        print(f"CD     : {avg_cd:.4f}")
        print(f"3DIoU  : {avg_iou3d:.4f}")
        print(f"SSIM   : {avg_ssim:.4f}")
        print(f"PSNR   : {avg_psnr:.4f}")
        print(f"LPIPS  : {avg_lpips:.4f}")
        print(f"FPS    : {fps:.2f}")

        if avg_miou > best:

            best = avg_miou

            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args)
                },
                args.out
            )

            print(f"Saved : {args.out}")


if __name__ == "__main__":
    main()