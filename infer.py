import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from models import EndToEndHongcunModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out_dir", default="outputs")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    num_classes = ckpt.get("args", {}).get("num_classes", 5)
    model = EndToEndHongcunModel(num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    out_dir = Path(args.out_dir); out_dir.mkdir(exist_ok=True)
    img = Image.open(args.image).convert("RGB").resize((224, 224))
    x = TF.normalize(TF.to_tensor(img), [0.485,0.456,0.406], [0.229,0.224,0.225]).unsqueeze(0)

    with torch.no_grad():
        out = model(x)
        mask = torch.argmax(out["seg_logits"], dim=1)[0].numpy().astype(np.uint8)
        occ = out["occ"][0, 0].numpy()

    Image.fromarray((mask * (255 // max(num_classes - 1, 1))).astype(np.uint8)).save(out_dir / "semantic_mask.png")
    np.save(out_dir / "occupancy_32x32x32.npy", occ)
    print(f"saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
