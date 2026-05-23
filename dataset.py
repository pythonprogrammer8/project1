from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class HeritageDataset(Dataset):
    def __init__(self, root, split="train", input_size=224, num_classes=5):
        self.root = Path(root)
        self.input_size = input_size
        self.num_classes = num_classes
        self.images = sorted((self.root / "images").glob("*.*"))
        split_file = self.root / f"{split}.txt"
        if split_file.exists():
            names = {x.strip() for x in split_file.read_text().splitlines() if x.strip()}
            self.images = [p for p in self.images if p.stem in names]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        ip = self.images[idx]
        image = Image.open(ip).convert("RGB").resize((self.input_size, self.input_size))
        mask_path = self.root / "masks" / f"{ip.stem}.png"
        if mask_path.exists():
            mask = Image.open(mask_path).resize((self.input_size, self.input_size), Image.NEAREST)
            mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        else:
            mask = torch.zeros(self.input_size, self.input_size, dtype=torch.long)

        voxel_path = self.root / "voxels" / f"{ip.stem}.npz"
        if voxel_path.exists():
            data = np.load(voxel_path)
            occ = torch.from_numpy(data["occ"]).float().unsqueeze(0)
            sem_vox = torch.from_numpy(data["sem"]).long()
        else:
            occ = torch.zeros(1, 32, 32, 32)
            sem_vox = torch.zeros(32, 32, 32, dtype=torch.long)

        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return {"image": image, "mask": mask, "occ": occ, "sem_vox": sem_vox, "name": ip.stem}
