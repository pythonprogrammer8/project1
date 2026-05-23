# CSA-DeepLabv3+ + SeDReNet + PhotoRecon-GS Code

This code implements the architecture described in the uploaded paper:

- CSA-DeepLabv3+ for semantic segmentation
- SeDReNet for semantic voxel reconstruction
- PhotoRecon-GS-style Gaussian rendering hooks
- ED, reconstruction CE, photometric, and Dice losses
- mIoU, PA, Dice, CD, 3DIoU, SSIM, PSNR, LPIPS, FPS metrics

## Dataset layout

```text
dataset/
  images/
    xxx.jpg
  masks/
    xxx.png        # semantic mask, class indices
  depth/
    xxx.npy        # optional depth map
  voxels/
    xxx.npz        # occupancy and semantic voxel labels
```

## Run

```bash
pip install -r requirements.txt
python train.py --data_root ./r"D:\SUMOSE S\PYTHON WORKS\Work_174_code(APR25-YUAN-1112)\dataset\archive (5)\m60" --epochs 500 --batch_size 32 --lr 1e-4 --num_classes 5
python infer.py --checkpoint checkpoints/best.pt --image path/to/image.jpg --out_dir outputs
```

For real Hongcun/Tanks-and-Temples training, prepare masks, depth/pose, and voxel/point-cloud supervision or pseudo-labels.
