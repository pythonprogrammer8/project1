import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet101, ResNet101_Weights


class CoordConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels + 2, out_channels, kernel_size, stride, padding, dilation=dilation, bias=bias)

    def forward(self, x):
        b, _, h, w = x.shape
        yy = torch.linspace(-1, 1, h, device=x.device).view(1, 1, h, 1).expand(b, 1, h, w)
        xx = torch.linspace(-1, 1, w, device=x.device).view(1, 1, 1, w).expand(b, 1, h, w)
        return self.conv(torch.cat([x, xx, yy], dim=1))


class ECA(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

    def forward(self, x):
        y = F.adaptive_avg_pool2d(x, 1).squeeze(-1).transpose(-1, -2)
        y = torch.sigmoid(self.conv(y)).transpose(-1, -2).unsqueeze(-1)
        return x * y


class ChannelAttention(nn.Module):
    def __init__(self, channels, ratio=16):
        super().__init__()
        hidden = max(channels // ratio, 4)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

    def forward(self, x):
        avg = self.mlp(F.adaptive_avg_pool2d(x, 1))
        mx = self.mlp(F.adaptive_max_pool2d(x, 1))
        return x * torch.sigmoid(avg + mx)


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        att = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * att


class CSAModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.eca = ECA(channels)
        self.ca = ChannelAttention(channels)
        self.spa = SpatialAttention()
        self.weights = nn.Parameter(torch.ones(3))

    def forward(self, x):
        w = torch.softmax(self.weights, dim=0)
        out = w[0] * self.ca(x) + w[1] * self.eca(x) + w[2] * self.spa(x)
        return x * torch.sigmoid(out)


class MRDPP(nn.Module):
    def __init__(self, in_ch, out_ch=256):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(True)),
            nn.Sequential(CoordConv2d(in_ch, out_ch, 3, padding=6, dilation=6), nn.BatchNorm2d(out_ch), nn.ReLU(True)),
            nn.Sequential(CoordConv2d(in_ch, out_ch, 3, padding=12, dilation=12), nn.BatchNorm2d(out_ch), nn.ReLU(True)),
            nn.Sequential(CoordConv2d(in_ch, out_ch, 3, padding=18, dilation=18), nn.BatchNorm2d(out_ch), nn.ReLU(True)),
        ])
        self.global_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_ch, out_ch, 1), nn.ReLU(True))
        self.csa = CSAModule(out_ch)
        self.project = nn.Sequential(nn.Conv2d(out_ch * 5, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(True), nn.Dropout(0.5))

    def forward(self, x):
        h, w = x.shape[-2:]
        feats = [self.csa(branch(x)) for branch in self.branches]
        gp = F.interpolate(self.global_pool(x), size=(h, w), mode="bilinear", align_corners=False)
        return self.project(torch.cat(feats + [gp], dim=1))


class CSADeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        weights = ResNet101_Weights.DEFAULT if pretrained else None
        backbone = resnet101(weights=weights, replace_stride_with_dilation=[False, True, True])
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.mrdpp = MRDPP(2048, 256)
        self.low_proj = nn.Sequential(nn.Conv2d(256, 48, 1, bias=False), nn.BatchNorm2d(48), nn.ReLU(True))
        self.decoder = nn.Sequential(
            CoordConv2d(304, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            CoordConv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = self.stem(x)
        low = self.layer1(x)
        x = self.layer2(low)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.mrdpp(x)
        x = F.interpolate(x, size=low.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, self.low_proj(low)], dim=1)
        logits = self.decoder(x)
        return F.interpolate(logits, size=size, mode="bilinear", align_corners=False)


class CoordConv3d(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, padding=1, dilation=1):
        super().__init__()
        self.conv = nn.Conv3d(in_ch + 3, out_ch, k, padding=padding, dilation=dilation)

    def forward(self, x):
        b, _, d, h, w = x.shape
        zz = torch.linspace(-1, 1, d, device=x.device).view(1, 1, d, 1, 1).expand(b, 1, d, h, w)
        yy = torch.linspace(-1, 1, h, device=x.device).view(1, 1, 1, h, 1).expand(b, 1, d, h, w)
        xx = torch.linspace(-1, 1, w, device=x.device).view(1, 1, 1, 1, w).expand(b, 1, d, h, w)
        return self.conv(torch.cat([x, xx, yy, zz], dim=1))


class Residual3DBlock(nn.Module):
    def __init__(self, ch, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            CoordConv3d(ch, ch, 3, padding=dilation, dilation=dilation), nn.BatchNorm3d(ch), nn.ReLU(True),
            CoordConv3d(ch, ch, 3, padding=dilation, dilation=dilation), nn.BatchNorm3d(ch)
        )

    def forward(self, x):
        return F.relu(x + self.net(x), inplace=True)


class SeDReNet(nn.Module):
    def __init__(self, num_classes=5, voxel_size=32):
        super().__init__()
        self.voxel_size = voxel_size
        self.encoder = nn.Sequential(
            CoordConv2d(3 + num_classes, 64, 3), nn.ReLU(True), nn.MaxPool2d(2),
            CoordConv2d(64, 128, 3), nn.ReLU(True), nn.MaxPool2d(2),
            CoordConv2d(128, 256, 3, dilation=2, padding=2), nn.ReLU(True), nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(256, 2048), nn.ReLU(True), nn.Dropout(0.5),
                                nn.Linear(2048, 4 * 4 * 4 * 128), nn.ReLU(True))
        self.decoder = nn.Sequential(
            Residual3DBlock(128, dilation=1),
            nn.ConvTranspose3d(128, 64, 4, stride=2, padding=1), nn.ReLU(True),
            Residual3DBlock(64, dilation=2),
            nn.ConvTranspose3d(64, 32, 4, stride=2, padding=1), nn.ReLU(True),
            Residual3DBlock(32, dilation=2),
            nn.ConvTranspose3d(32, 32, 4, stride=2, padding=1), nn.ReLU(True),
        )
        self.occ_head = nn.Conv3d(32, 1, 1)
        self.sem_head = nn.Conv3d(32, num_classes, 1)

    def forward(self, image, seg_logits):
        seg_prob = torch.softmax(seg_logits, dim=1)
        x = torch.cat([image, seg_prob], dim=1)
        z = self.encoder(x)
        z = self.fc(z).view(image.size(0), 128, 4, 4, 4)
        v = self.decoder(z)
        if v.shape[-1] != self.voxel_size:
            v = F.interpolate(v, size=(self.voxel_size, self.voxel_size, self.voxel_size), mode="trilinear", align_corners=False)
        occ = torch.sigmoid(self.occ_head(v))
        sem = self.sem_head(v)
        return occ, sem


class EndToEndHongcunModel(nn.Module):
    def __init__(self, num_classes=5, voxel_size=32, pretrained=True):
        super().__init__()
        self.segmenter = CSADeepLabV3Plus(num_classes, pretrained=pretrained)
        self.reconstructor = SeDReNet(num_classes, voxel_size)

    def forward(self, image):
        seg_logits = self.segmenter(image)
        occ, sem_vox = self.reconstructor(image, seg_logits)
        return {"seg_logits": seg_logits, "occ": occ, "sem_vox": sem_vox}
