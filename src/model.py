import torch
import torch.nn as nn
import timm

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class SwinSiameseCD(nn.Module):
    def __init__(self, pretrained=True, img_size=(224, 224)):
        super().__init__()
        # Use swin_tiny_patch4_window7_224 and disable fused_attn to allow hook extraction
        self.encoder = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=pretrained,
            features_only=True,
            fused_attn=False
        )
        
        # Swin tiny features_only returns channels: [96, 192, 384, 768]
        # and spatial shapes: [H/4, W/4], [H/8, W/8], [H/16, W/16], [H/32, W/32]
        
        # Fusion projection layers: reduce concatenated + diff features
        # Concat(F_T1, F_T2, |F_T1 - F_T2|) -> channel multiplier is 3
        self.proj4 = nn.Conv2d(768 * 3, 384, 1)
        self.proj3 = nn.Conv2d(384 * 3, 192, 1)
        self.proj2 = nn.Conv2d(192 * 3, 96, 1)
        self.proj1 = nn.Conv2d(96 * 3, 96, 1)

        # Decoder stages
        # Stage 4 (H/32) -> Stage 3 (H/16)
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = DoubleConv(384 + 192, 192) # Concat(up4(proj4), proj3)
        
        # Stage 3 (H/16) -> Stage 2 (H/8)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = DoubleConv(192 + 96, 96) # Concat(up3(dec3), proj2)
        
        # Stage 2 (H/8) -> Stage 1 (H/4)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = DoubleConv(96 + 96, 96) # Concat(up2(dec2), proj1)
        
        # Final prediction layer: Stage 1 (H/4) -> target image size H, W
        self.up_final = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.final_conv = nn.Conv2d(96, 1, 1)

    def forward(self, t1, t2):
        # Forward through Siamese Swin encoder
        feats1 = self.encoder(t1)
        feats2 = self.encoder(t2)
        
        # Swin features are N, H, W, C -> permute to N, C, H, W
        f1, f2, f3, f4 = [x.permute(0, 3, 1, 2) for x in feats1]
        g1, g2, g3, g4 = [x.permute(0, 3, 1, 2) for x in feats2]

        # Feature fusion at each level: concat(t1, t2, |t1 - t2|)
        fuse4 = self.proj4(torch.cat([f4, g4, torch.abs(f4 - g4)], dim=1))
        fuse3 = self.proj3(torch.cat([f3, g3, torch.abs(f3 - g3)], dim=1))
        fuse2 = self.proj2(torch.cat([f2, g2, torch.abs(f2 - g2)], dim=1))
        fuse1 = self.proj1(torch.cat([f1, g1, torch.abs(f1 - g1)], dim=1))

        # Decode
        x = self.up4(fuse4)
        if x.shape[2:] != fuse3.shape[2:]:
            x = nn.functional.interpolate(x, size=fuse3.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, fuse3], dim=1)
        x = self.dec3(x)

        x = self.up3(x)
        if x.shape[2:] != fuse2.shape[2:]:
            x = nn.functional.interpolate(x, size=fuse2.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, fuse2], dim=1)
        x = self.dec2(x)

        x = self.up2(x)
        if x.shape[2:] != fuse1.shape[2:]:
            x = nn.functional.interpolate(x, size=fuse1.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, fuse1], dim=1)
        x = self.dec1(x)

        x = self.up_final(x)
        logits = self.final_conv(x)
        return logits

class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.stage2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.stage3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        self.stage4 = DoubleConv(256, 512)

    def forward(self, x):
        f1 = self.stage1(x)
        f2 = self.stage2(self.pool1(f1))
        f3 = self.stage3(self.pool2(f2))
        f4 = self.stage4(self.pool3(f3))
        return f1, f2, f3, f4

class BaselineSiameseCD(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = CNNEncoder()
        
        # Projections
        self.proj4 = nn.Conv2d(512 * 3, 256, 1)
        self.proj3 = nn.Conv2d(256 * 3, 128, 1)
        self.proj2 = nn.Conv2d(128 * 3, 64, 1)
        self.proj1 = nn.Conv2d(64 * 3, 64, 1)

        # Decoder stages
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = DoubleConv(256 + 128, 128)
        
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = DoubleConv(128 + 64, 64)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = DoubleConv(64 + 64, 64)
        
        self.final_conv = nn.Conv2d(64, 1, 1)

    def forward(self, t1, t2):
        feats1 = self.encoder(t1)
        feats2 = self.encoder(t2)
        
        f1, f2, f3, f4 = feats1
        g1, g2, g3, g4 = feats2

        # Feature fusion
        fuse4 = self.proj4(torch.cat([f4, g4, torch.abs(f4 - g4)], dim=1))
        fuse3 = self.proj3(torch.cat([f3, g3, torch.abs(f3 - g3)], dim=1))
        fuse2 = self.proj2(torch.cat([f2, g2, torch.abs(f2 - g2)], dim=1))
        fuse1 = self.proj1(torch.cat([f1, g1, torch.abs(f1 - g1)], dim=1))

        # Decode
        x = self.up4(fuse4)
        if x.shape[2:] != fuse3.shape[2:]:
            x = nn.functional.interpolate(x, size=fuse3.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, fuse3], dim=1)
        x = self.dec3(x)

        x = self.up3(x)
        if x.shape[2:] != fuse2.shape[2:]:
            x = nn.functional.interpolate(x, size=fuse2.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, fuse2], dim=1)
        x = self.dec2(x)

        x = self.up2(x)
        if x.shape[2:] != fuse1.shape[2:]:
            x = nn.functional.interpolate(x, size=fuse1.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, fuse1], dim=1)
        x = self.dec1(x)

        logits = self.final_conv(x)
        return logits

if __name__ == '__main__':
    print("Testing SwinSiameseCD and BaselineSiameseCD models...")
    t1 = torch.randn(2, 3, 224, 224)
    t2 = torch.randn(2, 3, 224, 224)
    
    # 1. Swin Siamese Model Test (without pre-trained weights for quick dry run)
    swin_model = SwinSiameseCD(pretrained=False, img_size=(224, 224))
    swin_out = swin_model(t1, t2)
    print(f"Swin output shape: {swin_out.shape} (Expected: [2, 1, 224, 224])")
    swin_params = sum(p.numel() for p in swin_model.parameters())
    print(f"Swin model parameter count: {swin_params:,}")
    
    # 2. Baseline Siamese Model Test
    baseline_model = BaselineSiameseCD()
    baseline_out = baseline_model(t1, t2)
    print(f"Baseline output shape: {baseline_out.shape} (Expected: [2, 1, 224, 224])")
    baseline_params = sum(p.numel() for p in baseline_model.parameters())
    print(f"Baseline model parameter count: {baseline_params:,}")
    print("Model tests completed successfully!")
