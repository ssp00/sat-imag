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

class SiameseUNetCD(nn.Module):
    """
    Siamese U-Net Change Detection:
    Contracting encoder extracting multi-scale bi-temporal features,
    fused at each level via concatenation and difference,
    and decoded with expansive U-Net skip connections.
    """
    def __init__(self, in_channels=3):
        super().__init__()
        # Encoder (Contracting Path)
        self.enc1 = DoubleConv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)  # 224 -> 112
        
        self.enc2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)  # 112 -> 56
        
        self.enc3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)  # 56 -> 28
        
        self.enc4 = DoubleConv(128, 256)
        self.pool4 = nn.MaxPool2d(2)  # 28 -> 14
        
        self.bottleneck = DoubleConv(256, 512) # 14x14
        
        # Multi-scale fusion layers
        self.fuse_bottleneck = nn.Sequential(
            nn.Conv2d(512 * 3, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.fuse4 = nn.Sequential(
            nn.Conv2d(256 * 3, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.fuse3 = nn.Sequential(
            nn.Conv2d(128 * 3, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.fuse2 = nn.Sequential(
            nn.Conv2d(64 * 3, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.fuse1 = nn.Sequential(
            nn.Conv2d(32 * 3, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Decoder (Expansive Path with Skip Connections)
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec4 = DoubleConv(256 + 128, 128)
        
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = DoubleConv(128 + 64, 64)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = DoubleConv(64 + 32, 32)
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = DoubleConv(32 + 32, 32)
        
        self.final_conv = nn.Conv2d(32, 1, 1)

    def forward_encoder(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))
        return e1, e2, e3, e4, b

    def forward(self, t1, t2):
        e1_1, e2_1, e3_1, e4_1, b_1 = self.forward_encoder(t1)
        e1_2, e2_2, e3_2, e4_2, b_2 = self.forward_encoder(t2)
        
        fb = self.fuse_bottleneck(torch.cat([b_1, b_2, torch.abs(b_1 - b_2)], dim=1))
        f4 = self.fuse4(torch.cat([e4_1, e4_2, torch.abs(e4_1 - e4_2)], dim=1))
        f3 = self.fuse3(torch.cat([e3_1, e3_2, torch.abs(e3_1 - e3_2)], dim=1))
        f2 = self.fuse2(torch.cat([e2_1, e2_2, torch.abs(e2_1 - e2_2)], dim=1))
        f1 = self.fuse1(torch.cat([e1_1, e1_2, torch.abs(e1_1 - e1_2)], dim=1))
        
        x = self.up4(fb)
        x = self.dec4(torch.cat([x, f4], dim=1))
        
        x = self.up3(x)
        x = self.dec3(torch.cat([x, f3], dim=1))
        
        x = self.up2(x)
        x = self.dec2(torch.cat([x, f2], dim=1))
        
        x = self.up1(x)
        x = self.dec1(torch.cat([x, f1], dim=1))
        
        logits = self.final_conv(x)
        return logits

# ----------------------------------------------------
# ChangeFormer Components
# ----------------------------------------------------
class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding with overlapping convolutions."""
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=64):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=patch_size // 2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class SpatialReductionAttention(nn.Module):
    """Spatial-Reduction Self-Attention (SRA) for efficient Transformer feature extraction."""
    def __init__(self, dim, num_heads=4, sr_ratio=1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class MixFFN(nn.Module):
    """Mix-FFN: Feed-Forward Network with 3x3 depthwise convolution for implicit positional encoding."""
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = self.fc1(x)
        x = x.transpose(1, 2).view(B, -1, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.act(x)
        x = self.fc2(x)
        return x

class ChangeFormerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, sr_ratio=1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SpatialReductionAttention(dim, num_heads, sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = MixFFN(dim, dim * 2, dim)

    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.ffn(self.norm2(x), H, W)
        return x

class ChangeFormerEncoder(nn.Module):
    """Hierarchical 4-stage Transformer Encoder for ChangeFormer."""
    def __init__(self, in_chans=3, embed_dims=[32, 64, 128, 256]):
        super().__init__()
        # Stage 1: H/4 x W/4
        self.patch_embed1 = OverlapPatchEmbed(patch_size=7, stride=4, in_chans=in_chans, embed_dim=embed_dims[0])
        self.block1 = ChangeFormerBlock(dim=embed_dims[0], num_heads=2, sr_ratio=8)
        self.norm1 = nn.LayerNorm(embed_dims[0])

        # Stage 2: H/8 x W/8
        self.patch_embed2 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.block2 = ChangeFormerBlock(dim=embed_dims[1], num_heads=4, sr_ratio=4)
        self.norm2 = nn.LayerNorm(embed_dims[1])

        # Stage 3: H/16 x W/16
        self.patch_embed3 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.block3 = ChangeFormerBlock(dim=embed_dims[2], num_heads=4, sr_ratio=2)
        self.norm3 = nn.LayerNorm(embed_dims[2])

        # Stage 4: H/32 x W/32
        self.patch_embed4 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])
        self.block4 = ChangeFormerBlock(dim=embed_dims[3], num_heads=8, sr_ratio=1)
        self.norm4 = nn.LayerNorm(embed_dims[3])

    def forward(self, x):
        B = x.shape[0]
        # Stage 1
        x1, H1, W1 = self.patch_embed1(x)
        x1 = self.block1(x1, H1, W1)
        x1 = self.norm1(x1).transpose(1, 2).reshape(B, -1, H1, W1)

        # Stage 2
        x2, H2, W2 = self.patch_embed2(x1)
        x2 = self.block2(x2, H2, W2)
        x2 = self.norm2(x2).transpose(1, 2).reshape(B, -1, H2, W2)

        # Stage 3
        x3, H3, W3 = self.patch_embed3(x2)
        x3 = self.block3(x3, H3, W3)
        x3 = self.norm3(x3).transpose(1, 2).reshape(B, -1, H3, W3)

        # Stage 4
        x4, H4, W4 = self.patch_embed4(x3)
        x4 = self.block4(x4, H4, W4)
        x4 = self.norm4(x4).transpose(1, 2).reshape(B, -1, H4, W4)

        return x1, x2, x3, x4

class ChangeFormerCD(nn.Module):
    """
    ChangeFormer: A Transformer-Based Dual-Stream Network for Change Detection.
    Uses hierarchical dual-stream Mix-Transformer encoder, multi-scale difference extraction,
    and unified MLP change decoder.
    """
    def __init__(self, in_chans=3, embed_dims=[32, 64, 128, 256], decoder_dim=128):
        super().__init__()
        self.encoder = ChangeFormerEncoder(in_chans=in_chans, embed_dims=embed_dims)
        
        # Difference projection heads
        self.proj1 = nn.Conv2d(embed_dims[0], decoder_dim, 1)
        self.proj2 = nn.Conv2d(embed_dims[1], decoder_dim, 1)
        self.proj3 = nn.Conv2d(embed_dims[2], decoder_dim, 1)
        self.proj4 = nn.Conv2d(embed_dims[3], decoder_dim, 1)

        # Fusion and prediction head
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(decoder_dim * 4, decoder_dim, 3, padding=1),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_dim, decoder_dim // 2, 3, padding=1),
            nn.BatchNorm2d(decoder_dim // 2),
            nn.ReLU(inplace=True)
        )
        self.final_conv = nn.Conv2d(decoder_dim // 2, 1, 1)

    def forward(self, t1, t2):
        # Dual-stream / Siamese hierarchical transformer encoding
        f1_1, f2_1, f3_1, f4_1 = self.encoder(t1)
        f1_2, f2_2, f3_2, f4_2 = self.encoder(t2)

        # Multi-scale difference features
        diff1 = torch.abs(f1_1 - f1_2) # H/4 x W/4
        diff2 = torch.abs(f2_1 - f2_2) # H/8 x W/8
        diff3 = torch.abs(f3_1 - f3_2) # H/16 x W/16
        diff4 = torch.abs(f4_1 - f4_2) # H/32 x W/32

        # Project all difference features to unified embedding dimension
        p1 = self.proj1(diff1)
        p2 = nn.functional.interpolate(self.proj2(diff2), size=diff1.shape[2:], mode='bilinear', align_corners=True)
        p3 = nn.functional.interpolate(self.proj3(diff3), size=diff1.shape[2:], mode='bilinear', align_corners=True)
        p4 = nn.functional.interpolate(self.proj4(diff4), size=diff1.shape[2:], mode='bilinear', align_corners=True)

        # Concatenate multi-scale representations (H/4 x W/4)
        fused = torch.cat([p1, p2, p3, p4], dim=1)
        x = self.fuse_conv(fused)

        # Upsample back to original resolution H x W (224 x 224)
        x = nn.functional.interpolate(x, size=t1.shape[2:], mode='bilinear', align_corners=True)
        logits = self.final_conv(x)
        return logits

if __name__ == '__main__':
    print("Testing all 4 models: Swin, CNN Baseline, Siamese U-Net, and ChangeFormer...")
    t1 = torch.randn(2, 3, 224, 224)
    t2 = torch.randn(2, 3, 224, 224)
    
    # 1. Swin Siamese Model
    swin_model = SwinSiameseCD(pretrained=False, img_size=(224, 224))
    swin_out = swin_model(t1, t2)
    print(f"1. Swin output shape: {swin_out.shape} | Params: {sum(p.numel() for p in swin_model.parameters()):,}")
    
    # 2. Baseline Siamese Model
    baseline_model = BaselineSiameseCD()
    baseline_out = baseline_model(t1, t2)
    print(f"2. Baseline output shape: {baseline_out.shape} | Params: {sum(p.numel() for p in baseline_model.parameters()):,}")
    
    # 3. Siamese U-Net Model
    unet_model = SiameseUNetCD()
    unet_out = unet_model(t1, t2)
    print(f"3. Siamese U-Net output shape: {unet_out.shape} | Params: {sum(p.numel() for p in unet_model.parameters()):,}")
    
    # 4. ChangeFormer Model
    changeformer_model = ChangeFormerCD()
    cf_out = changeformer_model(t1, t2)
    print(f"4. ChangeFormer output shape: {cf_out.shape} | Params: {sum(p.numel() for p in changeformer_model.parameters()):,}")
    print("All 4 model tests completed successfully!")
