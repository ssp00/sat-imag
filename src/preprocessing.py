import random
import torch
import torchvision.transforms.functional as TF

class PairedTransforms:
    def __init__(self, size=(128, 128), train=True):
        self.size = size
        self.train = train

    def __call__(self, img1, img2, mask):
        # Resize images using BILINEAR (default), mask using NEAREST
        img1 = TF.resize(img1, self.size)
        img2 = TF.resize(img2, self.size)
        mask = TF.resize(mask, self.size, interpolation=TF.InterpolationMode.NEAREST)

        if self.train:
            # Random Horizontal Flip
            if random.random() > 0.5:
                img1 = TF.hflip(img1)
                img2 = TF.hflip(img2)
                mask = TF.hflip(mask)

            # Random Vertical Flip
            if random.random() > 0.5:
                img1 = TF.vflip(img1)
                img2 = TF.vflip(img2)
                mask = TF.vflip(mask)

            # Random Rotation (0, 90, 180, 270 to keep layout grid)
            angles = [0, 90, 180, 270]
            angle = random.choice(angles)
            if angle != 0:
                img1 = TF.rotate(img1, angle)
                img2 = TF.rotate(img2, angle)
                mask = TF.rotate(mask, angle)

        # Convert images to Tensor and Normalize
        img1 = TF.to_tensor(img1)
        img2 = TF.to_tensor(img2)
        
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img1 = TF.normalize(img1, mean=mean, std=std)
        img2 = TF.normalize(img2, mean=mean, std=std)

        # Convert mask to float Tensor (binary: 0 or 1)
        mask_tensor = TF.to_tensor(mask)
        mask_tensor = (mask_tensor > 0.5).float() # Returns (1, H, W)

        return img1, img2, mask_tensor
