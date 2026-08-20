import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from src.preprocessing import PairedTransforms

class LevirCDDataset(Dataset):
    def __init__(self, base_dir, split='train', img_size=(128, 128), max_samples=None, transform=None):
        """
        Args:
            base_dir (str): Path to dataset folder (containing train, val, test subdirs)
            split (str): 'train', 'val', or 'test'
            img_size (tuple): Target image resolution (H, W)
            max_samples (int, optional): Limit dataset size to a small subset for fast execution
            transform (callable, optional): Custom transform function
        """
        self.base_dir = base_dir
        self.split = split
        self.img_size = img_size
        self.max_samples = max_samples
        
        self.split_path = os.path.join(base_dir, split)
        if not os.path.exists(self.split_path):
            raise FileNotFoundError(f"Split path {self.split_path} not found.")

        self.dir_a = os.path.join(self.split_path, 'A')
        self.dir_b = os.path.join(self.split_path, 'B')
        self.dir_label = os.path.join(self.split_path, 'label')

        for d in [self.dir_a, self.dir_b, self.dir_label]:
            if not os.path.exists(d):
                raise FileNotFoundError(f"Subdirectory {d} not found.")

        # Find matching filenames
        files_a = set(os.listdir(self.dir_a))
        files_b = set(os.listdir(self.dir_b))
        files_label = set(os.listdir(self.dir_label))

        # Intersection of filenames to ensure all exist
        self.filenames = sorted(list(files_a.intersection(files_b).intersection(files_label)))
        
        if len(self.filenames) == 0:
            raise ValueError(f"No matching file pairs found in {self.split_path}!")

        # Print data loading status
        print(f"Loaded '{split}' split: found {len(self.filenames)} matching image pairs.")

        # Subsample if requested
        if max_samples is not None:
            # Deterministic subsampling (sort and take first N)
            self.filenames = self.filenames[:max_samples]
            print(f"  Subsampled '{split}' to {len(self.filenames)} samples.")

        # Transform setup
        if transform is not None:
            self.transform = transform
        else:
            self.transform = PairedTransforms(size=img_size, train=(split == 'train'))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        
        path_a = os.path.join(self.dir_a, filename)
        path_b = os.path.join(self.dir_b, filename)
        path_l = os.path.join(self.dir_label, filename)

        try:
            img_a = Image.open(path_a).convert('RGB')
            img_b = Image.open(path_b).convert('RGB')
            img_l = Image.open(path_l).convert('L') # Load as grayscale mask
        except Exception as e:
            raise RuntimeError(f"Error loading images for {filename}: {e}")

        # Check raw image sizes
        if img_a.size != img_b.size:
            raise ValueError(f"Size mismatch between T1 and T2 images for {filename}: {img_a.size} vs {img_b.size}")

        # Apply paired transformation
        t1, t2, mask = self.transform(img_a, img_b, img_l)
        
        return {
            't1': t1,
            't2': t2,
            'mask': mask,
            'filename': filename
        }

if __name__ == '__main__':
    # Dry run script to verify dataset implementation
    base_dir = r"d:\vit\SEMESTER-5\IV\DA\LEVIR CD\LEVIR CD"
    print("Testing LevirCDDataset...")
    try:
        train_ds = LevirCDDataset(base_dir, split='train', img_size=(128, 128), max_samples=5)
        print(f"Dataset length: {len(train_ds)}")
        sample = train_ds[0]
        print(f"Sample 't1' shape: {sample['t1'].shape}")
        print(f"Sample 't2' shape: {sample['t2'].shape}")
        print(f"Sample 'mask' shape: {sample['mask'].shape}")
        print(f"Sample filename: {sample['filename']}")
        print("Dataset dry run successful!")
    except Exception as e:
        print(f"Dataset dry run failed: {e}")
