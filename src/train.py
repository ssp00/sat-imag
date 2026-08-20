import os
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import LevirCDDataset
from src.model import SwinSiameseCD, BaselineSiameseCD
from src.losses import BCEDiceLoss
from tqdm import tqdm

def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_metrics(preds, targets, eps=1e-7):
    # Flatten tensors for pixel-level calculations
    preds = preds.view(-1)
    targets = targets.view(-1)
    
    tp = (preds * targets).sum().item()
    fp = (preds * (1.0 - targets)).sum().item()
    fn = ((1.0 - preds) * targets).sum().item()
    tn = ((1.0 - preds) * (1.0 - targets)).sum().item()

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = (2.0 * precision * recall) / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
        'dice': dice
    }

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch in tqdm(loader, desc="  Training", leave=False):
        t1 = batch['t1'].to(device)
        t2 = batch['t2'].to(device)
        mask = batch['mask'].to(device)

        optimizer.zero_grad()
        logits = model(t1, t2)
        loss = criterion(logits, mask)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * t1.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_masks = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="  Validation", leave=False):
            t1 = batch['t1'].to(device)
            t2 = batch['t2'].to(device)
            mask = batch['mask'].to(device)

            logits = model(t1, t2)
            loss = criterion(logits, mask)
            running_loss += loss.item() * t1.size(0)

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            all_preds.append(preds.cpu())
            all_masks.append(mask.cpu())

    epoch_loss = running_loss / len(loader.dataset)
    
    all_preds = torch.cat(all_preds, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    
    metrics = compute_metrics(all_preds, all_masks)
    metrics['loss'] = epoch_loss
    return metrics

def run_training(config):
    set_seed(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() and config['use_cuda'] else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(config['checkpoint_dir'], exist_ok=True)

    # Initialize datasets
    print("Loading datasets...")
    train_ds = LevirCDDataset(
        base_dir=config['dataset_dir'],
        split='train',
        img_size=(config['img_size'], config['img_size']),
        max_samples=config['max_train_samples']
    )
    val_ds = LevirCDDataset(
        base_dir=config['dataset_dir'],
        split='val',
        img_size=(config['img_size'], config['img_size']),
        max_samples=config['max_val_samples']
    )

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

    # Initialize model
    model_type = config['model_type'].lower()
    print(f"Initializing model '{model_type}'...")
    if model_type == 'swin':
        model = SwinSiameseCD(pretrained=config['pretrained'], img_size=(config['img_size'], config['img_size']))
    elif model_type == 'baseline':
        model = BaselineSiameseCD()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    model.to(device)

    # Optimizer & Scheduler & Loss
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    criterion = BCEDiceLoss(bce_weight=1.0, dice_weight=1.0)

    best_val_f1 = -1.0
    history = {
        'train_loss': [], 'val_loss': [],
        'val_accuracy': [], 'val_precision': [], 'val_recall': [],
        'val_f1': [], 'val_iou': [], 'val_dice': []
    }

    for epoch in range(1, config['epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['epochs']}:")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Val Precision: {val_metrics['precision']:.4f} | Val Recall: {val_metrics['recall']:.4f} | Val F1: {val_metrics['f1']:.4f} | Val IoU: {val_metrics['iou']:.4f}")

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_iou'].append(val_metrics['iou'])
        history['val_dice'].append(val_metrics['dice'])

        # Save checkpoint
        if val_metrics['f1'] >= best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_path = os.path.join(config['checkpoint_dir'], f"best_model_{model_type}.pth")
            torch.save(model.state_dict(), best_path)
            print(f"  --> Saved new BEST model to {best_path} (F1: {best_val_f1:.4f})")

    # Save final model
    last_path = os.path.join(config['checkpoint_dir'], f"last_model_{model_type}.pth")
    torch.save(model.state_dict(), last_path)
    print(f"  --> Saved final model to {last_path}")

    # Save metrics history
    metrics_out_dir = os.path.join(config['project_dir'], 'outputs', 'metrics')
    os.makedirs(metrics_out_dir, exist_ok=True)
    history_path = os.path.join(metrics_out_dir, f"history_{model_type}.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
        
    print(f"History saved to {history_path}")
    return history

if __name__ == '__main__':
    # Local default configurations for verification
    default_config = {
        'seed': 42,
        'use_cuda': True,
        'dataset_dir': r"d:\vit\SEMESTER-5\IV\DA\LEVIR CD\LEVIR CD",
        'project_dir': r"d:\vit\SEMESTER-5\IV\DA",
        'checkpoint_dir': r"d:\vit\SEMESTER-5\IV\DA\checkpoints",
        'model_type': 'baseline', # 'swin' or 'baseline'
        'pretrained': True,
        'img_size': 224,
        'batch_size': 4,
        'lr': 1e-4,
        'epochs': 3,
        'num_workers': 0,
        'max_train_samples': 4,
        'max_val_samples': 2
    }
    print("Starting dry-run training for Baseline...")
    run_training(default_config)
    print("\nDry-run training for Baseline completed successfully!")
