import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
from src.dataset import LevirCDDataset
from src.model import SwinSiameseCD, BaselineSiameseCD
from src.losses import BCEDiceLoss
from src.train import compute_metrics
from src.xai import SwinAttentionRollout, GradCAM, compute_xai_overlap, compute_xai_stability
from src.visualization import plot_training_curves, plot_precision_recall_f1, plot_distributions

def denormalize(tensor):
    # Denormalize ImageNet stats
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    denorm = tensor * std + mean
    denorm = torch.clamp(denorm, 0.0, 1.0)
    return denorm.permute(1, 2, 0).numpy()

def generate_diff_map(pred, target):
    # pred: (H, W) binary numpy, target: (H, W) binary numpy
    H, W = pred.shape
    diff_map = np.zeros((H, W, 3), dtype=np.uint8)
    
    tp = (pred == 1) & (target == 1)
    fp = (pred == 1) & (target == 0)
    fn = (pred == 0) & (target == 1)
    tn = (pred == 0) & (target == 0)
    
    diff_map[tp] = [46, 204, 113]   # Green (True Positive)
    diff_map[fp] = [231, 76, 60]    # Red (False Positive)
    diff_map[fn] = [52, 152, 219]   # Blue (False Negative)
    diff_map[tn] = [44, 62, 80]     # Dark Blue-Grey (True Negative / Background)
    
    return diff_map

def run_evaluation():
    dataset_dir = r"d:\vit\SEMESTER-5\IV\DA\LEVIR CD\LEVIR CD"
    project_dir = r"d:\vit\SEMESTER-5\IV\DA"
    checkpoint_dir = os.path.join(project_dir, 'checkpoints')
    outputs_dir = os.path.join(project_dir, 'outputs')
    
    os.makedirs(os.path.join(outputs_dir, 'predictions'), exist_ok=True)
    os.makedirs(os.path.join(outputs_dir, 'xai'), exist_ok=True)
    os.makedirs(os.path.join(outputs_dir, 'metrics'), exist_ok=True)
    
    # Load test set (subsampled for speed)
    test_ds = LevirCDDataset(
        base_dir=dataset_dir,
        split='test',
        img_size=(224, 224),
        max_samples=5
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on device: {device}")
    
    # Initialize models and load best weights
    swin_model = SwinSiameseCD(pretrained=False, img_size=(224, 224))
    swin_model.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'best_model_swin.pth'), map_location=device))
    swin_model.to(device)
    swin_model.eval()
    
    baseline_model = BaselineSiameseCD()
    baseline_model.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'best_model_baseline.pth'), map_location=device))
    baseline_model.to(device)
    baseline_model.eval()

    # Setup XAI
    swin_rollout = SwinAttentionRollout(swin_model)
    gradcam_target_swin = swin_model.dec1.conv[3] # Final conv in Swin Siamese decoder
    gradcam_target_baseline = baseline_model.dec1.conv[3] # Final conv in CNN Baseline decoder
    
    gcam_swin = GradCAM(swin_model, gradcam_target_swin)
    gcam_baseline = GradCAM(baseline_model, gradcam_target_baseline)

    swin_preds = []
    baseline_preds = []
    targets = []
    filenames = []
    
    swin_xai_overlaps = []
    swin_xai_stabilities = []
    baseline_xai_overlaps = []
    baseline_xai_stabilities = []

    # Iterate over test set
    for i, batch in enumerate(test_loader):
        t1 = batch['t1'].to(device)
        t2 = batch['t2'].to(device)
        mask = batch['mask'].to(device)
        filename = batch['filename'][0]
        
        filenames.append(filename)
        targets.append(mask.cpu())
        
        # 1. Swin predictions and XAI
        with torch.enable_grad():
            # Get Attention Rollout map
            # Re-register Swin Rollout hooks just to be clean
            swin_rollout.attention_maps.clear()
            rollout_map = swin_rollout.get_rollout(t1, t2) # returns (1, 1, 224, 224)
            
            # Get GradCAM
            gcam_map_swin = gcam_swin.generate_heatmap(t1, t2)
            
            # Compute stability on GradCAM
            stability_swin = compute_xai_stability(swin_model, gcam_swin, t1, t2, is_swin=False)
            
        with torch.no_grad():
            logits_swin = swin_model(t1, t2)
            pred_swin = (torch.sigmoid(logits_swin) > 0.5).float().cpu()
            swin_preds.append(pred_swin)

        # 2. Baseline predictions and XAI
        with torch.enable_grad():
            gcam_map_baseline = gcam_baseline.generate_heatmap(t1, t2)
            stability_baseline = compute_xai_stability(baseline_model, gcam_baseline, t1, t2, is_swin=False)
            
        with torch.no_grad():
            logits_baseline = baseline_model(t1, t2)
            pred_baseline = (torch.sigmoid(logits_baseline) > 0.5).float().cpu()
            baseline_preds.append(pred_baseline)

        # Calculate XAI Overlaps
        overlap_swin = compute_xai_overlap(gcam_map_swin[0, 0], mask[0, 0].cpu())
        overlap_baseline = compute_xai_overlap(gcam_map_baseline[0, 0], mask[0, 0].cpu())
        
        swin_xai_overlaps.append(overlap_swin)
        swin_xai_stabilities.append(stability_swin)
        baseline_xai_overlaps.append(overlap_baseline)
        baseline_xai_stabilities.append(stability_baseline)

        # Generate prediction and XAI visualizations for InfoVis
        t1_np = denormalize(batch['t1'][0])
        t2_np = denormalize(batch['t2'][0])
        gt_np = mask[0, 0].cpu().numpy()
        pred_swin_np = pred_swin[0, 0].numpy()
        pred_base_np = pred_baseline[0, 0].numpy()

        diff_swin = generate_diff_map(pred_swin_np, gt_np)
        diff_base = generate_diff_map(pred_base_np, gt_np)

        # Calculate mask change percentages
        gt_change_pct = (gt_np.sum() / gt_np.size) * 100.0
        swin_change_pct = (pred_swin_np.sum() / pred_swin_np.size) * 100.0
        base_change_pct = (pred_base_np.sum() / pred_base_np.size) * 100.0

        # Save Aligned Predictions Grid (T1 | T2 | GT | Prediction | Diff Map)
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        
        # Row 1: Swin Transformer
        axes[0, 0].imshow(t1_np)
        axes[0, 0].set_title("T1 / Earlier")
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(t2_np)
        axes[0, 1].set_title("T2 / Later")
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(gt_np, cmap='gray')
        axes[0, 2].set_title(f"GT Mask ({gt_change_pct:.2f}% change)")
        axes[0, 2].axis('off')
        
        axes[0, 3].imshow(pred_swin_np, cmap='gray')
        axes[0, 3].set_title(f"Swin Pred ({swin_change_pct:.2f}% change)")
        axes[0, 3].axis('off')
        
        axes[0, 4].imshow(diff_swin)
        axes[0, 4].set_title("Swin Diff Map\n(Green=TP, Red=FP, Blue=FN)")
        axes[0, 4].axis('off')
        
        # Row 2: CNN Baseline
        axes[1, 0].imshow(t1_np)
        axes[1, 0].set_title("T1 / Earlier")
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(t2_np)
        axes[1, 1].set_title("T2 / Later")
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(gt_np, cmap='gray')
        axes[1, 2].set_title(f"GT Mask ({gt_change_pct:.2f}% change)")
        axes[1, 2].axis('off')
        
        axes[1, 3].imshow(pred_base_np, cmap='gray')
        axes[1, 3].set_title(f"Baseline Pred ({base_change_pct:.2f}% change)")
        axes[1, 3].axis('off')
        
        axes[1, 4].imshow(diff_base)
        axes[1, 4].set_title("Baseline Diff Map\n(Green=TP, Red=FP, Blue=FN)")
        axes[1, 4].axis('off')
        
        plt.suptitle(f"Change Detection Aligned Visual Comparison: {filename}", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(outputs_dir, 'predictions', f"pred_compare_{filename}"), dpi=150)
        plt.close()

        # Save Aligned XAI Grid (T1 | T2 | Prediction | Swin Attention Rollout | Swin Grad-CAM | Baseline Grad-CAM)
        fig, axes = plt.subplots(1, 6, figsize=(22, 4.5))
        axes[0].imshow(t1_np)
        axes[0].set_title("T1 / Earlier")
        axes[0].axis('off')
        
        axes[1].imshow(t2_np)
        axes[1].set_title("T2 / Later")
        axes[1].axis('off')
        
        axes[2].imshow(gt_np, cmap='gray')
        axes[2].set_title("Ground Truth Mask")
        axes[2].axis('off')
        
        # Swin Attention Rollout heatmap
        rollout_np = rollout_map[0, 0].numpy()
        # Normalize
        rollout_np = (rollout_np - rollout_np.min()) / (rollout_np.max() - rollout_np.min() + 1e-8)
        axes[3].imshow(t2_np)
        axes[3].imshow(rollout_np, cmap='jet', alpha=0.5)
        axes[3].set_title("Swin Attn Rollout")
        axes[3].axis('off')
        
        # Swin Grad-CAM heatmap
        gcam_swin_np = gcam_map_swin[0, 0].numpy()
        axes[4].imshow(t2_np)
        axes[4].imshow(gcam_swin_np, cmap='jet', alpha=0.5)
        axes[4].set_title(f"Swin Grad-CAM\n(Overlap IoU: {overlap_swin:.2f})")
        axes[4].axis('off')
        
        # Baseline Grad-CAM heatmap
        gcam_base_np = gcam_map_baseline[0, 0].numpy()
        axes[5].imshow(t2_np)
        axes[5].imshow(gcam_base_np, cmap='jet', alpha=0.5)
        axes[5].set_title(f"Baseline Grad-CAM\n(Overlap IoU: {overlap_baseline:.2f})")
        axes[5].axis('off')

        plt.suptitle(f"Explainable AI Diagnostic Visualizations: {filename}", fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(outputs_dir, 'xai', f"xai_{filename}"), dpi=150)
        plt.close()

    # Clean up XAI hooks
    swin_rollout.remove_hooks()
    gcam_swin.remove_hooks()
    gcam_baseline.remove_hooks()

    # Concatenate predictions for global evaluation metrics
    all_swin_preds = torch.cat(swin_preds, dim=0)
    all_base_preds = torch.cat(baseline_preds, dim=0)
    all_targets = torch.cat(targets, dim=0)

    metrics_swin = compute_metrics(all_swin_preds, all_targets)
    metrics_base = compute_metrics(all_base_preds, all_targets)

    # Print Report
    print("\n=========================================")
    print("FINAL EVALUATION REPORT ON TEST SET")
    print("=========================================")
    print(f"Model: Siamese Swin Transformer")
    print(f"Test samples: {len(test_loader)}")
    print(f"Accuracy:  {metrics_swin['accuracy']:.4f}")
    print(f"Precision: {metrics_swin['precision']:.4f}")
    print(f"Recall:    {metrics_swin['recall']:.4f}")
    print(f"F1-score:  {metrics_swin['f1']:.4f}")
    print(f"IoU:       {metrics_swin['iou']:.4f}")
    print(f"Dice:      {metrics_swin['dice']:.4f}")
    
    print("\n-----------------------------------------")
    print(f"Model: Siamese CNN Baseline")
    print(f"Test samples: {len(test_loader)}")
    print(f"Accuracy:  {metrics_base['accuracy']:.4f}")
    print(f"Precision: {metrics_base['precision']:.4f}")
    print(f"Recall:    {metrics_base['recall']:.4f}")
    print(f"F1-score:  {metrics_base['f1']:.4f}")
    print(f"IoU:       {metrics_base['iou']:.4f}")
    print(f"Dice:      {metrics_base['dice']:.4f}")
    print("=========================================")

    # Save test metrics JSON and CSV
    test_metrics = {
        'Swin': {k.upper(): v for k, v in metrics_swin.items()},
        'Baseline': {k.upper(): v for k, v in metrics_base.items()}
    }
    with open(os.path.join(outputs_dir, 'metrics', 'test_metrics.json'), 'w') as f:
        json.dump(test_metrics, f, indent=4)
        
    pd.DataFrame(test_metrics).T.to_csv(os.path.join(outputs_dir, 'metrics', 'test_metrics.csv'))
    print("Saved test_metrics.json and test_metrics.csv.")

    # Create CSV for Tableau / Power BI dashboard creation
    dashboard_records = []
    for idx, fname in enumerate(filenames):
        gt_np = targets[idx][0, 0].numpy()
        pred_swin_np = swin_preds[idx][0, 0].numpy()
        pred_base_np = baseline_preds[idx][0, 0].numpy()
        
        gt_change_pct = (gt_np.sum() / gt_np.size) * 100.0
        swin_change_pct = (pred_swin_np.sum() / pred_swin_np.size) * 100.0
        base_change_pct = (pred_base_np.sum() / pred_base_np.size) * 100.0
        
        # Calculate sample-level metrics for Swin
        sample_metrics_swin = compute_metrics(swin_preds[idx], targets[idx])
        sample_metrics_base = compute_metrics(baseline_preds[idx], targets[idx])
        
        # We will log Swin metrics primarily, or save separate lines per model to filter in Tableau!
        # Let's save a clean flat structure that dashboards love:
        dashboard_records.append({
            'sample_id': fname,
            'model_type': 'Swin-Transformer',
            'change_percentage_ground_truth': gt_change_pct,
            'change_percentage_prediction': swin_change_pct,
            'precision': sample_metrics_swin['precision'],
            'recall': sample_metrics_swin['recall'],
            'f1': sample_metrics_swin['f1'],
            'iou': sample_metrics_swin['iou'],
            'dice': sample_metrics_swin['dice'],
            'xai_overlap': swin_xai_overlaps[idx],
            'xai_stability': swin_xai_stabilities[idx]
        })
        
        dashboard_records.append({
            'sample_id': fname,
            'model_type': 'CNN-Baseline',
            'change_percentage_ground_truth': gt_change_pct,
            'change_percentage_prediction': base_change_pct,
            'precision': sample_metrics_base['precision'],
            'recall': sample_metrics_base['recall'],
            'f1': sample_metrics_base['f1'],
            'iou': sample_metrics_base['iou'],
            'dice': sample_metrics_base['dice'],
            'xai_overlap': baseline_xai_overlaps[idx],
            'xai_stability': baseline_xai_stabilities[idx]
        })

    df_dashboard = pd.DataFrame(dashboard_records)
    dashboard_csv_path = os.path.join(outputs_dir, 'metrics', 'dashboard_data.csv')
    df_dashboard.to_csv(dashboard_csv_path, index=False)
    print(f"Saved dashboard data to {dashboard_csv_path}")

    # Generate all academic / diagnostic plots
    # 1-4. Loss and performance validation curves
    with open(os.path.join(outputs_dir, 'metrics', 'history_baseline.json'), 'r') as f:
        hist_base = json.load(f)
    with open(os.path.join(outputs_dir, 'metrics', 'history_swin.json'), 'r') as f:
        hist_swin = json.load(f)
        
    plot_training_curves(hist_base, hist_swin, os.path.join(outputs_dir, 'visualizations'))
    
    # 5. Bar comparison plot
    plot_precision_recall_f1(metrics_base, metrics_swin, os.path.join(outputs_dir, 'visualizations'))
    
    # 6-8. Statistical distributions
    plot_distributions(df_dashboard[df_dashboard['model_type'] == 'Swin-Transformer'], os.path.join(outputs_dir, 'visualizations'))

if __name__ == '__main__':
    run_evaluation()
