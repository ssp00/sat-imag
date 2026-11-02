import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
from src.dataset import LevirCDDataset
from src.model import SwinSiameseCD, BaselineSiameseCD, SiameseUNetCD, ChangeFormerCD
from src.train import compute_metrics
from src.xai import SwinAttentionRollout, GradCAM, compute_xai_overlap, compute_xai_stability
from src.visualization import plot_training_curves, plot_precision_recall_f1, plot_distributions

def denormalize(tensor):
    """Denormalize ImageNet normalized tensor back to [0, 1] RGB image."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    denorm = tensor * std + mean
    denorm = torch.clamp(denorm, 0.0, 1.0)
    return denorm.permute(1, 2, 0).numpy()

def generate_diff_map(pred, target):
    """
    Color-coded segmentation error map:
    Green: True Positive (TP)
    Red: False Positive (FP)
    Blue: False Negative (FN)
    Dark Blue-Grey: True Negative (TN)
    """
    H, W = pred.shape
    diff_map = np.zeros((H, W, 3), dtype=np.uint8)

    tp = (pred == 1) & (target == 1)
    fp = (pred == 1) & (target == 0)
    fn = (pred == 0) & (target == 1)
    tn = (pred == 0) & (target == 0)

    diff_map[tp] = [46, 204, 113]   # Green (TP)
    diff_map[fp] = [231, 76, 60]    # Red (FP)
    diff_map[fn] = [52, 152, 219]   # Blue (FN)
    diff_map[tn] = [44, 62, 80]     # Dark Blue-Grey (TN)

    return diff_map

def compute_detailed_pixel_stats(pred_tensor, target_tensor, eps=1e-7):
    """Computes TP, TN, FP, FN, and detailed segmentation metrics for a single sample or batch."""
    p = pred_tensor.view(-1)
    t = target_tensor.view(-1)

    tp = int((p * t).sum().item())
    fp = int((p * (1.0 - t)).sum().item())
    fn = int(((1.0 - p) * t).sum().item())
    tn = int(((1.0 - p) * (1.0 - t)).sum().item())

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = (2.0 * precision * recall) / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)

    return {
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
        'dice': dice
    }

def run_evaluation():
    dataset_dir = r"d:\vit\SEMESTER-5\IV\DA\LEVIR CD\LEVIR CD"
    project_dir = r"d:\vit\SEMESTER-5\IV\DA"
    checkpoint_dir = os.path.join(project_dir, 'checkpoints')
    outputs_dir = os.path.join(project_dir, 'outputs')

    # Ensure required directory structure exists
    model_keys = ['siamese_cnn', 'siamese_unet', 'swin', 'changeformer']
    model_labels = {
        'siamese_cnn': 'Siamese CNN',
        'siamese_unet': 'Siamese U-Net',
        'swin': 'Swin Transformer',
        'changeformer': 'ChangeFormer'
    }

    for k in model_keys:
        os.makedirs(os.path.join(outputs_dir, 'predictions', k), exist_ok=True)
    os.makedirs(os.path.join(outputs_dir, 'xai'), exist_ok=True)
    os.makedirs(os.path.join(outputs_dir, 'metrics'), exist_ok=True)
    os.makedirs(os.path.join(outputs_dir, 'visualizations'), exist_ok=True)

    # Load test split (5 aligned pairs)
    test_ds = LevirCDDataset(
        base_dir=dataset_dir,
        split='test',
        img_size=(224, 224),
        max_samples=5
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating 4 models on device: {device}...")

    # Load 4 models
    models = {}

    # 1. Siamese CNN
    m_cnn = BaselineSiameseCD()
    m_cnn.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'best_model_baseline.pth'), map_location=device))
    m_cnn.to(device).eval()
    models['siamese_cnn'] = m_cnn

    # 2. Siamese U-Net
    m_unet = SiameseUNetCD()
    m_unet.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'best_model_siamese_unet.pth'), map_location=device))
    m_unet.to(device).eval()
    models['siamese_unet'] = m_unet

    # 3. Swin Transformer
    m_swin = SwinSiameseCD(pretrained=False, img_size=(224, 224))
    m_swin.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'best_model_swin.pth'), map_location=device))
    m_swin.to(device).eval()
    models['swin'] = m_swin

    # 4. ChangeFormer
    m_cf = ChangeFormerCD()
    m_cf.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'best_model_changeformer.pth'), map_location=device))
    m_cf.to(device).eval()
    models['changeformer'] = m_cf

    # Setup XAI generators
    swin_rollout = SwinAttentionRollout(m_swin)
    gcam_swin = GradCAM(m_swin, m_swin.dec1.conv[3])
    gcam_cnn = GradCAM(m_cnn, m_cnn.dec1.conv[3])
    gcam_unet = GradCAM(m_unet, m_unet.dec1.conv[3])
    gcam_cf = GradCAM(m_cf, m_cf.fuse_conv[0])

    gcam_dict = {
        'siamese_cnn': gcam_cnn,
        'siamese_unet': gcam_unet,
        'swin': gcam_swin,
        'changeformer': gcam_cf
    }

    # Tracking containers
    all_predictions = {k: [] for k in model_keys}
    all_targets = []
    all_filenames = []
    all_xai_heatmaps = {k: [] for k in model_keys}
    all_xai_overlaps = {k: [] for k in model_keys}
    all_xai_stabilities = {k: [] for k in model_keys}

    # Test loop
    for batch_idx, batch in enumerate(test_loader):
        t1 = batch['t1'].to(device)
        t2 = batch['t2'].to(device)
        mask = batch['mask'].to(device)
        filename = batch['filename'][0]

        all_filenames.append(filename)
        all_targets.append(mask.cpu())

        t1_np = denormalize(batch['t1'][0])
        t2_np = denormalize(batch['t2'][0])
        gt_np = mask[0, 0].cpu().numpy().astype(np.uint8)

        sample_preds_np = {}

        # Evaluate each model on this sample
        for m_key in model_keys:
            model = models[m_key]

            # Forward pass
            with torch.no_grad():
                logits = model(t1, t2)
                pred = (torch.sigmoid(logits) > 0.5).float().cpu()
                all_predictions[m_key].append(pred)
                pred_np = pred[0, 0].numpy().astype(np.uint8)
                sample_preds_np[m_key] = pred_np

            # Save individual binary mask PNG
            pred_mask_img = Image.fromarray(pred_np * 255)
            mask_out_path = os.path.join(outputs_dir, 'predictions', m_key, filename)
            pred_mask_img.save(mask_out_path)

            # XAI Heatmap & Metrics
            with torch.enable_grad():
                gcam_gen = gcam_dict[m_key]
                heatmap = gcam_gen.generate_heatmap(t1, t2)
                stability = compute_xai_stability(model, gcam_gen, t1, t2, is_swin=False)

            overlap = compute_xai_overlap(heatmap[0, 0], mask[0, 0].cpu())
            all_xai_heatmaps[m_key].append(heatmap[0, 0].numpy())
            all_xai_overlaps[m_key].append(overlap)
            all_xai_stabilities[m_key].append(stability)

        # Swin Attention Rollout (dedicated XAI for Swin Transformer)
        with torch.enable_grad():
            swin_rollout.attention_maps.clear()
            rollout_map = swin_rollout.get_rollout(t1, t2)[0, 0].numpy()
            rollout_norm = (rollout_map - rollout_map.min()) / (rollout_map.max() - rollout_map.min() + 1e-8)

        # Generate 4-Model Side-by-Side Comparison Plot for this sample
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))

        # Row 1: Inputs, GT, and Color Legend
        axes[0, 0].imshow(t1_np)
        axes[0, 0].set_title(f"T1 / Earlier Image\n({filename})", fontsize=11, fontweight='bold')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(t2_np)
        axes[0, 1].set_title(f"T2 / Later Image\n({filename})", fontsize=11, fontweight='bold')
        axes[0, 1].axis('off')

        gt_pct = (gt_np.sum() / gt_np.size) * 100.0
        axes[0, 2].imshow(gt_np, cmap='gray')
        axes[0, 2].set_title(f"Ground Truth Mask\n({gt_pct:.2f}% changed)", fontsize=11, fontweight='bold')
        axes[0, 2].axis('off')

        # Swin Attention Rollout
        axes[0, 3].imshow(t2_np)
        axes[0, 3].imshow(rollout_norm, cmap='jet', alpha=0.55)
        axes[0, 3].set_title("Swin Attention Rollout (XAI)", fontsize=11, fontweight='bold')
        axes[0, 3].axis('off')

        # Row 2: Predictions for all 4 models
        for idx, m_key in enumerate(model_keys):
            p_np = sample_preds_np[m_key]
            diff = generate_diff_map(p_np, gt_np)
            p_pct = (p_np.sum() / p_np.size) * 100.0
            p_f1 = compute_metrics(torch.tensor(p_np), torch.tensor(gt_np))['f1']

            axes[1, idx].imshow(diff)
            axes[1, idx].set_title(f"{model_labels[m_key]}\nChange: {p_pct:.2f}% | F1: {p_f1:.3f}\n(Green=TP, Red=FP, Blue=FN)",
                                   fontsize=10, fontweight='bold')
            axes[1, idx].axis('off')

        plt.suptitle(f"4-Model Aligned Benchmark Comparison: {filename}", fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(outputs_dir, 'predictions', f"multi_compare_{filename}"), dpi=150)
        plt.savefig(os.path.join(outputs_dir, 'predictions', f"pred_compare_{filename}"), dpi=150) # backwards compatibility
        plt.close()

        # Save Aligned XAI multi-grid
        fig_xai, axes_xai = plt.subplots(1, 6, figsize=(22, 4.5))
        axes_xai[0].imshow(t1_np); axes_xai[0].set_title("T1 Image"); axes_xai[0].axis('off')
        axes_xai[1].imshow(t2_np); axes_xai[1].set_title("T2 Image"); axes_xai[1].axis('off')
        axes_xai[2].imshow(gt_np, cmap='gray'); axes_xai[2].set_title("Ground Truth"); axes_xai[2].axis('off')

        # Swin Rollout
        axes_xai[3].imshow(t2_np)
        axes_xai[3].imshow(rollout_norm, cmap='jet', alpha=0.55)
        axes_xai[3].set_title("Swin Attn Rollout"); axes_xai[3].axis('off')

        # Swin Grad-CAM
        axes_xai[4].imshow(t2_np)
        axes_xai[4].imshow(all_xai_heatmaps['swin'][-1], cmap='jet', alpha=0.55)
        axes_xai[4].set_title(f"Swin Grad-CAM (IoU: {all_xai_overlaps['swin'][-1]:.2f})"); axes_xai[4].axis('off')

        # ChangeFormer Grad-CAM
        axes_xai[5].imshow(t2_np)
        axes_xai[5].imshow(all_xai_heatmaps['changeformer'][-1], cmap='jet', alpha=0.55)
        axes_xai[5].set_title(f"ChangeFormer CAM (IoU: {all_xai_overlaps['changeformer'][-1]:.2f})"); axes_xai[5].axis('off')

        plt.suptitle(f"Explainable AI Heatmap Diagnostics: {filename}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(outputs_dir, 'xai', f"xai_{filename}"), dpi=150)
        plt.close()

    # Clean up hooks
    swin_rollout.remove_hooks()
    for gcam in gcam_dict.values():
        gcam.remove_hooks()

    # Global multi-model test metrics
    global_targets = torch.cat(all_targets, dim=0)
    final_test_report = {}
    individual_model_dfs = {}

    print("\n=======================================================")
    print("FINAL 4-MODEL EVALUATION BENCHMARK ON TEST SET")
    print("=======================================================")
    print(f"{'Model':<20} | {'Accuracy':<9} | {'Precision':<9} | {'Recall':<9} | {'F1':<9} | {'IoU':<9} | {'Dice':<9}")
    print("-" * 80)

    for m_key in model_keys:
        all_p = torch.cat(all_predictions[m_key], dim=0)
        m_stats = compute_detailed_pixel_stats(all_p, global_targets)
        m_label = model_labels[m_key]

        final_test_report[m_label] = {
            'Accuracy': m_stats['accuracy'],
            'Precision': m_stats['precision'],
            'Recall': m_stats['recall'],
            'F1': m_stats['f1'],
            'IoU': m_stats['iou'],
            'Dice': m_stats['dice']
        }

        print(f"{m_label:<20} | {m_stats['accuracy']:<9.4f} | {m_stats['precision']:<9.4f} | {m_stats['recall']:<9.4f} | {m_stats['f1']:<9.4f} | {m_stats['iou']:<9.4f} | {m_stats['dice']:<9.4f}")

        # Save individual model metric CSV
        ind_df = pd.DataFrame([m_stats])
        ind_df.to_csv(os.path.join(outputs_dir, 'metrics', f"{m_key}_metrics.csv"), index=False)

    print("=======================================================")

    # Save summary test_metrics.csv and test_metrics.json
    df_summary = pd.DataFrame(final_test_report).T
    df_summary.to_csv(os.path.join(outputs_dir, 'metrics', 'test_metrics.csv'))
    with open(os.path.join(outputs_dir, 'metrics', 'test_metrics.json'), 'w') as f:
        json.dump(final_test_report, f, indent=4)
    print("Saved test_metrics.csv and test_metrics.json.")

    # Build comprehensive dashboard_data.csv with all sample-level fields required by Part 4
    dashboard_rows = []
    for idx, fname in enumerate(all_filenames):
        gt = all_targets[idx]
        gt_np = gt[0, 0].numpy()
        true_change_px = int(gt_np.sum())
        total_px = gt_np.size
        true_change_pct = (true_change_px / total_px) * 100.0

        for m_key in model_keys:
            pred = all_predictions[m_key][idx]
            pred_np = pred[0, 0].numpy()
            pred_change_px = int(pred_np.sum())
            pred_change_pct = (pred_change_px / total_px) * 100.0

            stats = compute_detailed_pixel_stats(pred, gt)
            overlap = all_xai_overlaps[m_key][idx]
            stability = all_xai_stabilities[m_key][idx]

            dashboard_rows.append({
                'sample_id': fname,
                'model': model_labels[m_key],
                'model_type': model_labels[m_key], # backwards compatibility
                'image_name': fname,
                'true_change_pixels': true_change_px,
                'predicted_change_pixels': pred_change_px,
                'true_change_percentage': true_change_pct,
                'predicted_change_percentage': pred_change_pct,
                'change_percentage_ground_truth': true_change_pct, # backwards compatibility
                'change_percentage_prediction': pred_change_pct,   # backwards compatibility
                'TP': stats['TP'],
                'TN': stats['TN'],
                'FP': stats['FP'],
                'FN': stats['FN'],
                'accuracy': stats['accuracy'],
                'precision': stats['precision'],
                'recall': stats['recall'],
                'f1': stats['f1'],
                'iou': stats['iou'],
                'dice': stats['dice'],
                'xai_overlap': overlap,
                'xai_stability': stability
            })

    df_dashboard = pd.DataFrame(dashboard_rows)
    master_dashboard_path1 = os.path.join(outputs_dir, 'dashboard_data.csv')
    master_dashboard_path2 = os.path.join(outputs_dir, 'metrics', 'dashboard_data.csv')
    df_dashboard.to_csv(master_dashboard_path1, index=False)
    df_dashboard.to_csv(master_dashboard_path2, index=False)
    print(f"Saved master dashboard data to {master_dashboard_path1} and {master_dashboard_path2}.")

    # Generate InfoVis academic plots
    histories = {}
    hist_files = {
        'Siamese CNN': 'history_baseline.json',
        'Siamese U-Net': 'history_siamese_unet.json',
        'Swin Transformer': 'history_swin.json',
        'ChangeFormer': 'history_changeformer.json'
    }
    for m_label, h_file in hist_files.items():
        h_path = os.path.join(outputs_dir, 'metrics', h_file)
        if os.path.exists(h_path):
            with open(h_path, 'r') as f:
                histories[m_label] = json.load(f)

    if len(histories) >= 2:
        plot_training_curves(histories, os.path.join(outputs_dir, 'visualizations'))

    metrics_for_plot = {
        m_label: {
            'precision': final_test_report[m_label]['Precision'],
            'recall': final_test_report[m_label]['Recall'],
            'f1': final_test_report[m_label]['F1']
        }
        for m_label in final_test_report
    }
    plot_precision_recall_f1(metrics_for_plot, os.path.join(outputs_dir, 'visualizations'))
    plot_distributions(df_dashboard, os.path.join(outputs_dir, 'visualizations'))
    print("Multi-model evaluation and visual asset generation completed successfully!")

if __name__ == '__main__':
    run_evaluation()
