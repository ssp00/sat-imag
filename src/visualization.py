import os
import json
import matplotlib.pyplot as plt
import numpy as np

def plot_training_curves(history_baseline, history_swin, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(history_baseline['train_loss']) + 1)
    
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # Loss Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs, history_baseline['train_loss'], 'o-', label='Baseline Train Loss', color='#1f77b4', linewidth=2)
    axes[0].plot(epochs, history_baseline['val_loss'], 's--', label='Baseline Val Loss', color='#aec7e8', linewidth=2)
    axes[0].plot(epochs, history_swin['train_loss'], 'o-', label='Swin Train Loss', color='#ff7f0e', linewidth=2)
    axes[0].plot(epochs, history_swin['val_loss'], 's--', label='Swin Val Loss', color='#ffbb78', linewidth=2)
    axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss Value', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, linestyle=':', alpha=0.6)

    # F1 and IoU Plots
    axes[1].plot(epochs, history_baseline['val_f1'], 'o-', label='Baseline Val F1', color='#2ca02c', linewidth=2)
    axes[1].plot(epochs, history_baseline['val_iou'], 's--', label='Baseline Val IoU', color='#98df8a', linewidth=2)
    axes[1].plot(epochs, history_swin['val_f1'], 'o-', label='Swin Val F1', color='#d62728', linewidth=2)
    axes[1].plot(epochs, history_swin['val_iou'], 's--', label='Swin Val IoU', color='#ff9896', linewidth=2)
    axes[1].set_title('Validation Metrics (F1 & IoU)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Score Value', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300)
    plt.close()
    print("Saved training curves to output directory.")

def plot_precision_recall_f1(metrics_baseline, metrics_swin, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    labels = ['Precision', 'Recall', 'F1-score']
    baseline_vals = [metrics_baseline['precision'], metrics_baseline['recall'], metrics_baseline['f1']]
    swin_vals = [metrics_swin['precision'], metrics_swin['recall'], metrics_swin['f1']]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, baseline_vals, width, label='Siamese CNN Baseline', color='#1f77b4', edgecolor='black', alpha=0.85)
    rects2 = ax.bar(x + width/2, swin_vals, width, label='Siamese Swin Transformer', color='#ff7f0e', edgecolor='black', alpha=0.85)
    
    ax.set_ylabel('Scores', fontsize=12)
    ax.set_title('Performance Comparison: Baseline vs Swin', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis='y', linestyle=':', alpha=0.6)
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='semibold')
                        
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'precision_recall_f1_comparison.png'), dpi=300)
    plt.close()
    print("Saved precision-recall-f1 comparison bar chart.")

def plot_distributions(dashboard_data, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Change-pixel distribution
    axes[0].hist(dashboard_data['change_percentage_ground_truth'], bins=10, color='#9467bd', edgecolor='black', alpha=0.85)
    axes[0].set_title('Change Pixel Distribution (GT)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('% of Changed Pixels', fontsize=11)
    axes[0].set_ylabel('Number of Samples', fontsize=11)
    axes[0].grid(True, linestyle=':', alpha=0.6)
    
    # GT vs Prediction Change Percentage Scatter
    axes[1].scatter(dashboard_data['change_percentage_ground_truth'], dashboard_data['change_percentage_prediction'], color='#e377c2', edgecolor='black', s=80, alpha=0.85)
    max_val = max(dashboard_data['change_percentage_ground_truth'].max(), dashboard_data['change_percentage_prediction'].max(), 0.1)
    axes[1].plot([0, max_val], [0, max_val], '--', color='grey', label='Perfect Match')
    axes[1].set_title('GT vs Predicted Change Percentage', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Ground Truth %', fontsize=11)
    axes[1].set_ylabel('Prediction %', fontsize=11)
    axes[1].legend()
    axes[1].grid(True, linestyle=':', alpha=0.6)
    
    # XAI overlap distribution
    axes[2].hist(dashboard_data['xai_overlap'], bins=5, color='#17becf', edgecolor='black', alpha=0.85)
    axes[2].set_title('XAI Overlap (Explanation IoU)', fontsize=13, fontweight='bold')
    axes[2].set_xlabel('Explanation IoU Score', fontsize=11)
    axes[2].set_ylabel('Number of Samples', fontsize=11)
    axes[2].grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'statistical_distributions.png'), dpi=300)
    plt.close()
    print("Saved statistical distributions to output directory.")
