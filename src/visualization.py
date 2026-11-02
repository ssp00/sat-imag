import os
import json
import matplotlib.pyplot as plt
import numpy as np

def plot_training_curves(histories_dict, output_dir):
    """
    Plots training loss, validation loss, F1, and IoU curves for all models in histories_dict.
    histories_dict: dict of {model_label: history_dict}
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    colors = {
        'Siamese CNN': '#1f77b4',
        'Siamese U-Net': '#2ca02c',
        'Swin Transformer': '#ff7f0e',
        'ChangeFormer': '#9467bd'
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for model_name, hist in histories_dict.items():
        color = colors.get(model_name, '#333333')
        epochs = range(1, len(hist['train_loss']) + 1)
        axes[0].plot(epochs, hist['train_loss'], 'o-', label=f"{model_name} (Train)", color=color, linewidth=2)
        axes[0].plot(epochs, hist['val_loss'], 's--', label=f"{model_name} (Val)", color=color, alpha=0.7, linewidth=1.5)

        axes[1].plot(epochs, hist['val_f1'], 'o-', label=f"{model_name} F1", color=color, linewidth=2)
        axes[1].plot(epochs, hist['val_iou'], '^:', label=f"{model_name} IoU", color=color, alpha=0.7, linewidth=1.5)

    axes[0].set_title('Training & Validation Loss Across Models', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss Value', fontsize=12)
    axes[0].legend(fontsize=9, loc='upper right')
    axes[0].grid(True, linestyle=':', alpha=0.6)

    axes[1].set_title('Validation Performance (F1 & IoU)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Metric Score', fontsize=12)
    axes[1].legend(fontsize=9, loc='lower right')
    axes[1].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300)
    plt.close()
    print("Saved multi-model training curves to output directory.")

def plot_precision_recall_f1(metrics_dict, output_dir):
    """
    Grouped bar chart comparing Precision, Recall, and F1-score across all models.
    metrics_dict: dict of {model_label: {'precision': float, 'recall': float, 'f1': float}}
    """
    os.makedirs(output_dir, exist_ok=True)
    models = list(metrics_dict.keys())
    metric_names = ['Precision', 'Recall', 'F1-score']

    x = np.arange(len(models))
    width = 0.25

    colors = ['#1f77b4', '#2ca02c', '#ff7f0e']

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, m_name in enumerate(['precision', 'recall', 'f1']):
        vals = [metrics_dict[m][m_name] for m in models]
        rects = ax.bar(x + (i - 1) * width, vals, width, label=metric_names[i], color=colors[i], edgecolor='black', alpha=0.85)
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, fontweight='semibold')

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Comprehensive 4-Model Evaluation: Precision vs Recall vs F1', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis='y', linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'precision_recall_f1_comparison.png'), dpi=300)
    plt.close()
    print("Saved precision-recall-f1 comparison bar chart.")

def plot_distributions(dashboard_data, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # 1. Change-pixel distribution
    axes[0].hist(dashboard_data['true_change_percentage'] if 'true_change_percentage' in dashboard_data.columns else dashboard_data['change_percentage_ground_truth'],
                 bins=10, color='#9467bd', edgecolor='black', alpha=0.85)
    axes[0].set_title('Ground Truth Change Pixel Distribution', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('% of Changed Pixels', fontsize=11)
    axes[0].set_ylabel('Number of Samples', fontsize=11)
    axes[0].grid(True, linestyle=':', alpha=0.6)

    # 2. GT vs Prediction Change Percentage Scatter
    gt_col = 'true_change_percentage' if 'true_change_percentage' in dashboard_data.columns else 'change_percentage_ground_truth'
    pred_col = 'predicted_change_percentage' if 'predicted_change_percentage' in dashboard_data.columns else 'change_percentage_prediction'
    model_col = 'model' if 'model' in dashboard_data.columns else 'model_type'

    for model_name, grp in dashboard_data.groupby(model_col):
        axes[1].scatter(grp[gt_col], grp[pred_col], s=60, alpha=0.8, edgecolors='black', label=model_name)

    max_val = max(dashboard_data[gt_col].max(), dashboard_data[pred_col].max(), 1.0)
    axes[1].plot([0, max_val], [0, max_val], '--', color='grey', label='Ideal Match (y=x)')
    axes[1].set_title('Actual vs Predicted Change Percentage', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Ground Truth %', fontsize=11)
    axes[1].set_ylabel('Predicted %', fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, linestyle=':', alpha=0.6)

    # 3. XAI Overlap Distribution
    axes[2].hist(dashboard_data['xai_overlap'].dropna(), bins=8, color='#17becf', edgecolor='black', alpha=0.85)
    axes[2].set_title('XAI Overlap (Explanation IoU) Distribution', fontsize=13, fontweight='bold')
    axes[2].set_xlabel('Explanation IoU Score', fontsize=11)
    axes[2].set_ylabel('Sample Count', fontsize=11)
    axes[2].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'statistical_distributions.png'), dpi=300)
    plt.close()
    print("Saved statistical distributions to output directory.")
