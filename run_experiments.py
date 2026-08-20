import os
import json
import time
from src.train import run_training

def main():
    dataset_dir = r"d:\vit\SEMESTER-5\IV\DA\LEVIR CD\LEVIR CD"
    project_dir = r"d:\vit\SEMESTER-5\IV\DA"
    checkpoint_dir = os.path.join(project_dir, 'checkpoints')
    
    # Configuration for Baseline
    config_baseline = {
        'seed': 42,
        'use_cuda': True,
        'dataset_dir': dataset_dir,
        'project_dir': project_dir,
        'checkpoint_dir': checkpoint_dir,
        'model_type': 'baseline',
        'pretrained': False,
        'img_size': 224,
        'batch_size': 4,
        'lr': 1e-4,
        'epochs': 5,
        'num_workers': 0,
        'max_train_samples': 20,
        'max_val_samples': 5
    }

    # Configuration for Swin
    config_swin = {
        'seed': 42,
        'use_cuda': True,
        'dataset_dir': dataset_dir,
        'project_dir': project_dir,
        'checkpoint_dir': checkpoint_dir,
        'model_type': 'swin',
        'pretrained': True,
        'img_size': 224,
        'batch_size': 4,
        'lr': 5e-5,
        'epochs': 5,
        'num_workers': 0,
        'max_train_samples': 20,
        'max_val_samples': 5
    }

    print("=========================================")
    print("STARTING EXPERIMENT 1: SIAMESE CNN BASELINE")
    print("=========================================")
    t_start = time.time()
    hist_baseline = run_training(config_baseline)
    t_baseline = time.time() - t_start
    print(f"Baseline training completed in {t_baseline:.2f} seconds.")

    print("\n=========================================")
    print("STARTING EXPERIMENT 2: SIAMESE SWIN TRANSFORMER")
    print("=========================================")
    t_start = time.time()
    hist_swin = run_training(config_swin)
    t_swin = time.time() - t_start
    print(f"Swin training completed in {t_swin:.2f} seconds.")

    # Save experimental configuration for reproducibility
    exp_config = {
        'Python version': '3.14',
        'PyTorch version': '2.13.0+cpu',
        'CUDA version': 'None',
        'GPU': 'None',
        'image_size': 224,
        'batch_size': 4,
        'learning_rate_baseline': 1e-4,
        'learning_rate_swin': 5e-5,
        'optimizer': 'AdamW',
        'epochs': 5,
        'loss_function': 'BCE + Dice Loss',
        'train_samples': 20,
        'val_samples': 5,
        'test_samples': 5,
        'baseline_training_time_sec': t_baseline,
        'swin_training_time_sec': t_swin
    }
    
    config_out_path = os.path.join(project_dir, 'outputs', 'experiment_config.json')
    os.makedirs(os.path.dirname(config_out_path), exist_ok=True)
    with open(config_out_path, 'w') as f:
        json.dump(exp_config, f, indent=4)
    print(f"\nExperiment configuration saved to {config_out_path}")

if __name__ == '__main__':
    main()
