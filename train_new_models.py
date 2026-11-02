import os
import json
import time
from src.train import run_training

def main():
    dataset_dir = r"d:\vit\SEMESTER-5\IV\DA\LEVIR CD\LEVIR CD"
    project_dir = r"d:\vit\SEMESTER-5\IV\DA"
    checkpoint_dir = os.path.join(project_dir, 'checkpoints')

    # Config for Siamese U-Net
    config_unet = {
        'seed': 42,
        'use_cuda': True,
        'dataset_dir': dataset_dir,
        'project_dir': project_dir,
        'checkpoint_dir': checkpoint_dir,
        'model_type': 'siamese_unet',
        'pretrained': False,
        'img_size': 224,
        'batch_size': 2,
        'lr': 1e-4,
        'epochs': 5,
        'num_workers': 0,
        'max_train_samples': 20,
        'max_val_samples': 5
    }

    # Config for ChangeFormer
    config_changeformer = {
        'seed': 42,
        'use_cuda': True,
        'dataset_dir': dataset_dir,
        'project_dir': project_dir,
        'checkpoint_dir': checkpoint_dir,
        'model_type': 'changeformer',
        'pretrained': False,
        'img_size': 224,
        'batch_size': 2,
        'lr': 1e-4,
        'epochs': 5,
        'num_workers': 0,
        'max_train_samples': 20,
        'max_val_samples': 5
    }

    print("=========================================")
    print("STARTING TRAINING: SIAMESE U-NET")
    print("=========================================")
    t_start = time.time()
    hist_unet = run_training(config_unet)
    t_unet = time.time() - t_start
    print(f"Siamese U-Net training completed in {t_unet:.2f} seconds.")

    print("\n=========================================")
    print("STARTING TRAINING: CHANGEFORMER")
    print("=========================================")
    t_start = time.time()
    hist_cf = run_training(config_changeformer)
    t_cf = time.time() - t_start
    print(f"ChangeFormer training completed in {t_cf:.2f} seconds.")

    # Update experiment_config.json to reflect all 4 models
    exp_config_path = os.path.join(project_dir, 'outputs', 'experiment_config.json')
    exp_config = {}
    if os.path.exists(exp_config_path):
        with open(exp_config_path, 'r') as f:
            exp_config = json.load(f)

    exp_config['models_trained'] = ['Siamese CNN', 'Siamese U-Net', 'Swin Transformer', 'ChangeFormer']
    exp_config['siamese_unet_training_time_sec'] = t_unet
    exp_config['changeformer_training_time_sec'] = t_cf
    exp_config['learning_rate_siamese_unet'] = 1e-4
    exp_config['learning_rate_changeformer'] = 1e-4

    with open(exp_config_path, 'w') as f:
        json.dump(exp_config, f, indent=4)
    print(f"\nUpdated experiment configuration saved to {exp_config_path}")

if __name__ == '__main__':
    main()
