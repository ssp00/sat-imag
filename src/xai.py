import torch
import torch.nn as nn
import numpy as np

class SwinAttentionRollout:
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.attention_maps = []
        self.register_hooks()

    def hook_fn(self, module, input, output):
        # output of softmax has shape: (num_windows * B, num_heads, N, N)
        self.attention_maps.append(output.detach().cpu())

    def register_hooks(self):
        # Register hooks to the softmax modules in stage 4 (layers_3 or layers[3])
        if hasattr(self.model, 'encoder'):
            encoder = self.model.encoder
            stage = None
            if hasattr(encoder, 'layers_3'):
                stage = encoder.layers_3
            elif hasattr(encoder, 'layers') and len(encoder.layers) > 3:
                stage = encoder.layers[3]
                
            if stage is not None:
                for block in stage.blocks:
                    hook = block.attn.softmax.register_forward_hook(self.hook_fn)
                    self.hooks.append(hook)

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def get_rollout(self, t1, t2):
        self.attention_maps.clear()
        device = next(self.model.parameters()).device
        
        # Forward pass (triggers hooks)
        _ = self.model(t1.to(device), t2.to(device))
        
        if len(self.attention_maps) < 4:
            # Fallback or error
            # For 2 images, Siamese model forwards each through encoder.
            # Stage 4 has 2 blocks, so we expect 2 blocks * 2 images = 4 maps.
            raise ValueError(f"Expected at least 4 attention maps, got {len(self.attention_maps)}")
            
        t1_maps = self.attention_maps[:2]
        t2_maps = self.attention_maps[2:4]
        
        rollout_t1 = self.compute_rollout(t1_maps)
        rollout_t2 = self.compute_rollout(t2_maps)
        
        # Combined rollout is the average of T1 and T2 rollout maps
        combined_rollout = (rollout_t1 + rollout_t2) / 2.0
        return combined_rollout

    def compute_rollout(self, maps):
        # Average over heads
        A0 = maps[0].mean(dim=1) # (W*B, 49, 49)
        A1 = maps[1].mean(dim=1) # (W*B, 49, 49)
        
        # Add Identity
        I = torch.eye(49).unsqueeze(0)
        A0_ext = A0 + I
        A1_ext = A1 + I
        
        # Normalize
        A0_ext = A0_ext / A0_ext.sum(dim=-1, keepdim=True)
        A1_ext = A1_ext / A1_ext.sum(dim=-1, keepdim=True)
        
        # Rollout
        R = torch.bmm(A1_ext, A0_ext)
        
        # Token attention averages
        rollout = R.mean(dim=1) # (W*B, 49)
        
        B = rollout.size(0)
        rollout = rollout.view(B, 1, 7, 7)
        
        # Interpolate back to 224x224
        rollout_resized = torch.nn.functional.interpolate(
            rollout, size=(224, 224), mode='bilinear', align_corners=True
        )
        return rollout_resized

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        self.register_hooks()

    def forward_hook(self, module, input, output):
        self.activations = output.detach()

    def backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def register_hooks(self):
        self.hooks.append(self.target_layer.register_forward_hook(self.forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(self.backward_hook))

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def generate_heatmap(self, t1, t2):
        self.model.zero_grad()
        device = next(self.model.parameters()).device
        logits = self.model(t1.to(device), t2.to(device))
        
        # Backward on sum of logits to visualize prediction drivers
        score = logits.sum()
        score.backward()

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = torch.relu(cam)

        cam_min = cam.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
        cam_max = cam.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        cam_resized = torch.nn.functional.interpolate(
            cam, size=(224, 224), mode='bilinear', align_corners=True
        )
        return cam_resized.cpu()

def compute_xai_overlap(heatmap, ground_truth, threshold=0.5):
    """
    Computes Explanation IoU: intersection(explanation_region, ground_truth) / union(explanation_region, ground_truth)
    """
    exp_region = (heatmap > threshold).float()
    gt_region = (ground_truth > 0.5).float()
    
    intersection = (exp_region * gt_region).sum().item()
    union = exp_region.sum().item() + gt_region.sum().item() - intersection
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

def compute_xai_stability(model, xai_generator, t1, t2, noise_std=0.05, is_swin=False):
    """
    Stability Measure: Computes cosine similarity between original and perturbed heatmap
    """
    # Original heatmap
    if is_swin:
        original_map = xai_generator.get_rollout(t1, t2)
    else:
        original_map = xai_generator.generate_heatmap(t1, t2)
        
    # Add noise
    t1_perturbed = t1 + torch.randn_like(t1) * noise_std
    t2_perturbed = t2 + torch.randn_like(t2) * noise_std
    
    # Perturbed heatmap
    if is_swin:
        # Re-register or clear rollout maps
        perturbed_map = xai_generator.get_rollout(t1_perturbed, t2_perturbed)
    else:
        perturbed_map = xai_generator.generate_heatmap(t1_perturbed, t2_perturbed)
        
    # Cosine similarity
    orig_flat = original_map.view(-1)
    pert_flat = perturbed_map.view(-1)
    
    cos_sim = torch.nn.functional.cosine_similarity(orig_flat, pert_flat, dim=0).item()
    return cos_sim
