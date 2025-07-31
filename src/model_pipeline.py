import warnings
import torch
import time
from typing import Dict, Any

warnings.filterwarnings("ignore", category=FutureWarning)
from models.cav_mae_sync import CAVMAE


model = CAVMAE(audio_length=416, contrastive_heads=False, num_register_tokens=8, cls_token=True, keep_register_tokens=False)
pretrain_path = '/data/wanglinge/project/cav-mae/src/weight/init/ori_mae_11.pth'
mdl_weight = torch.load(pretrain_path, map_location='cpu')
new_state_dict = {}
for k, v in mdl_weight.items():
    if k.startswith('module.'):
        new_state_dict[k[7:]] = v  # strip 'module.'
    else:
        new_state_dict[k] = v
model_state = model.state_dict()
filtered_dict = {k: v for k, v in new_state_dict.items() if k in model_state and v.shape == model_state[k].shape}

# report mismatches
ignored_keys = [k for k in new_state_dict if k not in filtered_dict]
if ignored_keys:
    print(f"⚠️ Skipped loading {len(ignored_keys)} parameters due to shape mismatch or absence in model:")
    for k in ignored_keys:
        print(f"  - {k}")

model.load_state_dict(filtered_dict, strict=False)
print(f'✅ Successfully loaded {len(filtered_dict)} matching parameters.')

weight_path = '/home/chenyingying/tmp/cav-mae-sync/src/ckpt/audio_model.25.pth'
weight_dict = torch.load(weight_path, map_location='cpu')
# Remove 'module.' prefix if present
if 'module.' in list(weight_dict.keys())[0]:
    weight_dict = {k.replace('module.', ''): v for k, v in weight_dict.items()}
# Load the state dict into the model
# if hasattr(model, 'patch_embed_a'):
#     model.patch_embed_a.num_patches = int((model.audio_length / 16) * (128 / 16))
# else:
#     raise AttributeError("Model does not have 'patch_embed_a' attribute.")

model.load_state_dict(weight_dict, strict=True)
print("Model loaded successfully from", weight_path)
# sum up parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params / 1e6:.2f}M")

v_input = torch.randn(1, 1, 416, 128)

v = torch.randn(1, 3, 224, 224)  # Video input
a = torch.randn(1, 128, 416)     # Audio input
loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, mask_a, mask_v, c_acc, recon_a, recon_v, cls_a, cls_v = model(a, v)
print(f"Loss: {loss.item()}, Loss MAE: {loss_mae.item()}, Loss MAE A: {loss_mae_a.item()}, Loss MAE V: {loss_mae_v.item()}, Loss C: {loss_c.item()}")