import warnings
import torch
import time
from typing import Dict, Any

warnings.filterwarnings("ignore", category=FutureWarning)
from models.cav_mae_sync import CAVMAE


model = CAVMAE(audio_length=416, contrastive_heads=False, num_register_tokens=8, cls_token=True)
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
