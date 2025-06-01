from archs_tinygrad.rrdbnet_arch import RRDBNet as RRDBNetTinygrad
from archs_torch.rrdbnet_arch import RRDBNet as RRDBNetTorch

from archs_tinygrad.srvgg_arch import SRVGGNetCompact as SRVGGNetCompactTinygrad
from archs_torch.srvgg_arch import SRVGGNetCompact as SRVGGNetCompactTorch

import torch
from tinygrad import Tensor, nn
import numpy as np
from tinygrad.nn.state import safe_save, safe_load, get_state_dict, load_state_dict
from utils import *
from safetensors.torch import save_model

###

model_name = "RealESRGAN_x4plus"
model_torch = RRDBNetTorch(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
loadnet = torch.load(f"weights/{model_name}.pth", map_location=torch.device('cpu'))
model_torch.load_state_dict(loadnet['params_ema'], strict=True)
save_model(model_torch, f"weights/{model_name}.safetensors")
print(f"{model_name}: done")

###

model_name = "RealESRNet_x4plus"
model_torch = RRDBNetTorch(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
loadnet = torch.load(f"weights/{model_name}.pth", map_location=torch.device('cpu'))
model_torch.load_state_dict(loadnet['params_ema'], strict=True)
save_model(model_torch, f"weights/{model_name}.safetensors")
print(f"{model_name}: done")

###

model_name = "RealESRGAN_x4plus_anime_6B"
model_torch = RRDBNetTorch(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
loadnet = torch.load(f"weights/{model_name}.pth", map_location=torch.device('cpu'))
model_torch.load_state_dict(loadnet['params_ema'], strict=True)
save_model(model_torch, f"weights/{model_name}.safetensors")
print(f"{model_name}: done")

###

model_name = "RealESRGAN_x2plus"
model_torch = RRDBNetTorch(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
loadnet = torch.load(f"weights/{model_name}.pth", map_location=torch.device('cpu'))
model_torch.load_state_dict(loadnet['params_ema'], strict=True)
save_model(model_torch, f"weights/{model_name}.safetensors")
print(f"{model_name}: done")

###

model_name = "realesr-animevideov3"
model_torch = SRVGGNetCompactTorch(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
loadnet = torch.load(f"weights/{model_name}.pth", map_location=torch.device('cpu'))
model_torch.load_state_dict(loadnet['params'], strict=True)
save_model(model_torch, f"weights/{model_name}.safetensors")
print(f"{model_name}: done")

###

model_name = "realesr-general-x4v3"
model_torch = SRVGGNetCompactTorch(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
loadnet = torch.load(f"weights/{model_name}.pth", map_location=torch.device('cpu'))
model_torch.load_state_dict(loadnet['params'], strict=True)
save_model(model_torch, f"weights/{model_name}.safetensors")
print(f"{model_name}: done")

###