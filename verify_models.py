from archs_tinygrad.rrdbnet_arch import RRDBNet as RRDBNetTinygrad
from archs_torch.rrdbnet_arch import RRDBNet as RRDBNetTorch
import torch
from tinygrad import Tensor, nn
import numpy as np
from tinygrad.nn.state import safe_save, safe_load, get_state_dict, load_state_dict
from utils import *

###

pth_path = "weights/RealESRGAN_x4plus.pth"

model_torch = RRDBNetTorch(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
loadnet = torch.load(pth_path, map_location=torch.device('cpu'))
model_torch.load_state_dict(loadnet['params_ema'], strict=True)

model_tinygrad = RRDBNetTinygrad(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
load_rrdbnet(model_tinygrad, pth_path, 23)

out_torch = model_torch(torch.ones(1, 3, 32, 32))
out_tinygrad = model_tinygrad(Tensor.ones(1, 3, 32, 32))
print("RealESRGAN_x4plus.pth:", np.allclose(out_torch.detach().numpy(), out_tinygrad.numpy()))

###

pth_path = "weights/RealESRNet_x4plus.pth"

model_torch = RRDBNetTorch(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
loadnet = torch.load(pth_path, map_location=torch.device('cpu'))
model_torch.load_state_dict(loadnet['params_ema'], strict=True)

model_tinygrad = RRDBNetTinygrad(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
load_rrdbnet(model_tinygrad, pth_path, 23)

out_torch = model_torch(torch.ones(1, 3, 32, 32))
out_tinygrad = model_tinygrad(Tensor.ones(1, 3, 32, 32))
print("RealESRNet_x4plus.pth:", np.allclose(out_torch.detach().numpy(), out_tinygrad.numpy()))

###

pth_path = "weights/RealESRGAN_x4plus_anime_6B.pth"

model_torch = RRDBNetTorch(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
loadnet = torch.load(pth_path, map_location=torch.device('cpu'))
model_torch.load_state_dict(loadnet['params_ema'], strict=True)

model_tinygrad = RRDBNetTinygrad(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
load_rrdbnet(model_tinygrad, pth_path, 6)

out_torch = model_torch(torch.ones(1, 3, 32, 32))
out_tinygrad = model_tinygrad(Tensor.ones(1, 3, 32, 32))
print("RealESRGAN_x4plus_anime_6B.pth:", np.allclose(out_torch.detach().numpy(), out_tinygrad.numpy()))

###

pth_path = "weights/RealESRGAN_x2plus.pth"

model_torch = RRDBNetTorch(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
loadnet = torch.load(pth_path, map_location=torch.device('cpu'))
model_torch.load_state_dict(loadnet['params_ema'], strict=True)

model_tinygrad = RRDBNetTinygrad(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
load_rrdbnet(model_tinygrad, pth_path, 23)

out_torch = model_torch(torch.ones(1, 3, 32, 32))
out_tinygrad = model_tinygrad(Tensor.ones(1, 3, 32, 32))
print("RealESRGAN_x2plus.pth:", np.allclose(out_torch.detach().numpy(), out_tinygrad.numpy()))

###