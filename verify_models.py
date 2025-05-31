from archs_tinygrad.rrdbnet_arch import RRDBNet as RRDBNetTinygrad
from archs_torch.rrdbnet_arch import RRDBNet as RRDBNetTorch

from archs_tinygrad.srvgg_arch import SRVGGNetCompact as SRVGGNetCompactTinygrad
from archs_torch.srvgg_arch import SRVGGNetCompact as SRVGGNetCompactTorch

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

pth_path = "weights/realesr-animevideov3.pth"

model_torch = SRVGGNetCompactTorch(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
loadnet = torch.load(pth_path, map_location=torch.device('cpu'))
model_torch.load_state_dict(loadnet['params'], strict=True)

model_tinygrad = SRVGGNetCompactTinygrad(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
load_srvgg(model_tinygrad, pth_path, 16 * 2 + 3)

out_torch = model_torch(torch.ones(1, 3, 32, 32))
out_tinygrad = model_tinygrad(Tensor.ones(1, 3, 32, 32))

print("realesr-animevideov3.pth:", np.allclose(out_tinygrad.numpy(), out_torch.detach().numpy(), atol=1e-5, rtol=1e-5))

###

pth_path = "weights/realesr-general-x4v3.pth"

model_torch = SRVGGNetCompactTorch(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
loadnet = torch.load(pth_path, map_location=torch.device('cpu'))
model_torch.load_state_dict(loadnet['params'], strict=True)

model_tinygrad = SRVGGNetCompactTinygrad(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
load_srvgg(model_tinygrad, pth_path, 32 * 2 + 3)

out_torch = model_torch(torch.ones(1, 3, 32, 32))
out_tinygrad = model_tinygrad(Tensor.ones(1, 3, 32, 32))

print("realesr-general-x4v3.pth", np.allclose(out_tinygrad.numpy(), out_torch.detach().numpy(), atol=1e-5, rtol=1e-5))

###