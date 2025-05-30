from archs_tinygrad.rrdbnet_arch import RRDBNet as RRDBNetTinygrad
from archs_torch.rrdbnet_arch import RRDBNet as RRDBNetTorch
import torch
from tinygrad import Tensor, nn
import numpy as np

model_torch = RRDBNetTorch(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

loadnet = torch.load("weights/RealESRGAN_x4plus.pth", map_location=torch.device('cpu'))

# prefer to use params_ema
if 'params_ema' in loadnet:
    keyname = 'params_ema'
else:
    keyname = 'params'

model_torch.load_state_dict(loadnet[keyname], strict=True)

model = RRDBNetTinygrad(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

model.conv_first.weight = Tensor(loadnet['params_ema']['conv_first.weight'].numpy())
model.conv_first.bias = Tensor(loadnet['params_ema']['conv_first.bias'].numpy())

for i in range(23):
    # rdb1
    model.body[i].rdb1.conv1.weight = Tensor(loadnet['params_ema'][f'body.{i}.rdb1.conv1.weight'].numpy())
    model.body[i].rdb1.conv1.bias = Tensor(loadnet['params_ema'][f'body.{i}.rdb1.conv1.bias'].numpy())

    model.body[i].rdb1.conv2.weight = Tensor(loadnet['params_ema'][f'body.{i}.rdb1.conv2.weight'].numpy())
    model.body[i].rdb1.conv2.bias = Tensor(loadnet['params_ema'][f'body.{i}.rdb1.conv2.bias'].numpy())

    model.body[i].rdb1.conv3.weight = Tensor(loadnet['params_ema'][f'body.{i}.rdb1.conv3.weight'].numpy())
    model.body[i].rdb1.conv3.bias = Tensor(loadnet['params_ema'][f'body.{i}.rdb1.conv3.bias'].numpy())

    model.body[i].rdb1.conv4.weight = Tensor(loadnet['params_ema'][f'body.{i}.rdb1.conv4.weight'].numpy())
    model.body[i].rdb1.conv4.bias = Tensor(loadnet['params_ema'][f'body.{i}.rdb1.conv4.bias'].numpy())

    model.body[i].rdb1.conv5.weight = Tensor(loadnet['params_ema'][f'body.{i}.rdb1.conv5.weight'].numpy())
    model.body[i].rdb1.conv5.bias = Tensor(loadnet['params_ema'][f'body.{i}.rdb1.conv5.bias'].numpy())

    # rdb2
    model.body[i].rdb2.conv1.weight = Tensor(loadnet['params_ema'][f'body.{i}.rdb2.conv1.weight'].numpy())
    model.body[i].rdb2.conv1.bias = Tensor(loadnet['params_ema'][f'body.{i}.rdb2.conv1.bias'].numpy())

    model.body[i].rdb2.conv2.weight = Tensor(loadnet['params_ema'][f'body.{i}.rdb2.conv2.weight'].numpy())
    model.body[i].rdb2.conv2.bias = Tensor(loadnet['params_ema'][f'body.{i}.rdb2.conv2.bias'].numpy())

    model.body[i].rdb2.conv3.weight = Tensor(loadnet['params_ema'][f'body.{i}.rdb2.conv3.weight'].numpy())
    model.body[i].rdb2.conv3.bias = Tensor(loadnet['params_ema'][f'body.{i}.rdb2.conv3.bias'].numpy())

    model.body[i].rdb2.conv4.weight = Tensor(loadnet['params_ema'][f'body.{i}.rdb2.conv4.weight'].numpy())
    model.body[i].rdb2.conv4.bias = Tensor(loadnet['params_ema'][f'body.{i}.rdb2.conv4.bias'].numpy())

    model.body[i].rdb2.conv5.weight = Tensor(loadnet['params_ema'][f'body.{i}.rdb2.conv5.weight'].numpy())
    model.body[i].rdb2.conv5.bias = Tensor(loadnet['params_ema'][f'body.{i}.rdb2.conv5.bias'].numpy())

    # rdb 3
    model.body[i].rdb3.conv1.weight = Tensor(loadnet['params_ema'][f'body.{i}.rdb3.conv1.weight'].numpy())
    model.body[i].rdb3.conv1.bias = Tensor(loadnet['params_ema'][f'body.{i}.rdb3.conv1.bias'].numpy())

    model.body[i].rdb3.conv2.weight = Tensor(loadnet['params_ema'][f'body.{i}.rdb3.conv2.weight'].numpy())
    model.body[i].rdb3.conv2.bias = Tensor(loadnet['params_ema'][f'body.{i}.rdb3.conv2.bias'].numpy())

    model.body[i].rdb3.conv3.weight = Tensor(loadnet['params_ema'][f'body.{i}.rdb3.conv3.weight'].numpy())
    model.body[i].rdb3.conv3.bias = Tensor(loadnet['params_ema'][f'body.{i}.rdb3.conv3.bias'].numpy())

    model.body[i].rdb3.conv4.weight = Tensor(loadnet['params_ema'][f'body.{i}.rdb3.conv4.weight'].numpy())
    model.body[i].rdb3.conv4.bias = Tensor(loadnet['params_ema'][f'body.{i}.rdb3.conv4.bias'].numpy())

    model.body[i].rdb3.conv5.weight = Tensor(loadnet['params_ema'][f'body.{i}.rdb3.conv5.weight'].numpy())
    model.body[i].rdb3.conv5.bias = Tensor(loadnet['params_ema'][f'body.{i}.rdb3.conv5.bias'].numpy())

model.conv_body.weight = Tensor(loadnet['params_ema']['conv_body.weight'].numpy())
model.conv_body.bias = Tensor(loadnet['params_ema']['conv_body.bias'].numpy())

model.conv_up1.weight = Tensor(loadnet['params_ema']['conv_up1.weight'].numpy())
model.conv_up1.bias = Tensor(loadnet['params_ema']['conv_up1.bias'].numpy())

model.conv_up2.weight = Tensor(loadnet['params_ema']['conv_up2.weight'].numpy())
model.conv_up2.bias = Tensor(loadnet['params_ema']['conv_up2.bias'].numpy())

model.conv_hr.weight = Tensor(loadnet['params_ema']['conv_hr.weight'].numpy())
model.conv_hr.bias = Tensor(loadnet['params_ema']['conv_hr.bias'].numpy())

model.conv_last.weight = Tensor(loadnet['params_ema']['conv_last.weight'].numpy())
model.conv_last.bias = Tensor(loadnet['params_ema']['conv_last.bias'].numpy())

model_tinygrad = model

out_torch = model_torch(torch.ones(1, 3, 32, 32))
print(out_torch)
out_tinygrad = model_tinygrad(Tensor.ones(1, 3, 32, 32))
print(out_tinygrad.numpy())

print(np.allclose(out_torch.detach().numpy(), out_tinygrad.numpy()))