import torch
from tinygrad import Tensor

def load_rrdbnet(model, path, body_size):
    loadnet = torch.load(path, map_location=torch.device('cpu'))

    model.conv_first.weight = Tensor(loadnet['params_ema']['conv_first.weight'].numpy())
    model.conv_first.bias = Tensor(loadnet['params_ema']['conv_first.bias'].numpy())

    for i in range(body_size):
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

def load_srvgg(model, path):
    loadnet = torch.load(path, map_location=torch.device('cpu'))
    
    for i in range(35):
        model.body[i].weight = Tensor(loadnet['params'][f'body.{i}.weight'].numpy())
        if i % 2 == 0:
            model.body[i].bias = Tensor(loadnet['params'][f'body.{i}.bias'].numpy())