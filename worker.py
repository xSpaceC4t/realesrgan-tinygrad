import argparse
import cv2
import glob
import os
# from archs_tinygrad.rrdbnet_arch import RRDBNet
# from archs_tinygrad.srvgg_arch import SRVGGNetCompact
from utils import *
from tinygrad.nn.state import safe_load, load_state_dict
import socket
import threading
import asyncio
import pickle
import time
from tinygrad import Tensor, TinyJit
from net_utils import *

def init_model(model_name, mode='gpu'):
    if mode == 'gpu':
        from archs_tinygrad.rrdbnet_arch import RRDBNet
        from archs_tinygrad.srvgg_arch import SRVGGNetCompact
    elif mode == 'cpu':
        from archs_torch.rrdbnet_arch import RRDBNet
        from archs_torch.srvgg_arch import SRVGGNetCompact

    if model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    elif model_name == 'RealESRNet_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    elif model_name == 'RealESRGAN_x4plus_anime_6B':  # x4 RRDBNet model with 6 blocks
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
    elif model_name == 'RealESRGAN_x2plus':  # x2 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    elif model_name == 'realesr-animevideov3':  # x4 VGG-style model (XS size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
    elif model_name == 'realesr-general-x4v3':  # x4 VGG-style model (S size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')

    if mode == 'gpu':
        load_state_dict(model, safe_load(f'weights/{model_name}.safetensors'))
    elif mode == 'cpu':
        import torch    
        loadnet = torch.load(f'weights/{model_name}.pth', map_location=torch.device('cpu'))
        if 'params_ema' in loadnet:
            keyname = 'params_ema'
        else:
            keyname = 'params'
        model.load_state_dict(loadnet[keyname], strict=True)

    return model

@TinyJit
def forward_jit(model, x):
    return model(x)

async def tcp_echo_client(message):
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--address', type=str, default='127.0.0.1')
    parser.add_argument('-p', '--port', type=str, default='8888')
    parser.add_argument('-m', '--mode', type=str, choices=['cpu', 'gpu'], default='gpu')
    args = parser.parse_args()

    reader, writer = await asyncio.open_connection(args.address, args.port)

    model_name = await recv_tile(reader)
    model_name = model_name.decode()
    print('Using:', model_name)

    # model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')

    model = init_model(model_name, args.mode)
    # load_state_dict(model, safe_load(f'weights/{model_name}.safetensors'))
    if args.mode == 'cpu':
        import torch

    while True:
        tile = await recv_tile(reader)

        x = pickle.loads(tile)
        print(x)
        if args.mode == 'gpu':
            out = forward_jit(model, Tensor(x)).numpy()
        elif args.mode == 'cpu':
            out = model.forward(torch.tensor(x)).detach().cpu()

        obj = pickle.dumps(out)
        await send_tile(writer, obj)

    print('Close the connection')
    writer.close()

asyncio.run(tcp_echo_client('Hello World!'))