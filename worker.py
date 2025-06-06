import argparse
import cv2
import glob
import os
from archs_tinygrad.rrdbnet_arch import RRDBNet
from archs_tinygrad.srvgg_arch import SRVGGNetCompact
from utils import *
from tinygrad.nn.state import safe_load, load_state_dict
import socket
import threading
import asyncio
import pickle
import time
from tinygrad import Tensor, TinyJit
from net_utils import *

model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
load_state_dict(model, safe_load(f'weights/realesr-general-x4v3.safetensors'))

@TinyJit
def forward_jit(x):
    return model(x)

async def tcp_echo_client(message):
    reader, writer = await asyncio.open_connection('127.0.0.1', 8888)

    while True:
        tile = await recv_tile(reader)

        x = Tensor(pickle.loads(tile))
        print(x)
        out = forward_jit(x).numpy()

        obj = pickle.dumps(out)
        await send_tile(writer, obj)

    print('Close the connection')
    writer.close()

asyncio.run(tcp_echo_client('Hello World!'))