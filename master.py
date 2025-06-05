import argparse
import cv2
import glob
import os
from archs_tinygrad.rrdbnet_arch import RRDBNet
from archs_tinygrad.srvgg_arch import SRVGGNetCompact
from utils import *
from tinygrad.nn.state import safe_load, load_state_dict
from tinygrad import Tensor
import pickle
import socket
import threading
import asyncio
from net_utils import *
from tqdm import tqdm
import random
from collections import deque
import time

# tasks = asyncio.Queue()
tasks = deque()
done_tiles = 0
lock = asyncio.Lock()
progress_bar = tqdm(total=100, unit="it")

async def handle_echo(reader, writer):
    global done_tiles
    addr = writer.get_extra_info('peername')
    print(f"Connection from {addr}")

    while True:
        async with lock:
            if not tasks:
                break
            curr_task = tasks.popleft()

        x = np.random.rand(1, 3, 128, 128).astype(np.float32)
        obj = pickle.dumps(x)

        try:
            await send_tile(writer, obj)
            out = await recv_tile(reader)
            x = pickle.loads(out)

            done_tiles += 1
            async with lock:
                progress_bar.update(1)

        except:
            print('error occured')
            tasks.appendleft(curr_task)
            print('restoring:', curr_task)
            break

    print("Close the connection")
    writer.close()

async def background_job():
    img = cv2.imread(cv2.IMREAD_COLOR)

    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    tile_size = 128 
    _, _, height, width = img.shape

    tiles_x = math.ceil(width / tile_size)
    tiles_y = math.ceil(height / tile_size)

async def main():
    for i in range(1000):
        tasks.append(i)
    print(tasks)

    bg_task = asyncio.create_task(background_job())

    server = await asyncio.start_server(
        handle_echo, '0.0.0.0', 8888)
    addr = server.sockets[0].getsockname()
    print(f'Serving on {addr}')
    async with server:
        await server.serve_forever()
    
asyncio.run(main())