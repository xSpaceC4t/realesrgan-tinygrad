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
import argparse

tasks = asyncio.Queue()
done_tasks = asyncio.Queue()
done_tiles = 0
lock = asyncio.Lock()
progress_bar = None

async def handle_echo(reader, writer, model_name):
    global done_tiles
    addr = writer.get_extra_info('peername')
    print(f"Connection from {addr}")

    await send_tile(writer, model_name.encode())
    # writer.write(len(model_name).to_bytes(4, byteorder='big'))
    # await writer.drain()
    # writer.write(model_name.encode())
    # await writer.drain()

    while True:
        curr_task = await tasks.get()
        if curr_task is None:
            break

        # x = np.random.rand(1, 3, 128, 128).astype(np.float32)
        obj = pickle.dumps(curr_task[0])

        try:
            await send_tile(writer, obj)
            out = await recv_tile(reader)
            x = pickle.loads(out)

            await done_tasks.put((x, curr_task[1], curr_task[2]))

            done_tiles += 1
            async with lock:
                progress_bar.update(1)

        except:
            print('Error occured!')
            print('Restoring task:', curr_task)
            await tasks.put((curr_task[0], curr_task[1], curr_task[2]))
            break

    print(f"Close the connection {addr}")
    writer.close()

def get_task(img, x, y, tile_size=128, tile_pad=10):
    _, _, height, width = img.shape
    tile_size = 128 - (tile_pad * 2)

    tiles_x = math.ceil(width / tile_size)
    tiles_y = math.ceil(height / tile_size)

    # extract tile from input image
    ofs_x = x * tile_size
    ofs_y = y * tile_size

    # input tile area on total image
    input_start_x = ofs_x
    input_end_x = min(ofs_x + tile_size, width)
    input_start_y = ofs_y
    input_end_y = min(ofs_y + tile_size, height)

    # input tile area on total image with padding
    input_start_x_pad = max(input_start_x - tile_pad, 0)
    input_end_x_pad = min(input_end_x + tile_pad, width)
    input_start_y_pad = max(input_start_y - tile_pad, 0)
    input_end_y_pad = min(input_end_y + tile_pad, height)

    # input tile dimensions
    input_tile_width = input_end_x - input_start_x
    input_tile_height = input_end_y - input_start_y
    # tile_idx = y * tiles_x + x + 1

    if x == 0:
        input_end_x_pad += tile_pad
    if y == 0:
        input_end_y_pad += tile_pad
    if x == tiles_x - 1:
        input_start_x_pad = width - (tile_size + 2 * tile_pad)
    if y == tiles_y - 1:
        input_start_y_pad = height - (tile_size + 2 * tile_pad)

    input_tile = img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]
    info = (input_start_x, input_end_x, input_start_y, input_end_y)
    info_pad = (input_start_x_pad, input_end_x_pad, input_start_y_pad, input_end_y_pad)

    return (input_tile, info, info_pad)

def finish_task(done_task, output, tile_size=128, tile_pad=10, scale=4):
    tile_size = 128 - (tile_pad * 2)

    output_tile = done_task[0]
    (input_start_x, input_end_x, input_start_y, input_end_y) = done_task[1]
    (input_start_x_pad, input_end_x_pad, input_start_y_pad, input_end_y_pad) = done_task[2]

    # output tile area on total image
    output_start_x = input_start_x * scale
    output_end_x = input_end_x * scale
    output_start_y = input_start_y * scale
    output_end_y = input_end_y * scale

    # output tile area without padding
    output_start_x_tile = (input_start_x - input_start_x_pad) * scale
    output_end_x_tile = output_start_x_tile + tile_size * scale
    output_start_y_tile = (input_start_y - input_start_y_pad) * scale
    output_end_y_tile = output_start_y_tile + tile_size * scale

    # put tile into output image
    output[:, :, output_start_y:output_end_y,
                output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                                            output_start_x_tile:output_end_x_tile] 

async def background_job(input_path, output_path):
    os.makedirs(output_path, exist_ok=True)

    if os.path.isfile(input_path):
        paths = [input_path]
    else:
        paths = sorted(glob.glob(os.path.join(input_path, '*')))

    for idx, path in enumerate(paths):
        imgname, extension = os.path.splitext(os.path.basename(path))
        print('Testing', idx, imgname)

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = cv2.imread('image.jpg', cv2.IMREAD_COLOR)

        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        global done_tiles
        done_tiles = 0

        tile_pad = 10
        tile_size = 128 - (tile_pad * 2)
        _, _, height, width = img.shape
        output = np.zeros((1, 3, height * 4, width * 4)).astype(np.float32)

        tiles_x = math.ceil(width / tile_size)
        tiles_y = math.ceil(height / tile_size)
        total = tiles_x * tiles_y

        global progress_bar
        progress_bar = tqdm(total=total, unit="it")

        for y in range(tiles_y):
            for x in range(tiles_x):
                await tasks.put(get_task(img, x, y))
        # await tasks.put(None)

        while total:
            done_task = await done_tasks.get() 
            finish_task(done_task, output)
            total -= 1
        progress_bar.close()

        output_img = np.squeeze(output, axis=0).astype(np.float32)
        output_img = np.clip(output_img, 0, 1)
        output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))
        output = (output_img * 255.0).round().astype(np.uint8)

        extension = extension[1:]
        suffix = 'out'

        save_path = os.path.join(output_path, f'{imgname}_{suffix}.{extension}')
        cv2.imwrite(save_path, output)

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='inputs', help='Input image or folder')
    parser.add_argument(
        '-n',
        '--model_name',
        type=str,
        default='RealESRGAN_x4plus',
        help=('Model names: RealESRGAN_x4plus | RealESRNet_x4plus | RealESRGAN_x4plus_anime_6B | RealESRGAN_x2plus | '
              'realesr-animevideov3 | realesr-general-x4v3'))
    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder')
    parser.add_argument(
        '-dn',
        '--denoise_strength',
        type=float,
        default=0.5,
        help=('Denoise strength. 0 for weak denoise (keep noise), 1 for strong denoise ability. '
              'Only used for the realesr-general-x4v3 model'))
    parser.add_argument('-s', '--outscale', type=float, default=4, help='The final upsampling scale of the image')
    parser.add_argument(
        '--model_path', type=str, default=None, help='[Option] Model path. Usually, you do not need to specify it')
    parser.add_argument('--suffix', type=str, default='out', help='Suffix of the restored image')
    parser.add_argument('-t', '--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
    parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')
    parser.add_argument('--face_enhance', action='store_true', help='Use GFPGAN to enhance face')
    parser.add_argument(
        '--fp32', action='store_true', help='Use fp32 precision during inference. Default: fp16 (half precision).')
    parser.add_argument(
        '--alpha_upsampler',
        type=str,
        default='realesrgan',
        help='The upsampler for the alpha channels. Options: realesrgan | bicubic')
    parser.add_argument(
        '--ext',
        type=str,
        default='auto',
        help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs')
    parser.add_argument(
        '-g', '--gpu-id', type=int, default=None, help='gpu device to use (default=None) can be 0,1,2 for multi-gpu')
    parser.add_argument('-a', '--address', type=str, default='127.0.0.1')
    parser.add_argument('-p', '--port', type=str, default='8888')

    args = parser.parse_args()

    # input_dir = 'frames'
    # output_dir = 'results'

    args.model_name = args.model_name.split('.')[0]

    bg_task = asyncio.create_task(background_job(args.input, args.output))

    server = await asyncio.start_server(
        lambda r, w: handle_echo(r, w, args.model_name),
        # handle_echo, 
        args.address, args.port)
    addr = server.sockets[0].getsockname()
    print(f'Serving on {addr}')
    async with server:
        await server.serve_forever()
    
asyncio.run(main())
