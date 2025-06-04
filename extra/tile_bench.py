from archs_tinygrad.srvgg_arch import SRVGGNetCompact
from tinygrad import Tensor, TinyJit, Variable
from tqdm import tqdm
import time
from tinygrad.nn.state import safe_load, load_state_dict
import argparse

@TinyJit
def forward_jit(model, x):
    return model(x)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('size')
    args = parser.parse_args()

    size = int(args.size)

    # model_name = 'realesr-animevideov3'
    model_name = 'realesr-general-x4v3'
    model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
    load_state_dict(model, safe_load(f'weights/{model_name}.safetensors'))

    # tile_sizes = [64, 128, 256, 512]
    # tile_sizes = [128, 256]
    steps = 100 

    # gtx 1060 tile size = 256
    # gt 730 tile size = 128

    # for size in tile_sizes:
    x = Tensor.rand(1, 3, size, size)

    # warmup
    for i in range(10):
        # model(x).realize()
        forward_jit(model, x).numpy()

    for i in range(steps):
        start = time.time()
        # model(x).realize()
        forward_jit(model, x).numpy()
        total = time.time() - start
        print(f'{i} | size = {size} | pixel per second:', size * size / total)