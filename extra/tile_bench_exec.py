import os

tile_sizes = [128, 256, 512]

for size in tile_sizes:
    os.system(f'python tile_bench.py {size}')