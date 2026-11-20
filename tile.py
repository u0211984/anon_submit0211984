from math import ceil
from functools import lru_cache
from enum import IntEnum
import math
from typing import Tuple, Union, NamedTuple

class AccessBytes(NamedTuple):
    reads: tuple[int, ...]
    writes: tuple[int, ...]
    def totals(self) -> tuple[int, ...]:
        return tuple(r + w for r, w in zip(self.reads, self.writes))

def describe_accesses(profile: AccessBytes) -> str:
    return ", ".join(
        f"L{lvl}: R={formatBytes(r)} W={formatBytes(w)}"
        for lvl, (r, w) in enumerate(zip(profile.reads, profile.writes))
    )

class Dataflow(IntEnum):
    NONE = 0
    BEST = 1
    WST = 2
    AST = 3
    OST = 4

class InnerLoop(IntEnum):
    M = 0
    K = 1
    N = 2

def as_dataflow_code(df: Union["Dataflow", int, str]) -> int:
    if isinstance(df, Dataflow):
        return int(df)
    if isinstance(df, int):
        return df
    if isinstance(df, str):
        s = df.lower()
        if s == "none":
            return int(Dataflow.NONE)
        if s == "best":
            return int(Dataflow.BEST)
        if s in ("wst", "ws"):
            return int(Dataflow.WST)
        if s in ("ast", "as"):
            return int(Dataflow.AST)
        if s in ("ost", "os"):
            return int(Dataflow.OST)
    raise ValueError(f"Unknown dataflow: {df}")
def inner_code_from_order(order_dims: str) -> InnerLoop:
    ch = order_dims[2]
    if ch == 'm':
        return InnerLoop.M
    if ch == 'k':
        return InnerLoop.K
    if ch == 'n':
        return InnerLoop.N
    raise ValueError(f"Unknown inner loop in order: {order_dims}")

def inner_code_to_char(code: Union[int, InnerLoop]) -> str:
    return 'm' if code == InnerLoop.M or code == 0 else (
        'k' if code == InnerLoop.K or code == 1 else 'n'
    )

def generate_orders_inner_codes(dataflow_code: int, dim1: int, dim2: int, dim3: int) -> Tuple[InnerLoop, ...]:
    if dataflow_code == Dataflow.BEST:
        if dim1 >= dim2 and dim1 >= dim3:
            dataflow_code = Dataflow.WST
        elif dim2 >= dim1 and dim2 >= dim3:
            dataflow_code = Dataflow.OST
        else:
            dataflow_code = Dataflow.AST

    if dataflow_code == Dataflow.WST:
        out = (InnerLoop.K, InnerLoop.N)
        return out if dim2 != dim3 else (InnerLoop.K,)
    if dataflow_code == Dataflow.AST:
        out = (InnerLoop.K, InnerLoop.M)
        return out if dim2 != dim1 else (InnerLoop.K,)
    if dataflow_code == Dataflow.OST:
        out = (InnerLoop.M, InnerLoop.N)
        return out if dim1 != dim3 else (InnerLoop.M,)

    return (InnerLoop.K, InnerLoop.N, InnerLoop.M)

def generate_tile_space(mem_layer, num_levels: int, dim1: int, dim2: int, dim3: int):
    original = False
    tile_space = []
    tiles = [None] * num_levels
    for level in range(0, num_levels - 1):
        memory = mem_layer[level]
        if not original and level == 2:
            tiles[level] = memory.get_gemm_based_tile_dims(dim1, dim2, dim3)
        else:
            tiles[level] = memory.get_tile_dims()

    if num_levels == 1:
        tile_space = []
    elif num_levels == 2:
        tile_space = tiles[0]
    elif num_levels == 3:
        tile_space = [(x, y) for x in tiles[0] for y in tiles[1]]
    elif num_levels == 4:
        tile_space = []
        for x in tiles[2]:
            t1, t2, t3 = x
            if not original:
                tiles[1] = mem_layer[1].get_gemm_based_tile_dims(t1, t2, t3)
            tile_strategy = [
                (x, y, z) for y in tiles[1] for z in tiles[2]
            ]
            tile_space.extend(tile_strategy)
    else:
        raise NotImplementedError()

    return tile_space

@lru_cache(maxsize=65536)
def _sysarray_accesses_sig(M: int, N: int, K: int,
                           FMA_x: int, FMA_y: int,
                           dataflow_code: int, dtype_size: int) -> float:
    reuse = 1
    if dataflow_code == Dataflow.BEST:
        a = -(-M // FMA_x)
        b = -(-N // FMA_x)
        c = -(-K // FMA_y)
        reuse = max(a,b,c)
    elif dataflow_code == Dataflow.WST:
        reuse = -(-M // FMA_x)
    elif dataflow_code == Dataflow.AST:
        reuse = -(-N // FMA_x)
    elif dataflow_code == Dataflow.OST:
        reuse = -(-K // FMA_y)

    GEMM_flop = M * N * (2 * K - 1)
    load_bytes = GEMM_flop * 2.0 * FMA_y / (FMA_x * (2 * FMA_y - 1)) * dtype_size
    store_bytes = GEMM_flop / (reuse * (2 * FMA_y - 1)) * dtype_size
    return load_bytes + store_bytes

def _l2_accesses_sig(M: int, N: int, K: int,
                     l1_M: int, l1_K: int, l1_N: int,
                     dtype_size: int) -> tuple[int, int]:
    grid_x = -(-N // l1_N)
    grid_y = -(-M // l1_M)

    read_bytes = grid_x * (M * K * dtype_size) + grid_y * (K * N * dtype_size)
    write_bytes = M * N * dtype_size
    return (read_bytes, write_bytes)

@lru_cache(maxsize=65536)
def _hbm_accesses_sig_simplified(M: int, N: int, K: int,
                                 l1_M: int, l1_K: int, l1_N: int,
                                 dtype_size: int, total_bytes: int, l2_read: int,
                                 capacity: int, num_sms: int) -> tuple[int, int]:
    tile_A = int(l1_M * l1_K)
    tile_B = int(l1_K * l1_N)
    tile_C = int(l1_M * l1_N)

    capacity -= (num_sms * tile_C)

    rows = -(-M // l1_M)
    cols = -(-N // l1_N)
    k_blocks = -(-K // l1_K)

    total_accesses = rows * cols * k_blocks

    cold_misses = rows * k_blocks * tile_A + cols * k_blocks * tile_B

    working_set = rows * tile_A + cols * tile_B

    if working_set > capacity:
        excess_ratio = min(1.0, (working_set - capacity) / max(1, working_set))
        capacity_misses = int(cold_misses * excess_ratio)
    else:
        capacity_misses = 0

    total_misses = cold_misses + capacity_misses

    write_bytes = M * N * dtype_size if total_bytes > capacity * dtype_size else 0

    per_transfer_bytes = max(1, (tile_A + tile_B))
    read_bytes = l2_read * total_misses / (total_accesses * per_transfer_bytes)

    return (int(read_bytes), int(write_bytes))

@lru_cache(maxsize=131072)
def _simulate_accesses_sig(inner_code: int, M: int, K: int, N: int,
                           l2_M: int, l2_K: int, l2_N: int,
                           l1_M: int, l1_K: int, l1_N: int,
                           dtype_size: int, FMA_x: int, FMA_y: int, dataflow_code: Union[int, Dataflow],
                           capacity: int, total_bytes: int, num_sms: int) -> Tuple[float, int, int, int]:
    sys_bytes = _sysarray_accesses_sig(M, N, K, FMA_x, FMA_y, dataflow_code, dtype_size)

    num_tiles_M = -(-M // l2_M)
    num_tiles_K = -(-K // l2_K)
    num_tiles_N = -(-N // l2_N)
    max_reload = num_tiles_M * num_tiles_N * num_tiles_K

    reuse_M = -(-l2_M // l1_M)
    reuse_K = -(-l2_K // l1_K)
    reuse_N = -(-l2_N // l1_N)
    read_bytes = (l1_M * l1_K * dtype_size + l1_K * l1_N * dtype_size) * (reuse_M * reuse_N * reuse_K)
    write_bytes = (l1_M * l1_N * dtype_size) * (reuse_M * reuse_N)
    l1_shared_total = (read_bytes + write_bytes) * max_reload

    l2_read, l2_write = _l2_accesses_sig(M, N, K, l1_M, l1_K, l1_N, dtype_size)

    hbm_read, hbm_write = _hbm_accesses_sig_simplified(M, N, K, l1_M, l1_K, l1_N, dtype_size, total_bytes, l2_read, capacity, num_sms)

    return AccessBytes(
        reads=(sys_bytes, read_bytes * max_reload, l2_read, hbm_read),
        writes=(0, write_bytes * max_reload, l2_write, hbm_write)
    )

class TiledGEMM:
    def __init__(self, order_dims, tile_dims, core, mem_layer, dtype_size=2):
        self.num_bundle = core.num_bundle
        tile_dims = list(tile_dims)

        self.mem_layer = mem_layer
        self.dtype_size = dtype_size
        self.M, self.K, self.N = tile_dims[3]
        self.capacity = mem_layer[2].size_per_bundle
        self.order_dims = order_dims if isinstance(order_dims, str) else None

        self._inner_code = (
            inner_code_from_order(order_dims)
            if isinstance(order_dims, str)
            else int(order_dims)
        )
        self.FMA_x, self.FMA_y = core.FMA_dims

        self.dataflow = core.dataflow
        self._dataflow_code = as_dataflow_code(self.dataflow)

        self.mk_bytes_ = self.M * self.K * self.dtype_size
        self.kn_bytes_ = self.K * self.N * self.dtype_size
        self.mn_bytes_ = self.M * self.N * self.dtype_size
        self.total_bytes = self.mk_bytes_ + self.kn_bytes_ + self.mn_bytes_

        l2_src = tile_dims[2]
        l1_src = tile_dims[1]
        l0_src = tile_dims[0]
        self.l2_M = min(l2_src[0], self.M)
        self.l2_K = min(l2_src[1], self.K)
        self.l2_N = min(l2_src[2], self.N)
        self.l1_M = min(l1_src[0], self.l2_M)
        self.l1_K = min(l1_src[1], self.l2_K)
        self.l1_N = min(l1_src[2], self.l2_N)
        self.l0_M = min(l0_src[0], self.l1_M)
        self.l0_K = min(l0_src[1], self.l1_K)
        self.l0_N = min(l0_src[2], self.l1_N)

        self.mem_accesses = self.simulate_accesses()

    @property
    def GEMM_flop(self):
        return self.M * self.N * (2 * self.K - 1)

    @property
    def bundle_util(self):
        reuse_M = -(-self.M // self.l2_M)
        reuse_K = -(-self.K // self.l2_K)
        reuse_N = -(-self.N // self.l2_N)

        gemm_waves = self.mem_layer[1].calc_waves_per_sm(
            self.M, self.K, self.N,
            self.l2_M, self.l2_K, self.l2_N
        )

        if gemm_waves == -1:
            return 1.0

        full_waves = math.floor(gemm_waves)
        partial_wave = gemm_waves - full_waves

        return (full_waves + partial_wave * partial_wave) / gemm_waves

    def sysarray_accesses(self):
        return _sysarray_accesses_sig(self.M, self.N, self.K, self.FMA_x, self.FMA_y, self._dataflow_code, self.dtype_size)

    def simulate_accesses(self):
        return _simulate_accesses_sig(
            self._inner_code,
            self.M, self.K, self.N,
            self.l2_M, self.l2_K, self.l2_N,
            self.l1_M, self.l1_K, self.l1_N,
            self.dtype_size, self.FMA_x, self.FMA_y, self._dataflow_code,
            self.capacity, self.total_bytes, self.num_bundle,
        )

    @classmethod
    def enumerate_candidates(cls, core, mem_layer, dim1: int, dim2: int, dim3: int,
                              dtype_size: int = 2, original: bool = False):
        df_code = as_dataflow_code(core.dataflow)
        inner_codes = generate_orders_inner_codes(df_code, dim1, dim2, dim3)
        space = generate_tile_space(mem_layer, len(mem_layer), dim1, dim2, dim3)
        for (l0, l1, l2) in space:
            tile_dims = (l0, l1, l2, (dim1, dim2, dim3))
            for code in inner_codes:
                yield cls(code, tile_dims, core, mem_layer, dtype_size)

    def per_layer_capacity(self):
        ds = self.dtype_size
        l2_total = (self.l2_M * self.l2_K * ds) + (self.l2_K * self.l2_N * ds) + (self.l2_M * self.l2_N * ds)
        l1_total = (self.l1_M * self.l1_K * ds) + (self.l1_K * self.l1_N * ds) + (self.l1_M * self.l1_N * ds)
        return [l2_total / self.mem_layer[2].size_per_bundle, l1_total / self.mem_layer[1].size_per_bundle]

    @property
    def mk_bytes(self):
        return self.mk_bytes_

    @property
    def kn_bytes(self):
        return self.kn_bytes_

    @property
    def mn_bytes(self):
        return self.mn_bytes_

def formatBytes(size):
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"

if __name__ == "__main__":
    import os
    import time

    class _Core:
        def __init__(self, num_bundle, FMA_dims, dataflow):
            self.num_bundle = num_bundle
            self.FMA_dims = FMA_dims
            self.dataflow = dataflow

    class _Mem:
        def __init__(self, size_per_bundle):
            self.size_per_bundle = size_per_bundle

    dataflows = ["none", "best", "wst", "ast", "ost"]
    orders = ["mkn", "kmn", "nmk", "mnk", "knm", "nkm"]
    cores = [_Core(num_bundle=4, FMA_dims=(8, 4), dataflow=df) for df in dataflows]

    mem_layer = [
        _Mem(256 * 1024),
        _Mem(192 * 1024),
        _Mem(40 * 1024 * 1024),
        _Mem(40 * 1024 * 1024 * 1024),
    ]

    small = [64, 96, 128, 160]*2
    medium = [256, 384, 512, 768, 1024]*2
    large = [2048, 2560, 3072]*2
    sizes = small + medium + large

    tile_shapes = [
        ((16, 16, 16), (64, 64, 64), (256, 256, 256)),
        ((16, 32, 16), (64, 128, 64), (256, 512, 256)),
        ((32, 16, 32), (128, 64, 128), (512, 256, 512)),
    ]

    build_count = 0
    checksum = 0.0
    t0 = time.perf_counter()
    repeat = 20
    for _ in range(repeat):
        for M in sizes:
            for N in sizes[:8]:
                for K in sizes[:8]:
                    for (l0, l1, l2) in tile_shapes:
                        l2M = min(l2[0], M); l2K = min(l2[1], K); l2N = min(l2[2], N)
                        l1M = min(l1[0], l2M); l1K = min(l1[1], l2K); l1N = min(l1[2], l2N)
                        l0M = min(l0[0], l1M); l0K = min(l0[1], l1K); l0N = min(l0[2], l1N)
                        tile_dims = [
                            (l0M, l0K, l0N),
                            (l1M, l1K, l1N),
                            (l2M, l2K, l2N),
                            (M, K, N),
                        ]
                        for order in orders:
                            for core in cores:
                                tg = TiledGEMM(order, tile_dims, core, mem_layer, dtype_size=2)
                                ma0, ma1, ma2, ma3 = tg.mem_accesses
                                checksum += (ma0 + ma1 + ma2 + ma3) * 1e-18
                                for uc in tg.per_layer_capacity():
                                    checksum += uc * 1e-6
                                build_count += 1

    t1 = time.perf_counter()
    elapsed = (t1 - t0)/repeat
    print(
        f"Built {build_count} TiledGEMM instances in avg {elapsed:.3f}s (repeat={repeat}), checksum={checksum:.3f}"
    )
