from __future__ import annotations

import ctypes
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import warnings
from pathlib import Path

_SOURCE = Path(__file__).with_name("flash_native.c")
_FLAGS = ("-O3", "-std=c99", "-fPIC", "-shared", "-ffp-contract=off", "-fno-fast-math")
_FORCE_PYTHON = False
_LIB = None
_TRIED = False
_RESOURCE_KINDS = ("smem", "tensor", "tensor", "tensor", "tensor", "vector", "vector", "vector", "vector",
                   "sfu", "sfu", "sfu", "sfu")

def _compiler():
    for name in ("cc", "gcc", "clang"):
        path = shutil.which(name)
        if path:
            return path
    return None

def _compiler_line(cc) -> str | None:
    try:
        out = subprocess.run([cc, "--version"], check=True, capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    lines = out.stdout.decode("utf-8", "replace").splitlines()
    return lines[0].strip() if lines else ""

def _out_dirs():
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "rapid_llm_native"
    return (_SOURCE.parent / ".native", cache)

def _source_key(source: bytes, machine: str) -> str:
    return hashlib.sha256(b"\0".join((source, " ".join(_FLAGS).encode(), machine.encode(),
                                       sys.platform.encode()))).hexdigest()[:16]

def _name(key: str, cc_line: str, machine: str) -> str:
    build = hashlib.sha256(f"{key}\0{cc_line}".encode()).hexdigest()[:12]
    return f"flash_native-{key}-{build}-{machine}-{sys.platform}.so"

def _compile(cc, source: bytes, out_dir: Path, name: str) -> Path | None:
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=f".{name}.", suffix=".tmp", dir=out_dir)
    except OSError:
        return None
    os.close(fd)
    try:
        subprocess.run([cc, *_FLAGS, "-x", "c", "-o", tmp, "-"], input=source, check=True, capture_output=True,
                       timeout=120)
        os.chmod(tmp, 0o755)
        os.replace(tmp, out_dir / name)
        return out_dir / name
    except (OSError, subprocess.SubprocessError):
        return None
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass

def _load(path: Path):
    try:
        lib = ctypes.CDLL(str(path))
        ll, dbl = ctypes.POINTER(ctypes.c_longlong), ctypes.POINTER(ctypes.c_double)
        lib.flash_native_forward.argtypes = [ll, dbl, ll, ll, dbl, dbl, dbl, dbl, ll]
        lib.flash_native_forward.restype = ctypes.c_int
        lib.flash_native_backward.argtypes = [ll, dbl, ll, dbl, dbl, dbl, dbl, ll]
        lib.flash_native_backward.restype = ctypes.c_int
        lib.flash_native_test_set_serial0.argtypes = [ctypes.c_longlong]
        lib.flash_native_test_set_serial0.restype = None
        lib.flash_native_test_set_jump.argtypes = [ctypes.c_int]
        lib.flash_native_test_set_jump.restype = None
        lib.flash_native_jump_stats.argtypes = [ll]
        lib.flash_native_jump_stats.restype = None
        lib.flash_native_ff_stats.argtypes = [ll]
        lib.flash_native_ff_stats.restype = None
        return lib
    except (OSError, AttributeError):
        return None

def _open() -> tuple[object | None, str]:
    try:
        source = _SOURCE.read_bytes()
    except OSError:
        return None, f"cannot read {_SOURCE}"
    machine = platform.machine() or "unknown"
    key = _source_key(source, machine)
    dirs = _out_dirs()
    cc = _compiler()
    cc_line = _compiler_line(cc) if cc else None
    if cc_line is None:
        for out_dir in dirs:
            try:
                found = sorted(out_dir.glob(f"flash_native-{key}-*-{machine}-{sys.platform}.so"))
            except OSError:
                continue
            for path in found:
                lib = _load(path)
                if lib is not None:
                    return lib, ""
        return None, "no C compiler and no usable prebuilt library"
    name = _name(key, cc_line, machine)
    for out_dir in dirs:
        target = out_dir / name
        if target.is_file():
            lib = _load(target)
            if lib is not None:
                return lib, ""
    for out_dir in dirs:
        built = _compile(cc, source, out_dir, name)
        lib = _load(built) if built is not None else None
        if lib is not None:
            return lib, ""
    return None, f"building {_SOURCE.name} with {cc} failed"

_LOCK = threading.Lock()

def _library():
    global _LIB, _TRIED
    if _FORCE_PYTHON:
        return None
    if not _TRIED:
        with _LOCK:
            if not _TRIED:
                _LIB, reason = _open()
                _TRIED = True
                if _LIB is None:
                    warnings.warn(f"flash_native: {reason}; the FlashAttention timelines run the Python definition, "
                                  "which is much slower", RuntimeWarning, stacklevel=2)
    return _LIB

def _test_set_serial0(value: int) -> None:
    _library().flash_native_test_set_serial0(value)

def _test_set_jump(on: bool) -> None:
    _library().flash_native_test_set_jump(int(bool(on)))

def jump_stats() -> dict:
    out = (ctypes.c_longlong * 4)()
    _library().flash_native_jump_stats(out)
    return dict(zip(("snapshots", "certificates", "jumps", "periods_skipped"), out))

def available() -> bool:
    return _library() is not None

def _services(sms, service, first_take):
    names = ["l2", "hbm"] + [_RESOURCE_KINDS[i % 13] for i in range(13 * sms)]
    taken = sorted((first_take[i], i) for i in range(len(names)) if first_take[i] >= 0)
    services = {}
    for _, i in taken:
        kind = names[i]
        services[kind] = services.get(kind, 0.0) + service[i]
    return services

def _arrays(ncta, sms):
    nres = 2 + 13 * sms
    return ((ctypes.c_double * 1)(), (ctypes.c_double * ncta)(), (ctypes.c_double * ncta)(),
            (ctypes.c_double * nres)(), (ctypes.c_longlong * nres)())

def _check(rc):
    if rc != 0:
        raise RuntimeError(f"flash_native: native timeline failed (code {rc})")

def ff_stats() -> dict:
    out = (ctypes.c_longlong * 2)()
    _library().flash_native_ff_stats(out)
    return dict(zip(("skips", "tiles_skipped"), out))

def _ff(ff):
    return (0, 1, 0.0) if ff is None else (1, int(ff[1]), float(ff[0]))

def forward(*, sms, resident, warps, m, n, d, q_len, kv_len, dao, output_element_bytes, tiles, ranges,
            smem_rate, l2_rate, hbm_rate, tensor_rate, vector_rate, sfu_rate, mma_latency, smem_latency,
            l2_latency, hbm_latency, hbm_fraction, ff=None):
    lib = _library()
    if lib is None:
        return None
    ncta = len(tiles)
    ff_on, ff_n, ff_rtol = _ff(ff)
    ints = (ctypes.c_longlong * 14)(sms, resident, ncta, warps, m, n, d, q_len, kv_len, int(bool(dao)),
                                    output_element_bytes, int(ranges is not None), ff_on, ff_n)
    reals = (ctypes.c_double * 12)(smem_rate, l2_rate, hbm_rate, tensor_rate, vector_rate, sfu_rate,
                                   mma_latency, smem_latency, l2_latency, hbm_latency, hbm_fraction, ff_rtol)
    tile_arr = (ctypes.c_longlong * max(1, ncta))(*tiles)
    flat = [x for pair in ranges for x in pair] if ranges is not None else [0]
    range_arr = (ctypes.c_longlong * max(1, len(flat)))(*flat)
    cycles, starts, finishes, service, first = _arrays(ncta, sms)
    _check(lib.flash_native_forward(ints, reals, tile_arr, range_arr, cycles, starts, finishes, service, first))
    return cycles[0], list(starts), list(finishes), _services(sms, service, first)

def backward(*, kind, sms, resident, ctas, m, n, d, a, b, q_len, kv_len, grid0, nq, batch, q_heads, dao, minima,
             smem_rate, l2_rate, hbm_rate, tensor_rate, vector_rate, sfu_rate, mma_latency, smem_latency,
             l2_latency, hbm_latency, fraction, atomic_fraction, ff=None):
    lib = _library()
    if lib is None:
        return None
    code = {"dot_do_o": 0, "convert_dq": 1, "main": 2}[kind]
    ff_on, ff_n, ff_rtol = _ff(ff)
    ints = (ctypes.c_longlong * 18)(sms, resident, ctas, code, m, n, d, a, b, q_len, kv_len, grid0, nq, batch,
                                    q_heads, int(bool(dao)), ff_on, ff_n)
    reals = (ctypes.c_double * 13)(smem_rate, l2_rate, hbm_rate, tensor_rate, vector_rate, sfu_rate,
                                   mma_latency, smem_latency, l2_latency, hbm_latency, fraction, atomic_fraction,
                                   ff_rtol)
    minima_arr = (ctypes.c_longlong * max(1, len(minima)))(*minima)
    cycles, starts, finishes, service, first = _arrays(ctas, sms)
    _check(lib.flash_native_backward(ints, reals, minima_arr, cycles, starts, finishes, service, first))
    return cycles[0], list(starts), list(finishes), _services(sms, service, first)
