from __future__ import annotations

import hashlib
import os
import pickle
import platform
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

from opmodel import settings as _settings

FORMAT = "v1"
ENV_DIR = "GEMM_SEARCH_CACHE_DIR"
ENV_MEMO = "GEMM_SEARCH_MEMO"
_MEMO_ENTRIES = 2048
_UNSET = object()

_config: dict = {"directory": _UNSET, "memory": _UNSET}
_memo: "OrderedDict[bytes, list]" = OrderedDict()
_code_hash: str | None = None
stats = {"memory_hits": 0, "disk_hits": 0, "misses": 0, "stores": 0, "disk_errors": 0}

def configure(*, directory: Any = _UNSET, memory: Any = _UNSET) -> None:
    if directory is not _UNSET:
        _config["directory"] = directory
    if memory is not _UNSET:
        _config["memory"] = memory
    _memo.clear()

def reset() -> None:
    _config.update(directory=_UNSET, memory=_UNSET)
    _memo.clear()
    for name in stats:
        stats[name] = 0

def _directory() -> Path | None:
    value = _config["directory"]
    if value is _UNSET:
        value = os.environ.get(ENV_DIR) or None
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{ENV_DIR} must be an absolute path (predict_energy changes directory): {value!r}")
    return path

def enabled() -> bool:
    memory = _config["memory"]
    if memory is _UNSET:
        memory = os.environ.get(ENV_MEMO) == "1"
    return bool(memory) or _directory() is not None

def code_hash() -> str:
    global _code_hash
    if _code_hash is None:
        package = Path(__file__).resolve().parent
        root = package.parents[2]
        digest = hashlib.sha256()
        digest.update(f"{FORMAT}\0{sys.version}\0{platform.platform()}\0{platform.machine()}\0".encode())
        files = sorted(
            (path.relative_to(package).as_posix(), path)
            for directory, subdirs, names in os.walk(package)
            for path in (Path(directory) / name for name in names)
            if "__pycache__" not in path.parts and path.suffix not in (".pyc", ".pyo"))
        files += [(f"../../../{name}", root / name) for name in ("opmodel_ops.py", "base_timing.py")
                  if (root / name).is_file()]
        for name, path in files:
            data = path.read_bytes()
            digest.update(f"{name}\0{len(data)}\0".encode())
            digest.update(data)
        _code_hash = digest.hexdigest()
    return _code_hash

def _reads_key(reads: dict) -> bytes:
    return pickle.dumps(tuple(sorted((name, type(value).__name__, repr(value)) for name, value in reads.items())),
                        protocol=4)

def lookup_or_compute(arguments: tuple, catalog: tuple, compute: Callable[[], Any]) -> Any:
    key = pickle.dumps((FORMAT, arguments, catalog), protocol=4)
    variants = _memo.get(key)
    if variants is not None:
        for reads, value in variants:
            if _settings.unchanged(reads):
                stats["memory_hits"] += 1
                _memo.move_to_end(key)
                return value
    directory = _directory()
    entry_dir = None
    if directory is not None:
        key_sha = hashlib.sha256(key).hexdigest()
        entry_dir = directory / FORMAT / code_hash() / key_sha[:2] / key_sha
        value = _disk_lookup(entry_dir, key)
        if value is not None:
            reads, value = value
            _remember(key, reads, value)
            stats["disk_hits"] += 1
            return value
    stats["misses"] += 1
    with _settings.recording() as reads:
        value = compute()
    if value is None:
        return value
    reads = dict(reads)
    _remember(key, reads, value)
    if entry_dir is not None:
        _disk_store(entry_dir, key, reads, value)
    return value

def _remember(key: bytes, reads: dict, value: Any) -> None:
    variants = _memo.get(key)
    if variants is None:
        variants = _memo[key] = []
        if len(_memo) > _MEMO_ENTRIES:
            _memo.popitem(last=False)
    variants.append((reads, value))

def _disk_lookup(entry_dir: Path, key: bytes):
    try:
        names = sorted(os.listdir(entry_dir))
    except FileNotFoundError:
        return None
    except OSError:
        stats["disk_errors"] += 1
        return None
    for name in names:
        if not name.endswith(".pkl"):
            continue
        try:
            with open(entry_dir / name, "rb") as handle:
                stored_key, reads, value = pickle.load(handle)
        except Exception:
            stats["disk_errors"] += 1
            continue
        if stored_key == key and _settings.unchanged(reads):
            return reads, value
    return None

def _disk_store(entry_dir: Path, key: bytes, reads: dict, value: Any) -> None:
    try:
        payload = pickle.dumps((key, reads, value), protocol=4)
        entry_dir.mkdir(parents=True, exist_ok=True)
        final = entry_dir / (hashlib.sha256(_reads_key(reads)).hexdigest() + ".pkl")
        temporary = entry_dir / f".{final.name}.{os.getpid()}.{os.urandom(6).hex()}.tmp"
        try:
            with open(temporary, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, final)
        finally:
            if temporary.exists():
                temporary.unlink()
        stats["stores"] += 1
    except Exception:
        stats["disk_errors"] += 1
