from __future__ import annotations

import hashlib
import os
import pickle
import tempfile
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
CODE_FILES = ("flash_attention.py", "flash_attention_backward.py", "flash_attention_spills.py", "flash_native.c",
              "flash_native.py", "flash_kernel_resources.json", "flash_disk_cache.py")
MIN_SECONDS = 1.0
_DISABLED = False
_DIR: Path | None = None
_DIGEST: str | None = None

def code_digest() -> str:
    global _DIGEST
    if _DIGEST is None:
        h = hashlib.sha256()
        for name in CODE_FILES:
            h.update(name.encode() + b"\0" + (_HERE / name).read_bytes() + b"\0")
        _DIGEST = h.hexdigest()
    return _DIGEST

def _directory() -> Path:
    return _DIR if _DIR is not None else _HERE / ".native" / "flash_cache" / code_digest()[:16]

def _path(kind: str, exact_key: bytes) -> Path:
    key = hashlib.sha256(b"\0".join((kind.encode(), code_digest().encode(), exact_key))).hexdigest()
    return _directory() / f"{kind}-{key}.pkl"

def memo(kind: str, exact_key: bytes, compute):
    if _DISABLED or not isinstance(exact_key, bytes):
        return compute()
    path = _path(kind, exact_key)
    try:
        with open(path, "rb") as f:
            stored_key, value = pickle.load(f)
        if stored_key == exact_key:
            return value
    except Exception:
        pass
    t0 = time.perf_counter()
    value = compute()
    if time.perf_counter() - t0 >= MIN_SECONDS:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
            with os.fdopen(fd, "wb") as f:
                pickle.dump((exact_key, value), f, protocol=4)
            os.replace(tmp, path)
        except OSError:
            pass
    return value
