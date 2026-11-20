import hashlib
import io
import pickle
import types
import zlib
from contextlib import contextmanager

_ENABLED = [True]

def enabled() -> bool:
    return _ENABLED[0]

@contextmanager
def disabled():
    old = _ENABLED[0]
    _ENABLED[0] = False
    try:
        yield
    finally:
        _ENABLED[0] = old

def _bound_method(owner, module, qualname):
    raise AssertionError("trace keys are compared, never unpickled")

def _reduce_method(method):
    return _bound_method, (id(method.__self__), method.__func__.__module__, method.__func__.__qualname__)

def _encode(kernels, file):
    pickler = pickle.Pickler(file, protocol=5)
    pickler.dispatch_table = {types.MethodType: _reduce_method}
    pickler.fast = True
    pickler.dump(list(kernels))

def trace_key(kernels):
    buf = io.BytesIO()
    _encode(kernels, buf)
    return buf.getvalue()

class _Digest:
    def __init__(self):
        self.hash = hashlib.sha256()
        self.zip = zlib.compressobj(1)
        self.chunks = []

    def write(self, data):
        self.hash.update(data)
        self.chunks.append(self.zip.compress(data))
        return len(data)

class TraceKey:
    __slots__ = ("digest", "packed")

    def __init__(self, kernels):
        writer = _Digest()
        _encode(kernels, writer)
        writer.chunks.append(writer.zip.flush())
        self.digest = writer.hash.digest()
        self.packed = b"".join(writer.chunks)

    def __hash__(self):
        return hash(self.digest)

    def __eq__(self, other):
        if not isinstance(other, TraceKey):
            return NotImplemented
        return self.digest == other.digest and self.packed == other.packed
