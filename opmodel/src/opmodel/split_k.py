from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

REDUCTION_SCHEMES = {0: "none", 1: "inplace", 2: "compute_type", 4: "output_type"}

def normalize_split_k(split_k: int, reduction_scheme: str | int) -> tuple[int, str]:
    if isinstance(split_k, bool) or not isinstance(split_k, Integral) or split_k < 1:
        raise ValueError("split_k must be an integer >= 1")
    if isinstance(reduction_scheme, Integral) and not isinstance(reduction_scheme, bool):
        scheme = REDUCTION_SCHEMES.get(int(reduction_scheme))
    else:
        scheme = str(reduction_scheme).lower()
        scheme = scheme.removeprefix("cublaslt_reduction_scheme_")
        if scheme.isdigit():
            scheme = REDUCTION_SCHEMES.get(int(scheme))
    if scheme not in REDUCTION_SCHEMES.values():
        raise ValueError("reduction_scheme must be NONE/INPLACE/COMPUTE_TYPE/OUTPUT_TYPE (0/1/2/4)")
    if split_k > 1 and scheme == "none":
        raise ValueError("split_k > 1 requires a reduction scheme")
    return int(split_k), scheme

def split_k_lengths(k: int, split_k: int, cta_k: int,
                    reduction_scheme: str | int = "inplace") -> tuple[int, ...]:
    if k < 1 or cta_k < 1 or split_k < 1:
        raise ValueError("K, CTA K and split_k must be positive")
    _, scheme = normalize_split_k(split_k, reduction_scheme)
    if scheme in ("compute_type", "output_type"):
        partition = (k // cta_k // split_k) * cta_k
        return (partition,) * (split_k - 1) + (k - (split_k - 1) * partition,)
    stages = (k + cta_k - 1) // cta_k
    partition = ((stages + split_k - 1) // split_k) * cta_k
    return tuple(max(0, min(partition, k - i * partition)) for i in range(split_k))

@dataclass(frozen=True)
class SplitKCounts:
    workspace_bytes: int
    partial_write_bytes: int
    partial_read_bytes: int
    output_read_bytes: int
    output_write_bytes: int
    reduction_flops: int
    extra_launches: int
    partial_bytes: int

def split_k_counts(batch: int, m: int, n: int, k: int, split_k: int,
                   reduction_scheme: str | int, output_bytes: int = 2) -> SplitKCounts:
    s, scheme = normalize_split_k(split_k, reduction_scheme)
    elements = batch * m * n
    d = elements * output_bytes
    if min(batch, m, n, k, output_bytes) < 1:
        raise ValueError("GEMM dimensions and output bytes must be positive")
    separate = s > 1 and scheme in ("compute_type", "output_type")
    partial_bytes = 4 if scheme == "compute_type" else output_bytes
    workspace = s * elements * partial_bytes if separate else 0
    return SplitKCounts(
        workspace_bytes=workspace,
        partial_write_bytes=workspace,
        partial_read_bytes=workspace,
        output_read_bytes=(s - 1) * d if scheme == "inplace" else 0,
        output_write_bytes=s * d if scheme == "inplace" else d,
        reduction_flops=(s - 1) * elements,
        extra_launches=int(separate),
        partial_bytes=partial_bytes,
    )
