from __future__ import annotations

from typing import Tuple

DEFAULT_REDUCTION_ELEMENT_BYTES = 2.0

def per_tx_byte(collective_type: str, participants: int, element_bytes: float = DEFAULT_REDUCTION_ELEMENT_BYTES
                ) -> Tuple[float, float, float]:
    kind = str(collective_type).upper()
    n = max(2, int(participants))
    q = (n - 1) / n
    b = float(element_bytes)
    if kind == "ALL_REDUCE":
        return 1.0 / q, (1.0 + 2.0 * q) / q, 1.0 / (2.0 * b)
    if kind == "REDUCE_SCATTER":
        return (1.0 + 1.0 / n) / q, (2.0 + q) / q, 1.0 / b
    if kind == "ALL_GATHER":
        return 1.0 / q, (1.0 + 2.0 * q) / q, 0.0
    if kind in {"P2P", "ALL_TO_ALL", "PIPELINE", "SEND", "RECV"}:
        return 2.0, 4.0, 0.0
    raise ValueError(f"no NCCL traffic model for collective type {collective_type!r}")

__all__ = ["per_tx_byte", "DEFAULT_REDUCTION_ELEMENT_BYTES"]
