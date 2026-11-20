from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

def _nonnegative(value: float, name: str) -> float:
    parsed = float(value)
    if not isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative (got {value!r})")
    return parsed

@dataclass(frozen=True, slots=True)
class MemoryTraffic:
    read_bytes: float = 0.0
    write_bytes: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "read_bytes", _nonnegative(self.read_bytes, "read_bytes"))
        object.__setattr__(self, "write_bytes", _nonnegative(self.write_bytes, "write_bytes"))

    @property
    def total_bytes(self) -> float:
        return self.read_bytes + self.write_bytes

    def scaled(self, factor: float) -> "MemoryTraffic":
        scale = _nonnegative(factor, "scale factor")
        return MemoryTraffic(self.read_bytes * scale, self.write_bytes * scale)

    def __add__(self, other: "MemoryTraffic") -> "MemoryTraffic":
        if not isinstance(other, MemoryTraffic):
            return NotImplemented
        return MemoryTraffic(
            self.read_bytes + other.read_bytes,
            self.write_bytes + other.write_bytes,
        )

    def to_dict(self) -> dict[str, float]:
        return {"read_bytes": self.read_bytes, "write_bytes": self.write_bytes}

@dataclass(frozen=True, slots=True)
class OperationActivity:
    useful_tensor_flops: float = 0.0
    issued_tensor_flops: float = 0.0
    vector_flops: float = 0.0
    register: MemoryTraffic = field(default_factory=MemoryTraffic)
    smem: MemoryTraffic = field(default_factory=MemoryTraffic)
    l2: MemoryTraffic = field(default_factory=MemoryTraffic)
    hbm: MemoryTraffic = field(default_factory=MemoryTraffic)

    smem_staging_bytes: float = 0.0
    smem_staging_energy_bytes: float = 0.0

    l2_bypass_bytes: float = 0.0

    atomic_rmw_read_bytes: float = 0.0

    def __post_init__(self) -> None:
        for name in ("useful_tensor_flops", "issued_tensor_flops", "vector_flops",
                     "smem_staging_bytes", "smem_staging_energy_bytes", "l2_bypass_bytes",
                     "atomic_rmw_read_bytes"):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        if self.smem_staging_energy_bytes > self.smem_staging_bytes:
            raise ValueError("charged SMEM staging cannot exceed physical staging bytes")

    @property
    def excess_issued_tensor_flops(self) -> float:
        return max(0.0, self.issued_tensor_flops - self.useful_tensor_flops)

    def scaled(self, factor: float) -> "OperationActivity":
        scale = _nonnegative(factor, "scale factor")
        return OperationActivity(
            useful_tensor_flops=self.useful_tensor_flops * scale,
            issued_tensor_flops=self.issued_tensor_flops * scale,
            vector_flops=self.vector_flops * scale,
            register=self.register.scaled(scale),
            smem=self.smem.scaled(scale),
            l2=self.l2.scaled(scale),
            hbm=self.hbm.scaled(scale),
            smem_staging_bytes=self.smem_staging_bytes * scale,
            smem_staging_energy_bytes=self.smem_staging_energy_bytes * scale,
            l2_bypass_bytes=self.l2_bypass_bytes * scale,
            atomic_rmw_read_bytes=self.atomic_rmw_read_bytes * scale,
        )

    def __add__(self, other: "OperationActivity") -> "OperationActivity":
        if not isinstance(other, OperationActivity):
            return NotImplemented
        return OperationActivity(
            useful_tensor_flops=self.useful_tensor_flops + other.useful_tensor_flops,
            issued_tensor_flops=self.issued_tensor_flops + other.issued_tensor_flops,
            vector_flops=self.vector_flops + other.vector_flops,
            register=self.register + other.register,
            smem=self.smem + other.smem,
            l2=self.l2 + other.l2,
            hbm=self.hbm + other.hbm,
            smem_staging_bytes=self.smem_staging_bytes + other.smem_staging_bytes,
            smem_staging_energy_bytes=self.smem_staging_energy_bytes + other.smem_staging_energy_bytes,
            l2_bypass_bytes=self.l2_bypass_bytes + other.l2_bypass_bytes,
            atomic_rmw_read_bytes=self.atomic_rmw_read_bytes + other.atomic_rmw_read_bytes,
        )

    def l2_bypassed(self) -> "OperationActivity":
        return OperationActivity(
            useful_tensor_flops=self.useful_tensor_flops,
            issued_tensor_flops=self.issued_tensor_flops,
            vector_flops=self.vector_flops,
            register=self.register,
            smem=self.smem,
            l2=MemoryTraffic(),
            hbm=MemoryTraffic(self.l2.read_bytes + self.atomic_rmw_read_bytes, self.l2.write_bytes),
            smem_staging_bytes=self.smem_staging_bytes,
            smem_staging_energy_bytes=self.smem_staging_energy_bytes,
            l2_bypass_bytes=self.l2_bypass_bytes + self.l2.total_bytes,
            atomic_rmw_read_bytes=self.atomic_rmw_read_bytes,
        )

    def to_dict(self) -> dict[str, object]:
        result = {
            "useful_tensor_flops": self.useful_tensor_flops,
            "issued_tensor_flops": self.issued_tensor_flops,
            "vector_flops": self.vector_flops,
            "register": self.register.to_dict(),
            "smem": self.smem.to_dict(),
            "l2": self.l2.to_dict(),
            "hbm": self.hbm.to_dict(),
        }

        if self.smem_staging_bytes:
            result["smem_staging_bytes"] = self.smem_staging_bytes
            result["smem_staging_energy_bytes"] = self.smem_staging_energy_bytes
        if self.l2_bypass_bytes:
            result["l2_bypass_bytes"] = self.l2_bypass_bytes
        if self.atomic_rmw_read_bytes:
            result["atomic_rmw_read_bytes"] = self.atomic_rmw_read_bytes
        return result

ZERO_ACTIVITY = OperationActivity()

@dataclass(frozen=True, slots=True)
class OperationTimingResult:
    latency_s: float
    activity: OperationActivity = ZERO_ACTIVITY

    def __post_init__(self) -> None:
        object.__setattr__(self, "latency_s", _nonnegative(self.latency_s, "latency_s"))
