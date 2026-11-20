from __future__ import annotations

from typing import Mapping, NewType

AxisName = str
DeviceId = NewType("DeviceId", int)
StageId = NewType("StageId", int)
LayerId = int
MicroBatch = int
CommKey = str
Coords = Mapping[AxisName, int]

from program.axes import DP_AXIS  # noqa: E402,F401

__all__ = [
    "AxisName",
    "DeviceId",
    "StageId",
    "LayerId",
    "MicroBatch",
    "CommKey",
    "Coords",
    "DP_AXIS",
]
