from __future__ import annotations

import math

_SOURCES = {
    "luo_2501_12084": {
        "cycles": {"DRAM": 566, "SRAM-L2": 202.8, "SRAM-L1": 29},
        "provenance": {"DRAM": "luo_2501_12084", "SRAM-L2": "luo_2501_12084",
                       "SRAM-L1": "luo_2501_12084"},
    },
    "dgna_2607_19922": {
        "cycles": {"DRAM": 385, "SRAM-L2": 218, "SRAM-L1": 29},
        "provenance": {"DRAM": "dgna_2607_19922", "SRAM-L2": "dgna_2607_19922",
                       "SRAM-L1": "luo_2501_12084"},
    },
}

def apply_latency_source(hw: dict, source: str) -> dict:
    if source not in _SOURCES:
        raise ValueError(f"unknown A100 latency source: {source!r}")
    tech = hw["tech_param"]
    clock = float(tech["core"]["operating_frequency"])
    if not math.isfinite(clock) or clock <= 0:
        raise ValueError("A100 operating frequency must be finite and positive")
    choice = _SOURCES[source]

    levels = {name: tech[name] for name in choice["cycles"]}
    for name, cycles in choice["cycles"].items():
        levels[name]["latency_cycles"] = cycles
        levels[name]["latency"] = float(f"{cycles / clock:.12g}")
    return {"source": source, "clock_hz": clock,
            "levels": dict(choice["provenance"])}
