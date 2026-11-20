import math

import sim_settings
from opmodel import settings as S

class LinkEnergyConfigError(ValueError):
    pass

def resolve_rule(hw, dimension):
    configured = getattr(dimension, "link_energy_rule", None) if dimension is not None else None
    if configured is None:
        configured = getattr(getattr(hw, "network_layout", None), "link_energy_rule", None)
    layer = {} if configured is None else {"RAPID_LINK_ENERGY_RULE": configured}
    try:
        with S.use(layer):
            rule = S.get("RAPID_LINK_ENERGY_RULE")
    except ValueError as exc:
        raise LinkEnergyConfigError(str(exc)) from exc
    if rule not in ("per_payload_byte", "line_rate_while_active"):
        raise LinkEnergyConfigError("network.link_energy_rule is required for multi-GPU traffic; choose "
                                    "per_payload_byte or line_rate_while_active")
    return rule

def line_rate_bps(dimension):
    values = []
    for name in ("links_per_gpu", "lanes_per_link", "lane_rate_bps"):
        value = getattr(dimension, name, None)
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise LinkEnergyConfigError(f"line_rate_while_active requires positive {name} on the network dimension") from None
        if not math.isfinite(value) or value <= 0:
            raise LinkEnergyConfigError(f"line_rate_while_active requires positive {name} on the network dimension")
        values.append(value)
    rate = values[0] * values[1] * values[2]
    if not math.isfinite(rate):
        raise LinkEnergyConfigError("network dimension line rate product must be finite")
    return rate
