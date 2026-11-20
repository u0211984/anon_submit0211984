import os

from program.layout import CANONICAL_AXES, RankLayout, cluster_coords

__all__ = ["CANONICAL_AXES", "RankLayout", "cluster_coords", "_env_flag"]

from sim_settings import MODELLING_DEFAULTS_ON  # noqa: E402  (re-exported for callers)

def _env_flag(name: str) -> bool:
    from opmodel import settings as _settings
    if _settings.is_registered(name):
        return bool(_settings.get(name))
    value = os.environ.get(name)
    if value is None:
        return name in MODELLING_DEFAULTS_ON
    normalized = value.strip().lower()
    return normalized not in {"", "0", "false", "no"}
