from __future__ import annotations

import dataclasses
import pickle
from collections import OrderedDict
from typing import Any, Callable, Hashable

from opmodel import settings as _settings

_MEMOS: list = []

def clear_all() -> None:
    for memo in _MEMOS:
        memo.clear()

class SettingsMemo:
    def __init__(self, max_entries: int) -> None:
        self._entries: "OrderedDict[Hashable, list]" = OrderedDict()
        self._max_entries = max_entries
        self.hits = self.misses = 0
        _MEMOS.append(self)

    def get_or_compute(self, key: Hashable, compute: Callable[[], Any]) -> Any:
        variants = self._entries.get(key)
        if variants is not None:
            for reads, value in variants:
                if _settings.unchanged(reads):
                    self.hits += 1
                    self._entries.move_to_end(key)
                    return value
        self.misses += 1
        with _settings.recording() as reads:
            value = compute()
        if variants is None:
            variants = self._entries[key] = []
            if len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
        variants.append((dict(reads), value))
        return value

    def clear(self) -> None:
        self._entries.clear()

def typed(value: Any) -> tuple:
    return (value, *map(type, vars(value).values()))

_PLAIN_DATACLASS: dict = {}

def replace(obj: Any, /, **changes: Any) -> Any:
    cls = obj.__class__
    plain = _PLAIN_DATACLASS.get(cls)
    if plain is None:
        params = getattr(cls, "__dataclass_fields__", None)
        instance = getattr(obj, "__dict__", None)
        plain = bool(
            params is not None and instance is not None
            and all(f._field_type is dataclasses._FIELD and f.init for f in params.values())
            and list(instance) == list(params))
        if plain and _init_only_stores_fields(cls, params):
            plain = _COPY
        _PLAIN_DATACLASS[cls] = plain
    if plain is _COPY:
        state = obj.__dict__
        if len(state) == len(cls.__dataclass_fields__):
            for name in changes:
                if name not in state:
                    break
            else:
                new = _object_new(cls)
                fresh = new.__dict__
                fresh.update(state)
                fresh.update(changes)
                return new
    if plain:
        return cls(**{**obj.__dict__, **changes})
    return dataclasses.replace(obj, **changes)

_COPY = 2
_object_new = object.__new__

def _init_only_stores_fields(cls: type, params: dict) -> bool:
    init = cls.__dict__.get("__init__")
    code = getattr(init, "__code__", None)
    if code is None or code.co_filename != "<string>" or init.__qualname__ != f"{cls.__qualname__}.__init__":
        return False
    if hasattr(cls, "__post_init__") or cls.__new__ is not object.__new__:
        return False
    if not cls.__dataclass_params__.frozen and cls.__setattr__ is not object.__setattr__:
        return False
    for name in params:
        if hasattr(type(getattr(cls, name, None)), "__set__"):
            return False
    return True

_HARDWARE_BY_ID: dict = {}
_HARDWARE_TOKENS: dict = {}

def hardware_token(hardware: Any) -> int:
    entry = _HARDWARE_BY_ID.get(id(hardware))
    if entry is None or entry[0] is not hardware:
        if len(_HARDWARE_BY_ID) >= 256:
            _HARDWARE_BY_ID.clear()
        token = _HARDWARE_TOKENS.setdefault(pickle.dumps(hardware, protocol=4), len(_HARDWARE_TOKENS))
        entry = _HARDWARE_BY_ID[id(hardware)] = (hardware, token)
    return entry[1]
