"""Configuration: configs/default.yaml (the paper's method), optionally updated by experiment files and overrides.

Experiment files and `section.key=value` overrides may only change keys that exist in the defaults, so a typo is an
error instead of a silently ignored setting.
"""
import copy
from pathlib import Path

import yaml

DEFAULT = Path(__file__).resolve().parents[1] / 'configs' / 'default.yaml'


class Config(dict):
    """Nested dict with attribute access (cfg.material.lr_albedo)."""

    def __getattr__(self, name):
        try:
            value = self[name]
        except KeyError:
            raise AttributeError(name) from None
        return Config(value) if isinstance(value, dict) else value


def _update(base, new, where=''):
    for key, value in new.items():
        if key not in base:
            raise KeyError(f'unknown config key {where}{key}')
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise TypeError(f'{where}{key} is a section')
            _update(base[key], value, f'{where}{key}.')
        else:
            base[key] = value


def load_config(files=(), overrides=()):
    """Defaults, then each YAML file in `files`, then each 'a.b=value' string in `overrides` (value parsed as YAML)."""
    config = yaml.safe_load(DEFAULT.read_text())
    for f in files:
        _update(config, yaml.safe_load(Path(f).read_text()) or {})
    for item in overrides:
        key, sep, value = item.partition('=')
        if not sep:
            raise ValueError(f'override {item!r} is not key=value')
        nested = yaml.safe_load(value)
        for part in reversed(key.split('.')):
            nested = {part: nested}
        _update(config, nested)
    return Config(config)


def as_plain(config):
    return copy.deepcopy(dict(config))
