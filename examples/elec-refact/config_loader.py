from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from config_schema import TrainingSettings


REFERENCE_PATTERN = re.compile(r"\$\{([^{}]+)\}")
SCIENTIFIC_NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)[eE][-+]?\d+")


def deep_merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(dict(merged[key]), value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise TypeError(f"YAML configuration must contain a mapping: {path}")
    return value


def _load_with_includes(path: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path in stack:
        cycle = " -> ".join(str(item) for item in (*stack, path))
        raise ValueError(f"Cyclic YAML include detected: {cycle}")
    if not path.is_file():
        raise FileNotFoundError(path)

    value = _read_yaml(path)
    includes = value.pop("includes", [])
    if isinstance(includes, str):
        includes = [includes]
    if not isinstance(includes, list) or not all(
        isinstance(include, str) for include in includes
    ):
        raise TypeError(f"includes must be a string or list of strings: {path}")

    merged: dict[str, Any] = {}
    for include in includes:
        merged = deep_merge(
            merged,
            _load_with_includes(path.parent / include, (*stack, path)),
        )
    return deep_merge(merged, value)


def _lookup(root: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = root
    for part in dotted_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(f"Unknown configuration reference: {dotted_path}")
        current = current[part]
    return current


def _resolve_token(token: str, root: Mapping[str, Any], stack: tuple[str, ...]) -> Any:
    if token.startswith("env:"):
        expression = token.removeprefix("env:")
        name, separator, default = expression.partition(",")
        if name in os.environ:
            return os.environ[name]
        if separator:
            return default
        raise KeyError(f"Required environment variable is not set: {name}")

    if token in stack:
        cycle = " -> ".join((*stack, token))
        raise ValueError(f"Cyclic configuration reference detected: {cycle}")
    return _resolve_value(_lookup(root, token), root, (*stack, token))


def _resolve_value(
    value: Any, root: Mapping[str, Any], stack: tuple[str, ...] = ()
) -> Any:
    if isinstance(value, dict):
        return {key: _resolve_value(item, root, stack) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_value(item, root, stack) for item in value]
    if not isinstance(value, str):
        return value

    full_match = REFERENCE_PATTERN.fullmatch(value)
    if full_match:
        return _resolve_token(full_match.group(1), root, stack)

    def replace(match: re.Match[str]) -> str:
        return str(_resolve_token(match.group(1), root, stack))

    previous = value
    for _ in range(100):
        resolved = REFERENCE_PATTERN.sub(replace, previous)
        if resolved == previous:
            return resolved
        previous = resolved
    raise ValueError(f"Configuration interpolation did not converge: {value}")


def _set_dotted_path(config: dict[str, Any], expression: str) -> None:
    dotted_path, separator, raw_value = expression.partition("=")
    if not separator or not dotted_path:
        raise ValueError(f"Override must use key=value syntax: {expression!r}")
    parts = dotted_path.split(".")
    current = config
    for part in parts[:-1]:
        child = current.get(part)
        if child is None:
            child = {}
            current[part] = child
        if not isinstance(child, dict):
            raise ValueError(f"Cannot set a child of non-mapping key: {dotted_path}")
        current = child
    parsed_value = yaml.safe_load(raw_value)
    if isinstance(parsed_value, str) and SCIENTIFIC_NUMBER_PATTERN.fullmatch(
        parsed_value
    ):
        parsed_value = float(parsed_value)
    current[parts[-1]] = parsed_value


def load_training_settings(
    config_paths: Iterable[str | Path],
    overrides: Iterable[str] = (),
) -> TrainingSettings:
    merged: dict[str, Any] = {}
    paths = [Path(path) for path in config_paths]
    if not paths:
        raise ValueError("At least one YAML configuration path is required.")
    for path in paths:
        merged = deep_merge(merged, _load_with_includes(path))
    for override in overrides:
        _set_dotted_path(merged, override)
    resolved = _resolve_value(merged, merged)
    return TrainingSettings.from_dict(resolved)
