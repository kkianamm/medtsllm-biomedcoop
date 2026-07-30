#!/usr/bin/env python3
"""Deep-merge combined_overrides.toml into an existing working TOML config."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    import tomllib
except ImportError:  # Python 3.10
    import tomli as tomllib  # type: ignore

try:
    import toml
except ImportError as exc:
    raise SystemExit("Install the small TOML writer first: pip install toml") from exc


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--overrides", type=Path, default=None)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent.parent
    overrides = args.overrides or here / "configs" / "combined_overrides.toml"
    with args.base.open("rb") as handle:
        base = tomllib.load(handle)
    with overrides.open("rb") as handle:
        update = tomllib.load(handle)
    if "models" not in base or "medtsllm" not in base["models"]:
        raise SystemExit("Base config must contain [models.medtsllm].")
    merged = deep_merge(base, update)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(toml.dumps(merged))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
