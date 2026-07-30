#!/usr/bin/env python3
"""Install the combined-model overlay into a local medtsllm4 checkout."""
from __future__ import annotations

import argparse
import ast
import shutil
from pathlib import Path


def patch_registry(path: Path) -> None:
    text = path.read_text()
    import_line = "from .tri_medtsllm import TriMedTsLLM\n"
    if import_line not in text:
        lines = text.splitlines(keepends=True)
        insert_at = 0
        for i, line in enumerate(lines):
            if line.startswith("from .") or line.startswith("import "):
                insert_at = i + 1
        lines.insert(insert_at, import_line)
        text = "".join(lines)

    if '"tri_medtsllm"' not in text and "'tri_medtsllm'" not in text:
        marker = "model_lookup = {"
        index = text.find(marker)
        if index < 0:
            raise RuntimeError(f"Could not find model_lookup in {path}")
        brace = text.find("{", index)
        text = text[: brace + 1] + '\n    "tri_medtsllm": TriMedTsLLM,' + text[brace + 1 :]

    ast.parse(text)
    path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", type=Path, help="Path to the medtsllm4 checkout")
    args = parser.parse_args()
    repo = args.repo.expanduser().resolve()
    here = Path(__file__).resolve().parent

    required = [repo / "models" / "medtsllm.py", repo / "models" / "__init__.py"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit("Not a medtsllm4 checkout; missing: " + ", ".join(missing))

    for name in ("tri_components.py", "tri_medtsllm.py"):
        shutil.copy2(here / "models" / name, repo / "models" / name)
    patch_registry(repo / "models" / "__init__.py")

    (repo / "scripts").mkdir(exist_ok=True)
    (repo / "configs").mkdir(exist_ok=True)
    shutil.copy2(
        here / "scripts" / "make_config.py",
        repo / "scripts" / "make_combined_config.py",
    )
    shutil.copy2(
        here / "configs" / "combined_overrides.toml",
        repo / "configs" / "combined_overrides.toml",
    )
    print(f"Installed combined model into {repo}")
    print(
        "Next: python scripts/make_combined_config.py --base "
        "<working-biomedcoop-config> --output configs/combined.toml"
    )


if __name__ == "__main__":
    main()
