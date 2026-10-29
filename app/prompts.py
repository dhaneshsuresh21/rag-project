from __future__ import annotations

from functools import lru_cache

import yaml

from app.config import ROOT


@lru_cache
def load_prompt(name: str) -> dict:
    """Load a versioned prompt from prompts/<name>.yaml (must contain id + version)."""
    data = yaml.safe_load((ROOT / "prompts" / f"{name}.yaml").read_text(encoding="utf-8"))
    if not data.get("id") or not data.get("version"):
        raise ValueError(f"Prompt {name} must define 'id' and 'version'")
    return data


def prompt_ref(name: str) -> str:
    p = load_prompt(name)
    return f"{p['id']}@{p['version']}"
