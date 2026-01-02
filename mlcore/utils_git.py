from __future__ import annotations

import subprocess


def get_commit_hash(default: str = "unknown") -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return default
