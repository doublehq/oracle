"""Collision-resistant names for PuLP variables and constraints."""

import hashlib
import re


def safe_lp_name(value: object) -> str:
    raw = str(value)
    readable = re.sub(r"[^A-Za-z0-9_]", "_", raw)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]
    return f"{readable}_{digest}"
