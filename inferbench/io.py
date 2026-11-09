"""Small JSON helpers shared by runners and offline analysis."""

import json
import os
from pathlib import Path
import tempfile
import time


def stamp():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, obj):
    """Replace only a complete JSON file, including after interrupted writes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            name = handle.name
            json.dump(obj, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)
