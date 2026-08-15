"""Small shared helpers: seeding, logging, JSON/array IO, run manifests."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

LOGGER_NAME = "asa"


def get_logger(name: str = LOGGER_NAME) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s %(name)s  %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def set_seed(seed: int) -> None:
    """Seed every RNG we touch. Seed variance is reported separately from fold
    variance (see the released protocol), so seeding must be exact and total."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def use_deterministic_algorithms(enabled: bool = True) -> None:
    try:
        import torch

        torch.backends.cudnn.deterministic = enabled
        torch.backends.cudnn.benchmark = not enabled
    except ImportError:
        pass


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def write_json(path: str | Path, payload: Any, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=indent, default=_json_default))


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def stable_hash(text: str, digits: int = 12) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:digits]


def git_revision(cwd: str | Path | None = None) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def provenance(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Environment stamp written next to every result file."""
    payload: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "git_revision": git_revision(Path(__file__).resolve().parent),
    }
    try:
        import torch

        payload["torch"] = torch.__version__
        payload["cuda"] = torch.version.cuda
        payload["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        payload["torch"] = None
    try:
        import cv2

        payload["opencv"] = cv2.__version__
    except ImportError:
        payload["opencv"] = None
    if extra:
        payload.update(extra)
    return payload
