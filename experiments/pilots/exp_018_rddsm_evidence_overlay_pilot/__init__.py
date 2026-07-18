"""Observer-only RDDSM evidence-overlay pilot."""

import sys
from pathlib import Path


_SOURCE_ROOT = Path(__file__).resolve().parents[3] / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from .protocol import PROTOCOL_VERSION, RUN_ID

__all__ = ["PROTOCOL_VERSION", "RUN_ID"]
