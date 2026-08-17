"""Native microscopy dimension-aware I/O utilities.

Unlike the legacy 2-D analysis loader, this module preserves TIFF axes and
provides explicit C/Z/T selection/projection helpers. It avoids guessing when
metadata are available and raises on ambiguous requests instead of silently
collapsing scientifically meaningful dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import tifffile
except ImportError as exc:  # pragma: no cover
    tifffile = None
