"""Ensure the API resolves the joblib bundle when running pytest from any cwd."""

from __future__ import annotations

import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_BUNDLE = _ROOT / "artifacts" / "dt_inference_bundle.joblib"

if _BUNDLE.is_file():
    os.environ["ARTIFACT_PATH"] = str(_BUNDLE)
