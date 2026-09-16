# FILE: src/eval/record.py
"""Minimal experiment tracking.

One small JSON per run. Paste this into your notebook alongside the metrics
functions; the repo copy is the canonical one.

No git SHA - there is no repo inside the Kaggle session. Instead pass a short
code_version string you bump by hand when the approach changes. Crude, but it
means a number in the report can still be tied to a specific setup.
"""

import datetime
import json
from pathlib import Path

RESULTS = Path("/kaggle/working/results")


def record(name, params, metrics, code_version="", notes=""):
    """Write results/<name>.json and return the payload."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": name,
        "when": datetime.datetime.now().isoformat(timespec="seconds"),
        "code_version": code_version,
        "params": params,
        "metrics": metrics,
        "notes": notes,
    }
    (RESULTS / f"{name}.json").write_text(json.dumps(payload, indent=2))
    print(f"recorded {name}")
    return payload