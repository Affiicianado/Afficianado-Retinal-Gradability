"""Rebuild a results table from results/*.json, so nobody copies numbers by hand.

    python -m src.eval.summarise            # from the repository root
"""
import json
import sys
from pathlib import Path

import pandas as pd


def summarise(results_dir="results"):
    rows = []
    for f in sorted(Path(results_dir).glob("*.json")):
        d = json.loads(f.read_text())
        m = d.get("metrics", {})
        rows.append({"name": d.get("name", f.stem),
                     "when": d.get("when", ""),
                     "accuracy": m.get("accuracy"),
                     "macro_f1": m.get("macro_f1"),
                     "kappa": m.get("kappa"),
                     "notes": (d.get("notes") or "")[:60]})
    return pd.DataFrame(rows).sort_values("when").reset_index(drop=True)


if __name__ == "__main__":
    table = summarise(sys.argv[1] if len(sys.argv) > 1 else "results")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
