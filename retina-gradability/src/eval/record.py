"""One small JSON file per experiment.

Copied from the notebooks (the two signatures they used are merged here). This
replaced the MLflow tracking planned in the proposal: twenty lines, no server, and
the files sit in results/ where anyone can read them.

Every headline number in the report traces to one of these files, and each file
records when it ran, with which parameters. There is no git commit hash, because
nothing inside a Kaggle session is a git checkout; the `when` field and the Kaggle
notebook version history do that job instead.
"""
import datetime
import json
from pathlib import Path


def record(name, params, metrics, notes="", code_version="", results_dir="results"):
    """Write <results_dir>/<name>.json and return the payload."""
    payload = {"name": name,
               "when": datetime.datetime.now().isoformat(timespec="seconds")}
    if code_version:
        payload["code_version"] = code_version
    payload.update({"params": params, "metrics": metrics, "notes": notes})

    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(payload, indent=2, default=str))
    print(f"recorded {name}")
    return payload
