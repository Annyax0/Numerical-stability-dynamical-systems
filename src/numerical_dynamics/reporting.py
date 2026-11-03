"""Deterministic data export and machine-readable experiment metadata."""

import csv
import json
from pathlib import Path

FIELDS = ["experiment", "method", "dt", "t_final", "final_error", "max_error",
          "rms_error", "conservation_error", "runtime_s_median", "nfev", "n_steps"]


def export_csv(rows: list[dict], path: Path, fields: list[str] | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = fields or FIELDS
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)
    return path


def export_json(obj: dict | list, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path
