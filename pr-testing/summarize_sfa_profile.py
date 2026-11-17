"""Summarize per-export kernel time; summed device time is not wall time."""

import csv
import math
from collections import defaultdict
from pathlib import Path


def summarize(root):
    root = Path(root)
    files = sorted(root.rglob("kernel_details.csv"))
    if not files:
        files = sorted(root.rglob("op_summary*.csv"))
    if not files:
        raise ValueError(
            "no exported kernel_details.csv or op_summary CSV; inspect raw traces"
        )
    exports = []
    indexer = f2 = False
    for path in files:
        kernels = defaultdict(lambda: [0, 0.0])
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            columns = {
                "".join(c.lower() for c in key if c.isalnum()): key
                for key in reader.fieldnames or []
            }
            duration = next(
                (columns[k] for k in ("durationus", "taskdurationus") if k in columns),
                None,
            )
            name = next(
                (
                    columns[k]
                    for k in ("type", "optype", "name", "opname")
                    if k in columns
                ),
                None,
            )
            if duration is None or name is None:
                raise ValueError(f"unsupported kernel columns: {path}")
            for row in reader:
                kind = row[name]
                micros = float(row[duration])
                if micros < 0 or not math.isfinite(micros):
                    raise ValueError(f"invalid kernel duration: {path}")
                kernels[kind][0] += 1
                kernels[kind][1] += micros
                normalized = "".join(c.lower() for c in kind if c.isalnum())
                indexer |= "quantlightningindexer" in normalized
                f2 |= "kvquantsparseflashattention" in normalized
        exports.append(
            {
                "file": str(path.relative_to(root)),
                "device_time_us": sum(v[1] for v in kernels.values()),
                "top_kernels": [
                    {"name": k, "count": v[0], "duration_us": v[1]}
                    for k, v in sorted(kernels.items(), key=lambda kv: -kv[1][1])[:30]
                ],
            }
        )
    return {
        "exports": exports,
        "indexer_int8_observed": indexer,
        "f2_observed": f2,
        "note": "Per-export summed device time, not wall time. Raw traces retain rank metadata.",
    }
