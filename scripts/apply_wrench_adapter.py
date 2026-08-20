"""Apply a saved wrench adapter to a six-column CSV stream.

This is useful for offline diagnostics.  The same ``load_adapter(...).transform_numpy``
call can be placed immediately before a simulator WebSocket client sends the
``effort`` field; the TAVLA server should not adapt the value a second time.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from openpi.shared.wrench_adapter import load_adapter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    with args.input_csv.open(newline="") as stream:
        rows = list(csv.reader(stream))
    if not rows:
        raise ValueError(f"Empty CSV: {args.input_csv}")
    header = rows[0]
    values = np.asarray([[float(value) for value in row] for row in rows[1:] if row], dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError(f"Expected six numeric columns after header, got {values.shape}")
    adapter = load_adapter(args.adapter)
    output = adapter.transform_numpy(values)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(output.tolist())
    print(f"wrote {len(output)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
