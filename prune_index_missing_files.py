#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
from pathlib import Path

def prune(dataset_root: Path) -> None:
    csv_path = dataset_root / "index.csv"
    jsonl_path = dataset_root / "index.jsonl"

    if not csv_path.exists():
        raise SystemExit(f"Missing {csv_path}")

    # Read CSV rows
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    kept = []
    removed = 0
    for r in rows:
        rel = r.get("rel_path", "")
        if not rel:
            removed += 1
            continue
        if (dataset_root / rel).exists():
            kept.append(r)
        else:
            removed += 1

    # Write back CSV (same header order)
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(kept)

    # Rebuild JSONL from kept rows (canonical)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")

    print(f"Pruned index: kept={len(kept)} removed={removed}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, type=Path)
    args = p.parse_args()
    prune(args.dataset)

