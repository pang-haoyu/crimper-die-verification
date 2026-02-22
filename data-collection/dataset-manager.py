#!/usr/bin/env python3

"""
Smart Die Validation - Dataset Manager
Python 3.12 compatible.

Features:
- Imports images from an incoming folder into a structured dataset tree.
- Enforces per-session/per-die/per-variation quotas from a YAML config.
- Deduplicates by content hash.
- Writes an index (CSV + JSONL) with metadata for downstream training.

Usage examples:
  python dataset_manager.py status --config config/plan.yaml --dataset dataset
  python dataset_manager.py import --config config/plan.yaml --dataset dataset --incoming incoming \
      --split train --session s1 --die 17 --variation baseline
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image  # pip install pillow
import yaml  # pip install pyyaml


ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class ImportSpec:
    split: str
    session: str
    die: int
    variation: str


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_image(path: Path) -> Tuple[int, int, str]:
    """
    Verifies the file is a readable image and returns (width, height, format).
    """
    with Image.open(path) as im:
        im.verify()
    with Image.open(path) as im:
        width, height = im.size
        fmt = (im.format or "").lower()
    return width, height, fmt


def load_plan(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_index_files(dataset_root: Path) -> Tuple[Path, Path]:
    csv_path = dataset_root / "index.csv"
    jsonl_path = dataset_root / "index.jsonl"

    if not csv_path.exists():
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "id",
                "split",
                "session",
                "die_size",
                "variation",
                "rel_path",
                "sha256",
                "width",
                "height",
                "format",
                "imported_utc",
                "source_filename",
            ])

    if not jsonl_path.exists():
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_path.write_text("", encoding="utf-8")

    return csv_path, jsonl_path


def read_index_csv(csv_path: Path) -> List[dict]:
    if not csv_path.exists():
        return []
    rows = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def current_counts(index_rows: List[dict]) -> Dict[Tuple[str, str, int, str], int]:
    """
    Key: (split, session, die_size, variation) -> count
    """
    counts: Dict[Tuple[str, str, int, str], int] = {}
    for row in index_rows:
        key = (
            row["split"],
            row["session"],
            int(row["die_size"]),
            row["variation"],
        )
        counts[key] = counts.get(key, 0) + 1
    return counts


def existing_hashes(index_rows: List[dict]) -> set[str]:
    return {row["sha256"] for row in index_rows if row.get("sha256")}


def plan_quota(plan: dict, spec: ImportSpec) -> int:
    try:
        sess = plan["splits"][spec.split]["sessions"][spec.session]
        die_sizes = sess["die_sizes"]
        if spec.die not in die_sizes:
            raise KeyError(f"Die {spec.die} not in die_sizes for {spec.split}/{spec.session}")
        return int(sess["per_die"][spec.variation])
    except KeyError as e:
        raise SystemExit(f"Invalid plan reference: {e}")


def target_dir(dataset_root: Path, spec: ImportSpec) -> Path:
    return dataset_root / spec.split / spec.session / f"die_{spec.die}" / spec.variation


def next_image_id(index_rows: List[dict]) -> int:
    """
    Produces a monotonically increasing integer ID.
    """
    if not index_rows:
        return 1
    return max(int(r["id"]) for r in index_rows) + 1


def import_images(config: Path, dataset_root: Path, incoming: Path, spec: ImportSpec) -> None:
    plan = load_plan(config)
    csv_path, jsonl_path = ensure_index_files(dataset_root)
    index_rows = read_index_csv(csv_path)

    quota = plan_quota(plan, spec)
    counts = current_counts(index_rows)
    key = (spec.split, spec.session, spec.die, spec.variation)
    already = counts.get(key, 0)

    if already >= quota:
        raise SystemExit(
            f"Quota reached for {spec.split}/{spec.session}/die_{spec.die}/{spec.variation}: "
            f"{already}/{quota} images already imported."
        )

    # Gather incoming images
    files = [p for p in sorted(incoming.iterdir()) if p.is_file() and p.suffix.lower() in ALLOWED_EXTS]
    if not files:
        raise SystemExit(f"No image files found in {incoming} (allowed: {sorted(ALLOWED_EXTS)})")

    # Only import up to remaining quota
    remaining = quota - already
    files = files[:remaining]

    # Prepare
    out_dir = target_dir(dataset_root, spec)
    out_dir.mkdir(parents=True, exist_ok=True)

    hashes = existing_hashes(index_rows)
    next_id = next_image_id(index_rows)

    imported_any = False

    for src in files:
        # Validate image readability and hash
        w, h, fmt = verify_image(src)
        digest = sha256_file(src)

        if digest in hashes:
            print(f"SKIP duplicate: {src.name}")
            # Move to a duplicates bucket so the incoming folder stays clean
            dup_dir = dataset_root / "_duplicates"
            dup_dir.mkdir(parents=True, exist_ok=True)
            src.rename(dup_dir / src.name)
            continue

        # Name: <split>_<session>_die<die>_<variation>_<id>_<sha8>.<ext>
        sha8 = digest[:8]
        ext = src.suffix.lower()
        filename = f"{spec.split}_{spec.session}_die{spec.die}_{spec.variation}_{next_id:06d}_{sha8}{ext}"
        dst = out_dir / filename

        # Move file into dataset
        src.rename(dst)

        rel_path = dst.relative_to(dataset_root).as_posix()
        imported_utc = datetime.now(timezone.utc).isoformat()

        # Append to CSV + JSONL
        row = {
            "id": str(next_id),
            "split": spec.split,
            "session": spec.session,
            "die_size": str(spec.die),
            "variation": spec.variation,
            "rel_path": rel_path,
            "sha256": digest,
            "width": str(w),
            "height": str(h),
            "format": fmt,
            "imported_utc": imported_utc,
            "source_filename": src.name,
        }

        with csv_path.open("a", newline="", encoding="utf-8") as f:
            wcsv = csv.DictWriter(f, fieldnames=list(row.keys()))
            wcsv.writerow(row)

        with jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        index_rows.append(row)
        hashes.add(digest)
        next_id += 1
        imported_any = True

        print(f"IMPORTED: {rel_path}")

    if not imported_any:
        print("No new images imported (all were duplicates or quota already met).")


def status(config: Path, dataset_root: Path, split: str, session: str | None) -> None:
    plan = load_plan(config)
    csv_path, _ = ensure_index_files(dataset_root)
    index_rows = read_index_csv(csv_path)
    counts = current_counts(index_rows)

    sessions = plan["splits"][split]["sessions"]
    for s_name, s_cfg in sessions.items():
        if session and s_name != session:
            continue

        die_sizes = s_cfg["die_sizes"]
        per_die = s_cfg["per_die"]

        print(f"\n=== {split}/{s_name} ===")
        for die in die_sizes:
            parts = []
            ok = True
            for var, quota in per_die.items():
                c = counts.get((split, s_name, int(die), var), 0)
                parts.append(f"{var}:{c}/{quota}")
                if c != int(quota):
                    ok = False
            flag = "OK" if ok else "INCOMPLETE"
            print(f"die_{die}: {flag} | " + " | ".join(parts))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Smart Die Validation dataset manager")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_status = sub.add_parser("status", help="Show progress vs plan quotas")
    p_status.add_argument("--config", required=True, type=Path)
    p_status.add_argument("--dataset", required=True, type=Path)
    p_status.add_argument("--split", required=True, choices=["train", "val", "test"])
    p_status.add_argument("--session", required=False)

    p_import = sub.add_parser("import", help="Import images from incoming folder into dataset")
    p_import.add_argument("--config", required=True, type=Path)
    p_import.add_argument("--dataset", required=True, type=Path)
    p_import.add_argument("--incoming", required=True, type=Path)
    p_import.add_argument("--split", required=True, choices=["train", "val", "test"])
    p_import.add_argument("--session", required=True)
    p_import.add_argument("--die", required=True, type=int)
    p_import.add_argument("--variation", required=True, choices=["baseline", "angle", "lighting", "clean_oily", "rotation"])

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "status":
        status(args.config, args.dataset, args.split, args.session)
        return

    if args.cmd == "import":
        spec = ImportSpec(
            split=args.split,
            session=args.session,
            die=args.die,
            variation=args.variation,
        )
        import_images(args.config, args.dataset, args.incoming, spec)
        return


if __name__ == "__main__":
    main()

