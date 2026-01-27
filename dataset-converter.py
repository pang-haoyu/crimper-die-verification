#!/usr/bin/env python3

"""
Convert Smart Die collection dataset to PyTorch ImageFolder structure.

Input (crawled, not assumed rigidly):
  <dataset_root>/<split>/<session>/die_<N>/<variation>/*.(jpg|jpeg|png|webp)

Output:
  <out_root>/<split>/die_<N>/*.jpg

Notes:
- Discovers ALL die classes present (die_<N>) and converts them.
- No remapping. No fixed list of 8 classes.
- Uses hardlinks by default (mode=link) and falls back to copy if linking fails.
- Optional --prefix-with-session to guarantee filename uniqueness across sessions.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple, Set

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
DIE_DIR_RE = re.compile(r"^die_(\d+)$")


@dataclass(frozen=True)
class Stats:
    produced: int = 0
    linked: int = 0
    copied: int = 0
    skipped_existing: int = 0
    skipped_nonimage: int = 0


def iter_images_under_split(dataset_root: Path, split: str) -> Iterable[Tuple[Path, str, int]]:
    """
    Yields tuples: (img_path, session_name, die_size)

    Crawls:
      dataset_root/split/session/die_<N>/variation/*.ext
    """
    split_dir = dataset_root / split
    if not split_dir.exists():
        return

    for session_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        session_name = session_dir.name

        for die_dir in sorted(p for p in session_dir.iterdir() if p.is_dir()):
            m = DIE_DIR_RE.match(die_dir.name)
            if not m:
                continue
            die_size = int(m.group(1))

            # Variation dirs inside die_dir
            for var_dir in sorted(p for p in die_dir.iterdir() if p.is_dir()):
                for img in sorted(var_dir.iterdir()):
                    if not img.is_file():
                        continue
                    if img.suffix.lower() not in IMAGE_EXTS:
                        continue
                    yield img, session_name, die_size


def safe_link_or_copy(src: Path, dst: Path, mode: str) -> str:
    """
    mode:
      - "link": hardlink preferred; fallback to copy2
      - "copy": copy2 only
      - "move": move/rename (destructive)
    Returns one of: "exists", "linked", "copied", "moved"
    """
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        return "exists"

    if mode == "move":
        src.replace(dst)
        return "moved"

    if mode == "copy":
        shutil.copy2(src, dst)
        return "copied"

    # mode == "link"
    try:
        os.link(src, dst)
        return "linked"
    except OSError:
        shutil.copy2(src, dst)
        return "copied"


def build_dst_name(src: Path, session_name: Optional[str], prefix_with_session: bool) -> str:
    if prefix_with_session and session_name:
        return f"{session_name}_{src.name}"
    return src.name


def discover_splits(dataset_root: Path, requested: Optional[list[str]]) -> list[str]:
    """
    If requested is provided (non-empty), use it.
    Otherwise, auto-detect existing split folders under dataset_root (train/val/test/...).
    """
    if requested:
        return requested

    splits: list[str] = []
    for p in sorted(dataset_root.iterdir()):
        if p.is_dir():
            splits.append(p.name)
    return splits


def discover_die_classes(dataset_root: Path, splits: list[str]) -> Set[int]:
    """
    Returns the set of die sizes found under the selected splits.
    """
    found: Set[int] = set()
    for split in splits:
        for img_path, session_name, die_size in iter_images_under_split(dataset_root, split):
            found.add(die_size)
    return found


def convert(dataset_root: Path, out_root: Path, splits: list[str], mode: str, prefix_with_session: bool) -> Stats:
    stats = Stats()

    for split in splits:
        split_dir = dataset_root / split
        if not split_dir.exists():
            print(f"[WARN] Split not found, skipping: {split_dir}")
            continue

        for img_path, session_name, die_size in iter_images_under_split(dataset_root, split):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                stats = Stats(
                    produced=stats.produced,
                    linked=stats.linked,
                    copied=stats.copied,
                    skipped_existing=stats.skipped_existing,
                    skipped_nonimage=stats.skipped_nonimage + 1,
                )
                continue

            die_out_dir = out_root / split / f"die_{die_size}"
            dst_name = build_dst_name(img_path, session_name, prefix_with_session)
            dst_path = die_out_dir / dst_name

            result = safe_link_or_copy(img_path, dst_path, mode=mode)

            if result == "exists":
                stats = Stats(
                    produced=stats.produced,
                    linked=stats.linked,
                    copied=stats.copied,
                    skipped_existing=stats.skipped_existing + 1,
                    skipped_nonimage=stats.skipped_nonimage,
                )
            elif result == "linked":
                stats = Stats(
                    produced=stats.produced + 1,
                    linked=stats.linked + 1,
                    copied=stats.copied,
                    skipped_existing=stats.skipped_existing,
                    skipped_nonimage=stats.skipped_nonimage,
                )
            elif result in ("copied", "moved"):
                stats = Stats(
                    produced=stats.produced + 1,
                    linked=stats.linked,
                    copied=stats.copied + 1,
                    skipped_existing=stats.skipped_existing,
                    skipped_nonimage=stats.skipped_nonimage,
                )

    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert Smart Die dataset into PyTorch ImageFolder structure.")
    ap.add_argument("--dataset-root", type=Path, required=True, help="Root of collection dataset (contains split folders).")
    ap.add_argument("--out-root", type=Path, required=True, help="Output root, e.g. data_die/")
    ap.add_argument(
        "--splits",
        nargs="*",
        default=[],
        help="Splits to convert (e.g. train val test). If omitted, auto-detects subfolders under dataset-root.",
    )
    ap.add_argument(
        "--mode",
        choices=["link", "copy", "move"],
        default="link",
        help="How to materialize output files. 'link' is fastest (hardlinks).",
    )
    ap.add_argument(
        "--prefix-with-session",
        action="store_true",
        help="Prefix output filenames with session name to ensure uniqueness across sessions.",
    )
    args = ap.parse_args()

    dataset_root: Path = args.dataset_root
    out_root: Path = args.out_root
    splits = discover_splits(dataset_root, args.splits)

    die_classes = sorted(discover_die_classes(dataset_root, splits))
    if die_classes:
        print(f"[INFO] Discovered die classes ({len(die_classes)}): {', '.join(f'die_{d}' for d in die_classes)}")
    else:
        print("[WARN] No die classes discovered. Check dataset-root path and structure.")

    stats = convert(
        dataset_root=dataset_root,
        out_root=out_root,
        splits=splits,
        mode=args.mode,
        prefix_with_session=args.prefix_with_session,
    )

    print("\n=== Done ===")
    print(f"Splits converted:  {', '.join(splits)}")
    print(f"Images produced:   {stats.produced}")
    print(f"Linked:           {stats.linked}")
    print(f"Copied/Moved:     {stats.copied}")
    print(f"Skipped existing: {stats.skipped_existing}")
    print(f"Skipped nonimage: {stats.skipped_nonimage}")
    print(f"Output at:        {out_root.resolve()}")


if __name__ == "__main__":
    main()

