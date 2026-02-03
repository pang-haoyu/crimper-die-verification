#!/usr/bin/env python3

"""
Evaluate Smart Die Verification model on a TEST ImageFolder split.

Assumptions (consistent with training script):
- Data layout:
    data_root/
      train/ die_17/*.jpg ...
      val/   die_17/*.jpg ...
      test/  die_17/*.jpg ...     <-- this script evaluates this split
- Model is an embedding network (EfficientNet + projection + L2 norm).
- Verification uses cosine similarity to per-class centroids.
- Metrics: nearest-centroid accuracy + TAR@FAR for all/hard/adj negatives.

This script is compatible with the artifacts saved by train-script.py:
- deploy_model.pt (preferred) or best.pt (if you want exact training checkpoint bundle)
- centroids.npy (preferred; centroids computed on train set with deterministic transforms)
- export_meta.json (optional; provides FAR targets, thresholds, etc.)

Ref: training/eval logic in train-script.py. (Centroids + TAR@FAR) :contentReference[oaicite:2]{index=2}
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import timm


# -----------------------------
# Utilities (mirrors train-script.py)
# -----------------------------
def imagenet_normalize():
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    return mean, std


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=1, keepdim=True).clamp_min(eps))


def safe_quantile(x: np.ndarray, q: float) -> float:
    try:
        return float(np.quantile(x, q, method="linear"))
    except TypeError:
        return float(np.quantile(x, q, interpolation="linear"))


def tar_at_far(pos_scores: np.ndarray, neg_scores: np.ndarray, far: float) -> Tuple[float, float]:
    """
    Threshold tau chosen at (1 - FAR) quantile of negative scores.
    Accept if score >= tau.
    """
    if neg_scores.size == 0:
        raise ValueError("neg_scores is empty")
    if pos_scores.size == 0:
        raise ValueError("pos_scores is empty")

    tau = safe_quantile(neg_scores, 1.0 - far)
    tar = float(np.mean(pos_scores >= tau))
    return tar, float(tau)


def infer_adjacent_class_indices(class_names: List[str]) -> Optional[Dict[int, List[int]]]:
    """
    Infer adjacency by sorting the last integer in each class name, e.g. die_17, die_20, ...
    Returns: class_idx -> [adjacent_class_idxs]
    """
    nums: List[int] = []
    for name in class_names:
        digits = "".join([ch if ch.isdigit() else " " for ch in name]).split()
        if not digits:
            return None
        nums.append(int(digits[-1]))

    order = np.argsort(nums)
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))

    adj: Dict[int, List[int]] = {}
    for cls_idx in range(len(class_names)):
        pos = int(inv[cls_idx])
        neigh: List[int] = []
        if pos - 1 >= 0:
            neigh.append(int(order[pos - 1]))
        if pos + 1 < len(order):
            neigh.append(int(order[pos + 1]))
        adj[cls_idx] = neigh
    return adj


def build_val_transform(input_size: int) -> transforms.Compose:
    mean, std = imagenet_normalize()
    return transforms.Compose([
        transforms.Resize(int(input_size * 1.15)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


# -----------------------------
# Model definition (same as train-script.py)
# -----------------------------
class EfficientNetEmbedding(nn.Module):
    def __init__(self, backbone_name: str, embedding_dim: int, pretrained: bool = False):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(feat_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        emb = self.head(feats)
        return l2_normalize(emb)


# -----------------------------
# Metrics container
# -----------------------------
@dataclass
class TestReport:
    split: str
    data_root: str
    num_classes: int
    classes: List[str]
    num_images: int

    nearest_centroid_acc: float
    per_class_acc: Dict[str, float]

    tar_at_far_all: Dict[str, float]
    thr_at_far_all: Dict[str, float]

    tar_at_far_hard: Dict[str, float]
    thr_at_far_hard: Dict[str, float]

    tar_at_far_adj: Dict[str, float]
    thr_at_far_adj: Dict[str, float]


# -----------------------------
# Evaluation
# -----------------------------
@torch.no_grad()
def evaluate_split(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    centroids: torch.Tensor,          # (C, D), normalized
    class_names: List[str],
    far_targets: List[float],
    hard_topk: int = 2,
) -> TestReport:
    model.eval()
    device = next(model.parameters()).device

    num_classes = len(class_names)
    adj_map = infer_adjacent_class_indices(class_names)

    correct_nn = 0
    total = 0

    # per-class counters
    correct_by_cls = np.zeros(num_classes, dtype=np.int64)
    total_by_cls = np.zeros(num_classes, dtype=np.int64)

    pos_scores: List[np.ndarray] = []
    neg_scores_all: List[np.ndarray] = []
    neg_scores_hard: List[np.ndarray] = []
    neg_scores_adj: List[np.ndarray] = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        emb = model(x)                 # (B, D)
        sims = emb @ centroids.t()     # (B, C)

        preds = sims.argmax(dim=1)
        matches = (preds == y)

        correct_nn += int(matches.sum().item())
        total += int(y.numel())

        # per-class
        y_np = y.cpu().numpy()
        matches_np = matches.cpu().numpy().astype(np.int64)
        for cls in range(num_classes):
            mask = (y_np == cls)
            if mask.any():
                total_by_cls[cls] += int(mask.sum())
                correct_by_cls[cls] += int(matches_np[mask].sum())

        # positives: sim to true centroid
        pos = sims.gather(1, y.view(-1, 1)).squeeze(1)
        pos_scores.append(pos.cpu().numpy())

        # negatives (all wrong centroids)
        sims_masked = sims.clone()
        sims_masked.scatter_(1, y.view(-1, 1), float("-inf"))

        neg_all = sims_masked[torch.isfinite(sims_masked)].view(-1)
        neg_scores_all.append(neg_all.cpu().numpy())

        # hard negatives: top-K wrong centroids per sample
        if hard_topk > 0 and num_classes > 1:
            k = min(hard_topk, num_classes - 1)
            topk_vals, _ = torch.topk(sims_masked, k=k, dim=1)
            neg_scores_hard.append(topk_vals.reshape(-1).cpu().numpy())

        # adjacent negatives (domain-relevant)
        if adj_map is not None:
            vals: List[torch.Tensor] = []
            for i in range(y.size(0)):
                yi = int(y[i].item())
                adj_idxs = adj_map.get(yi, [])
                if not adj_idxs:
                    continue
                vals.append(sims[i, torch.tensor(adj_idxs, device=device)])
            if vals:
                neg_scores_adj.append(torch.cat(vals, dim=0).cpu().numpy())

    pos_scores_np = np.concatenate(pos_scores, axis=0) if pos_scores else np.array([], dtype=np.float32)
    neg_all_np = np.concatenate(neg_scores_all, axis=0) if neg_scores_all else np.array([], dtype=np.float32)
    neg_hard_np = np.concatenate(neg_scores_hard, axis=0) if neg_scores_hard else np.array([], dtype=np.float32)
    neg_adj_np = np.concatenate(neg_scores_adj, axis=0) if neg_scores_adj else np.array([], dtype=np.float32)

    tar_all: Dict[str, float] = {}
    thr_all: Dict[str, float] = {}
    tar_hard: Dict[str, float] = {}
    thr_hard: Dict[str, float] = {}
    tar_adj: Dict[str, float] = {}
    thr_adj: Dict[str, float] = {}

    for far in far_targets:
        far_key = f"{far:.3f}"

        tar, thr = tar_at_far(pos_scores_np, neg_all_np, far=far)
        tar_all[far_key] = tar
        thr_all[far_key] = thr

        if neg_hard_np.size > 0:
            tar, thr = tar_at_far(pos_scores_np, neg_hard_np, far=far)
            tar_hard[far_key] = tar
            thr_hard[far_key] = thr

        if neg_adj_np.size > 0:
            tar, thr = tar_at_far(pos_scores_np, neg_adj_np, far=far)
            tar_adj[far_key] = tar
            thr_adj[far_key] = thr

    nn_acc = float(correct_nn / max(total, 1))
    per_class_acc = {
        class_names[i]: float(correct_by_cls[i] / max(total_by_cls[i], 1))
        for i in range(num_classes)
    }

    return TestReport(
        split="test",
        data_root="",
        num_classes=num_classes,
        classes=class_names,
        num_images=int(total),
        nearest_centroid_acc=nn_acc,
        per_class_acc=per_class_acc,
        tar_at_far_all=tar_all,
        thr_at_far_all=thr_all,
        tar_at_far_hard=tar_hard,
        thr_at_far_hard=thr_hard,
        tar_at_far_adj=tar_adj,
        thr_at_far_adj=thr_adj,
    )


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser("Evaluate Smart Die model on test split (centroid verification).")

    ap.add_argument("--data-root", type=Path, required=True, help="Root containing test/ ImageFolder split.")
    ap.add_argument("--split", type=str, default="test", help="Split folder name (default: test).")

    ap.add_argument("--model-pt", type=Path, required=True,
                    help="Path to deploy_model.pt (preferred) or best.pt checkpoint.")
    ap.add_argument("--centroids-npy", type=Path, required=True,
                    help="Path to centroids.npy saved from training run.")
    ap.add_argument("--export-meta", type=Path, default=None,
                    help="Optional export_meta.json to reuse far_targets/hard_topk.")

    ap.add_argument("--backbone", type=str, default="efficientnet_b0",
                    help="Backbone name (should match training).")
    ap.add_argument("--embedding-dim", type=int, default=256,
                    help="Embedding dim (should match training).")
    ap.add_argument("--input-size", type=int, default=224,
                    help="Input size used during validation/test transforms.")

    ap.add_argument("--far", type=float, nargs="*", default=[0.10, 0.05, 0.01],
                    help="FAR targets for TAR@FAR (default: 0.10 0.05 0.01).")
    ap.add_argument("--hard-topk", type=int, default=2,
                    help="Top-K wrong centroids treated as hard negatives.")

    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)

    ap.add_argument("--out-json", type=Path, default=Path("test_report.json"),
                    help="Where to write the JSON report.")

    args = ap.parse_args()

    # If export_meta.json is provided, use it to override FAR targets / hard_topk where present
    if args.export_meta is not None and args.export_meta.exists():
        meta = json.loads(args.export_meta.read_text(encoding="utf-8"))
        if "far_targets" in meta and isinstance(meta["far_targets"], list) and meta["far_targets"]:
            args.far = [float(x) for x in meta["far_targets"]]
        if "hard_topk" in meta:
            args.hard_topk = int(meta["hard_topk"])

    device = torch.device("cpu")

    # Load class list from checkpoint if present
    ckpt = torch.load(args.model_pt, map_location="cpu")
    if isinstance(ckpt, dict) and "classes" in ckpt:
        class_names = list(ckpt["classes"])
    elif isinstance(ckpt, dict) and "meta" in ckpt and "classes" in ckpt["meta"]:
        class_names = list(ckpt["meta"]["classes"])
    else:
        raise SystemExit("Could not find 'classes' in checkpoint. Use deploy_model.pt or best.pt from train script.")

    num_classes = len(class_names)

    # Build model and load weights
    model = EfficientNetEmbedding(
        backbone_name=args.backbone,
        embedding_dim=args.embedding_dim,
        pretrained=False,   # weights loaded from checkpoint
    ).to(device)

    # deploy_model.pt stores: {"model_state": ..., "classes": ..., "meta": ...}
    # best.pt stores: {"model_state": ..., "arcface_state": ..., ...}
    if "model_state" not in ckpt:
        raise SystemExit("Checkpoint missing 'model_state'. Provide deploy_model.pt or best.pt from training run.")
    model.load_state_dict(ckpt["model_state"], strict=True)

    # Load centroids
    centroids_np = np.load(args.centroids_npy)
    if centroids_np.shape[0] != num_classes:
        raise SystemExit(
            f"Centroids class count mismatch: centroids has {centroids_np.shape[0]} classes, "
            f"checkpoint has {num_classes} classes."
        )
    centroids = torch.tensor(centroids_np, dtype=torch.float32, device=device)
    centroids = l2_normalize(centroids)

    # Data
    split_dir = args.data_root / args.split
    if not split_dir.exists():
        raise SystemExit(f"Missing split directory: {split_dir} (expected ImageFolder structure)")

    tfm = build_val_transform(args.input_size)
    ds = datasets.ImageFolder(root=str(split_dir), transform=tfm)

    # Sanity: classes must match training order
    if ds.classes != class_names:
        raise SystemExit(
            "Class folders in test split do not match training class order.\n"
            f"Training classes: {class_names}\n"
            f"Test classes:     {ds.classes}\n"
            "Fix by ensuring identical folder names and ordering across splits."
        )

    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False
    )

    report = evaluate_split(
        model=model,
        loader=loader,
        centroids=centroids,
        class_names=class_names,
        far_targets=[float(x) for x in args.far],
        hard_topk=int(args.hard_topk),
    )
    report.data_root = str(args.data_root.resolve())

    # Print a readable summary
    print("\n=== TEST SET REPORT ===")
    print(f"Data root: {report.data_root}")
    print(f"Num images: {report.num_images}")
    print(f"Nearest-centroid accuracy: {report.nearest_centroid_acc:.4f}")

    print("\nPer-class accuracy:")
    for k, v in report.per_class_acc.items():
        print(f"  {k:>10s}: {v:.4f}")

    def print_tar_block(title: str, tar: Dict[str, float], thr: Dict[str, float]):
        if not tar:
            print(f"\n{title}: (not available)")
            return
        print(f"\n{title}:")
        for far_key in sorted(tar.keys()):
            print(f"  FAR={far_key}  TAR={tar[far_key]:.4f}  thr={thr[far_key]:.4f}")

    print_tar_block("TAR@FAR (all negatives)", report.tar_at_far_all, report.thr_at_far_all)
    print_tar_block("TAR@FAR (hard negatives)", report.tar_at_far_hard, report.thr_at_far_hard)
    print_tar_block("TAR@FAR (adjacent negatives)", report.tar_at_far_adj, report.thr_at_far_adj)

    # Save JSON report
    args.out_json.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    print(f"\nSaved JSON report to: {args.out_json.resolve()}\n")


if __name__ == "__main__":
    main()

