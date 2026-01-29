#!/usr/bin/env python3

"""
Smart Die Verification (Metric Learning via ArcFace-style Training)
CPU-friendly training script for Ubuntu 24.04 / Python 3.12 / PyTorch (CPU).

What this script does
---------------------
1) Loads ImageFolder datasets (train + val).
2) Trains an embedding model:
   - Backbone: EfficientNet-B0 (ImageNet pretrained via timm)
   - Head: projection to embedding_dim, L2-normalized
   - Loss: ArcFace-style (angular margin softmax) with class weights
3) Computes class "reference embeddings" as centroids of TRAIN embeddings.
4) Evaluates verification metrics on VAL:
   - Uses cosine similarity to centroids
   - Reports TAR@FAR for:
       (a) all negatives (optimistic baseline)
       (b) hard negatives (top-K wrong centroids per sample)
       (c) adjacent negatives (wrong centroids from adjacent die sizes only), if parsable
   - Also reports nearest-centroid accuracy as a diagnostic
5) Tracks experiments with MLflow (local).
6) Saves:
   - best model checkpoint (by TAR@FAR=checkpoint_far on hard negatives by default)
   - final centroids, thresholds, and metadata

Dependencies
------------
pip install torch torchvision timm mlflow numpy scikit-learn

Directory layout (ImageFolder)
------------------------------
data_root/
  train/
    die_17/
      *.jpg
    die_20/
      *.jpg
    ...
  val/
    die_17/
      *.jpg
    die_20/
      *.jpg
    ...

Notes
-----
- ArcFace-style training with class weights (classification-like loss),
  but deployment is verification via centroids + threshold.
- Designed for CPU (no CUDA assumptions), but fixes are device-correct anyway.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
from torchvision import datasets, transforms

import timm


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def imagenet_normalize():
    # Standard ImageNet normalization for pretrained backbones
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    return mean, std


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=1, keepdim=True).clamp_min(eps))


def safe_quantile(x: np.ndarray, q: float) -> float:
    """
    Numpy quantile API changed slightly across versions. This wrapper keeps things robust.
    """
    try:
        return float(np.quantile(x, q, method="linear"))
    except TypeError:
        # Older numpy
        return float(np.quantile(x, q, interpolation="linear"))


# -----------------------------
# Balanced batch sampler
# -----------------------------
class MPerClassSampler(Sampler[List[int]]):
    """
    Creates balanced batches: M classes per batch, K samples per class.
    Batch size = M * K.

    Works with torchvision.datasets.ImageFolder (dataset.targets provides labels).
    """

    def __init__(self, labels: List[int], m: int, k: int, batches_per_epoch: int, seed: int = 0):
        self.labels = np.array(labels, dtype=np.int64)
        self.m = int(m)
        self.k = int(k)
        self.batch_size = self.m * self.k
        self.batches_per_epoch = int(batches_per_epoch)
        self.rng = np.random.default_rng(seed)

        self.unique_labels = np.unique(self.labels)
        self.idxs_by_label: Dict[int, np.ndarray] = {}
        for y in self.unique_labels:
            self.idxs_by_label[int(y)] = np.where(self.labels == y)[0]

        if self.m > len(self.unique_labels):
            raise ValueError(f"MPerClassSampler: m={self.m} > num_classes={len(self.unique_labels)}")

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self):
        for _ in range(self.batches_per_epoch):
            chosen = self.rng.choice(self.unique_labels, size=self.m, replace=False)
            batch_indices: List[int] = []
            for y in chosen:
                pool = self.idxs_by_label[int(y)]
                replace = len(pool) < self.k
                picked = self.rng.choice(pool, size=self.k, replace=replace)
                batch_indices.extend(picked.tolist())
            yield batch_indices


# -----------------------------
# Model: EfficientNet-B0 encoder + embedding head
# -----------------------------
class EfficientNetEmbedding(nn.Module):
    def __init__(self, backbone_name: str, embedding_dim: int, pretrained: bool = True):
        super().__init__()
        # num_classes=0 => return features, not logits
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
        emb = l2_normalize(emb)
        return emb


# -----------------------------
# ArcFace-style classification head (no acos)
# -----------------------------
class ArcFaceClassifier(nn.Module):
    """
    ArcFace-style classifier:
    - Maintains learnable weight matrix W (num_classes x embedding_dim)
    - Normalizes embeddings and weights
    - Computes cosine logits
    - Applies additive angular margin to target class only
    - Multiplies by scale s

    Uses the stable identity:
      cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
      sin(theta) = sqrt(1 - cos^2(theta))
    """

    def __init__(self, embedding_dim: int, num_classes: int, margin_m: float = 0.30, scale_s: float = 30.0):
        super().__init__()
        self.num_classes = int(num_classes)
        self.embedding_dim = int(embedding_dim)
        self.m = float(margin_m)
        self.s = float(scale_s)

        self.W = nn.Parameter(torch.randn(self.num_classes, self.embedding_dim))
        nn.init.xavier_uniform_(self.W)

        # Precompute trig constants (registered as buffers for device moves)
        self.register_buffer("cos_m", torch.tensor(np.cos(self.m), dtype=torch.float32))
        self.register_buffer("sin_m", torch.tensor(np.sin(self.m), dtype=torch.float32))

    def forward(self, emb: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Returns scaled logits with ArcFace margin applied to the true class.
        emb: (B, D), assumed L2-normalized
        y:   (B,)
        """
        # Normalize weights
        Wn = l2_normalize(self.W)

        # Cosine similarity (B, C)
        cos_theta = (emb @ Wn.t()).clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        # cos(theta+m) using identity
        sin_theta = torch.sqrt((1.0 - cos_theta * cos_theta).clamp_min(0.0))
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m

        # Apply margin only to target class
        one_hot = F.one_hot(y, num_classes=self.num_classes).type_as(cos_theta)
        logits = cos_theta * (1.0 - one_hot) + cos_theta_m * one_hot

        # Scale
        logits = logits * self.s
        return logits


# -----------------------------
# Evaluation (centroids + verification metrics)
# -----------------------------
@dataclass
class VerificationMetrics:
    # TAR@FAR computed using hard negatives (top-K wrong centroids)
    tar_at_far_hard: Dict[float, float]
    thr_at_far_hard: Dict[float, float]

    # TAR@FAR computed using all wrong centroids (optimistic baseline)
    tar_at_far_all: Dict[float, float]
    thr_at_far_all: Dict[float, float]

    # TAR@FAR computed using adjacent-only wrong centroids (domain-relevant)
    # If adjacency cannot be inferred, dicts will be empty.
    tar_at_far_adj: Dict[float, float]
    thr_at_far_adj: Dict[float, float]

    # Optional diagnostic
    nearest_centroid_acc: float


def infer_adjacent_class_indices(class_names: List[str]) -> Optional[Dict[int, List[int]]]:
    """
    Tries to infer adjacency based on an integer embedded in class names, e.g.:
      die_17, die_20, die_23  -> adjacency by sorted numeric value
    Returns mapping: class_idx -> [adjacent_class_idxs]
    If parsing fails (any class has no integer), returns None.
    """
    nums: List[int] = []
    for name in class_names:
        # Extract last integer group anywhere in the name (robust for die_17, 17, die-17, etc.)
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
        neighbors: List[int] = []
        if pos - 1 >= 0:
            neighbors.append(int(order[pos - 1]))
        if pos + 1 < len(order):
            neighbors.append(int(order[pos + 1]))
        adj[cls_idx] = neighbors

    return adj


def compute_centroids(
    model: nn.Module,
    loader: DataLoader,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute per-class centroid from TRAIN embeddings.
    Returns:
      centroids: (C, D), L2-normalized
    """
    model.eval()

    sums: Optional[torch.Tensor] = None
    counts = torch.zeros(num_classes, dtype=torch.long, device=device)  # FIX: correct device

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            emb = model(x)  # (B, D), normalized
            if sums is None:
                D = emb.size(1)
                sums = torch.zeros(num_classes, D, dtype=torch.float32, device=device)

            # Vectorized accumulation
            sums.index_add_(0, y, emb)
            ones = torch.ones_like(y, dtype=torch.long)
            counts.index_add_(0, y, ones)

    if sums is None:
        raise RuntimeError("Centroid computation saw no batches; check your loader/dataset.")

    if (counts == 0).any():
        missing = torch.where(counts == 0)[0].tolist()
        raise RuntimeError(f"No samples found for class indices {missing} in centroid computation loader.")

    centroids = sums / counts.unsqueeze(1).float()
    centroids = l2_normalize(centroids)
    return centroids


def tar_at_far(pos_scores: np.ndarray, neg_scores: np.ndarray, far: float) -> Tuple[float, float]:
    """
    Computes TAR@FAR by choosing a threshold tau such that FAR <= target.
    Accept if score >= tau.

    tau = (1 - FAR) quantile of negative scores.
    """
    if neg_scores.size == 0:
        raise ValueError("neg_scores is empty")
    if pos_scores.size == 0:
        raise ValueError("pos_scores is empty")

    tau = safe_quantile(neg_scores, 1.0 - far)
    tar = float(np.mean(pos_scores >= tau))
    return tar, float(tau)


def evaluate_verification(
    model: nn.Module,
    train_for_centroids: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    class_names: List[str],
    device: torch.device,
    far_targets: List[float],
    hard_topk: int = 2,
) -> VerificationMetrics:
    """
    1) Compute centroids from TRAIN embeddings
    2) For each VAL image, compute similarities to all centroids
    3) Build positive and negative score lists:
       - positives: similarity to true centroid
       - negatives_all: similarities to all wrong centroids
       - negatives_hard: top-K similarities among wrong centroids (hard negatives)
       - negatives_adj: similarities to adjacent wrong centroids (domain hard negatives)
    4) Compute TAR@FAR for far_targets
    """
    centroids = compute_centroids(model, train_for_centroids, num_classes, device)
    adj_map = infer_adjacent_class_indices(class_names)

    model.eval()
    pos_scores: List[np.ndarray] = []
    neg_scores_all: List[np.ndarray] = []
    neg_scores_hard: List[np.ndarray] = []
    neg_scores_adj: List[np.ndarray] = []

    correct_nn = 0
    total = 0

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)

            emb = model(x)  # (B, D)
            sims = emb @ centroids.t()  # (B, C)

            # nearest-centroid diagnostic accuracy
            preds = sims.argmax(dim=1)
            correct_nn += int((preds == y).sum().item())
            total += int(y.numel())

            # positives
            pos = sims.gather(1, y.view(-1, 1)).squeeze(1)  # (B,)
            pos_scores.append(pos.cpu().numpy())

            # negatives (all wrong)
            sims_masked = sims.clone()
            sims_masked.scatter_(1, y.view(-1, 1), float("-inf"))

            neg_all = sims_masked[torch.isfinite(sims_masked)].view(-1)
            neg_scores_all.append(neg_all.cpu().numpy())

            # hard negatives: top-K per sample
            if hard_topk > 0:
                k = min(hard_topk, num_classes - 1)
                topk_vals, _ = torch.topk(sims_masked, k=k, dim=1)
                neg_scores_hard.append(topk_vals.reshape(-1).cpu().numpy())

            # adjacent-only negatives
            if adj_map is not None:
                # For each sample i, take sims to adj classes of y[i]
                # (B, <=2) list -> flatten across batch
                vals: List[torch.Tensor] = []
                for i in range(y.size(0)):
                    yi = int(y[i].item())
                    adj_idxs = adj_map.get(yi, [])
                    if len(adj_idxs) == 0:
                        continue
                    vals.append(sims[i, torch.tensor(adj_idxs, device=device)])
                if vals:
                    neg_scores_adj.append(torch.cat(vals, dim=0).cpu().numpy())

    pos_scores_np = np.concatenate(pos_scores, axis=0)
    neg_all_np = np.concatenate(neg_scores_all, axis=0)
    neg_hard_np = np.concatenate(neg_scores_hard, axis=0) if neg_scores_hard else np.array([], dtype=np.float32)
    neg_adj_np = np.concatenate(neg_scores_adj, axis=0) if neg_scores_adj else np.array([], dtype=np.float32)

    tar_hard: Dict[float, float] = {}
    thr_hard: Dict[float, float] = {}
    tar_all: Dict[float, float] = {}
    thr_all: Dict[float, float] = {}
    tar_adj: Dict[float, float] = {}
    thr_adj: Dict[float, float] = {}

    for far in far_targets:
        tar, thr = tar_at_far(pos_scores_np, neg_all_np, far=far)
        tar_all[far] = tar
        thr_all[far] = thr

        if neg_hard_np.size > 0:
            tar, thr = tar_at_far(pos_scores_np, neg_hard_np, far=far)
            tar_hard[far] = tar
            thr_hard[far] = thr

        if neg_adj_np.size > 0:
            tar, thr = tar_at_far(pos_scores_np, neg_adj_np, far=far)
            tar_adj[far] = tar
            thr_adj[far] = thr

    nn_acc = float(correct_nn / max(total, 1))

    return VerificationMetrics(
        tar_at_far_hard=tar_hard,
        thr_at_far_hard=thr_hard,
        tar_at_far_all=tar_all,
        thr_at_far_all=thr_all,
        tar_at_far_adj=tar_adj,
        thr_at_far_adj=thr_adj,
        nearest_centroid_acc=nn_acc,
    )


# -----------------------------
# Training helpers
# -----------------------------
def freeze_backbone(model: EfficientNetEmbedding, freeze: bool) -> None:
    for p in model.backbone.parameters():
        p.requires_grad = not freeze


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(
    model: EfficientNetEmbedding,
    arcface: ArcFaceClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    class_weights: Optional[torch.Tensor],
) -> float:
    model.train()
    arcface.train()

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    running = 0.0
    n = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)

        emb = model(x)
        logits = arcface(emb, y)
        loss = loss_fn(logits, y)

        loss.backward()
        optimizer.step()

        running += float(loss.item()) * y.size(0)
        n += int(y.size(0))

    return running / max(n, 1)


# -----------------------------
# Main
# -----------------------------
def build_transforms(input_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    mean, std = imagenet_normalize()

    # Geometry-safe, conservative augmentations (recommended for subtle die differences)
    train_tfms = transforms.Compose([
        transforms.Resize(int(input_size * 1.15)),  # e.g. 224 -> 257
        transforms.RandomResizedCrop(
            size=input_size,
            scale=(0.90, 1.00),      # small scale jitter only
            ratio=(0.98, 1.02),      # keep aspect ratio stable
        ),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.10, contrast=0.10),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    val_tfms = transforms.Compose([
        transforms.Resize(int(input_size * 1.15)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return train_tfms, val_tfms


def main() -> None:
    ap = argparse.ArgumentParser(description="Train Smart Die verification embedding model (EfficientNet-B0 + ArcFace).")

    ap.add_argument("--data-root", type=Path, required=True, help="Root containing train/ and val/ ImageFolder splits.")
    ap.add_argument("--outdir", type=Path, default=Path("runs/smart_die"), help="Output directory for checkpoints/artifacts.")
    ap.add_argument("--run-name", type=str, default="exp", help="MLflow run name.")
    ap.add_argument("--mlflow-uri", type=str, default="", help="Optional MLflow tracking URI (default: local ./mlruns).")

    ap.add_argument("--backbone", type=str, default="efficientnet_b0", help="timm backbone name.")
    ap.add_argument("--embedding-dim", type=int, default=256)
    ap.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained weights for backbone.")

    ap.add_argument("--input-size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--freeze-epochs", type=int, default=3, help="Epochs to freeze backbone at start.")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)

    ap.add_argument("--arcface-margin", type=float, default=0.30)
    ap.add_argument("--arcface-scale", type=float, default=30.0)

    ap.add_argument("--batch-size", type=int, default=32, help="If using balanced sampler, this is ignored.")
    ap.add_argument("--num-workers", type=int, default=4)

    ap.add_argument("--use-balanced-sampler", action="store_true", help="Use M-per-class balanced batches.")
    ap.add_argument("--m-classes", type=int, default=8, help="M classes per batch for balanced sampler.")
    ap.add_argument("--k-samples", type=int, default=4, help="K samples per class for balanced sampler.")
    ap.add_argument("--batches-per-epoch", type=int, default=60, help="Batches per epoch for balanced sampler.")

    ap.add_argument("--eval-every", type=int, default=5, help="Run centroid verification eval every N epochs.")
    ap.add_argument("--hard-topk", type=int, default=2, help="Top-K wrong centroids used as hard negatives.")
    ap.add_argument("--checkpoint-far", type=float, default=0.05, help="FAR target used to select best checkpoint.")

    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    set_seed(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)

    # MLflow setup
    if args.mlflow_uri:
        mlflow.set_tracking_uri(args.mlflow_uri)

    mlflow.set_experiment("smart-die-verification")

    # Data
    train_dir = args.data_root / "train"
    val_dir = args.data_root / "val"

    if not train_dir.exists():
        raise SystemExit(f"Missing train directory: {train_dir}")
    if not val_dir.exists():
        raise SystemExit(f"Missing val directory: {val_dir}")

    train_tfms, val_tfms = build_transforms(args.input_size)

    train_ds = datasets.ImageFolder(root=str(train_dir), transform=train_tfms)
    val_ds = datasets.ImageFolder(root=str(val_dir), transform=val_tfms)

    class_names = train_ds.classes
    num_classes = len(class_names)

    # Basic sanity: same classes in train and val
    if val_ds.classes != train_ds.classes:
        raise SystemExit(
            "Train and val classes differ. Ensure ImageFolder class folders match exactly.\n"
            f"train classes: {train_ds.classes}\n"
            f"val classes:   {val_ds.classes}"
        )

    # Class weights (helps if there is slight imbalance)
    targets = np.array(train_ds.targets, dtype=np.int64)
    counts = np.bincount(targets, minlength=num_classes).astype(np.float32)
    weights = (counts.sum() / np.maximum(counts, 1.0))
    weights = weights / weights.mean()  # normalize around 1
    class_weights = torch.tensor(weights, dtype=torch.float32)

    # DataLoaders
    if args.use_balanced_sampler:
        sampler = MPerClassSampler(
            labels=train_ds.targets,
            m=args.m_classes,
            k=args.k_samples,
            batches_per_epoch=args.batches_per_epoch,
            seed=args.seed,
        )
        train_loader = DataLoader(
            train_ds,
            batch_sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=False,
        )
        effective_batch_size = args.m_classes * args.k_samples
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=False,
        )
        effective_batch_size = args.batch_size

    # For centroid computation we do NOT want training-time augmentations; use deterministic val_tfms
    train_ds_for_centroids = datasets.ImageFolder(root=str(train_dir), transform=val_tfms)
    train_centroid_loader = DataLoader(
        train_ds_for_centroids,
        batch_size=64,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=64,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    device = torch.device("cpu")

    # Model
    model = EfficientNetEmbedding(
        backbone_name=args.backbone,
        embedding_dim=args.embedding_dim,
        pretrained=args.pretrained,
    ).to(device)

    arcface = ArcFaceClassifier(
        embedding_dim=args.embedding_dim,
        num_classes=num_classes,
        margin_m=args.arcface_margin,
        scale_s=args.arcface_scale,
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(arcface.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    far_targets = [0.10, 0.05, 0.01]
    checkpoint_far = float(args.checkpoint_far)
    if checkpoint_far not in far_targets:
        far_targets = sorted(set(far_targets + [checkpoint_far]))

    best_score = -1.0
    best_epoch = -1

    run_meta = {
        "data_root": str(args.data_root.resolve()),
        "classes": class_names,
        "num_classes": num_classes,
        "backbone": args.backbone,
        "pretrained": bool(args.pretrained),
        "embedding_dim": args.embedding_dim,
        "input_size": args.input_size,
        "arcface_margin": args.arcface_margin,
        "arcface_scale": args.arcface_scale,
        "seed": args.seed,
    }

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params({
            "backbone": args.backbone,
            "pretrained": args.pretrained,
            "embedding_dim": args.embedding_dim,
            "input_size": args.input_size,
            "epochs": args.epochs,
            "freeze_epochs": args.freeze_epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "arcface_margin": args.arcface_margin,
            "arcface_scale": args.arcface_scale,
            "use_balanced_sampler": args.use_balanced_sampler,
            "effective_batch_size": effective_batch_size,
            "m_classes": args.m_classes,
            "k_samples": args.k_samples,
            "batches_per_epoch": args.batches_per_epoch,
            "eval_every": args.eval_every,
            "hard_topk": args.hard_topk,
            "checkpoint_far": checkpoint_far,
        })

        classes_path = args.outdir / "classes.json"
        classes_path.write_text(json.dumps(class_names, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(classes_path))

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()

            do_freeze = epoch <= args.freeze_epochs
            freeze_backbone(model, freeze=do_freeze)

            cw = class_weights.to(device)

            train_loss = train_one_epoch(
                model=model,
                arcface=arcface,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                class_weights=cw,
            )

            scheduler.step()
            lr_now = float(scheduler.get_last_lr()[0])

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("lr", lr_now, step=epoch)
            mlflow.log_metric(
                "trainable_params",
                count_trainable_params(model) + count_trainable_params(arcface),
                step=epoch,
            )

            do_eval = (epoch % args.eval_every == 0) or (epoch == args.epochs)
            if do_eval:
                metrics = evaluate_verification(
                    model=model,
                    train_for_centroids=train_centroid_loader,
                    val_loader=val_loader,
                    num_classes=num_classes,
                    class_names=class_names,
                    device=device,
                    far_targets=far_targets,
                    hard_topk=args.hard_topk,
                )

                mlflow.log_metric("val_nearest_centroid_acc", metrics.nearest_centroid_acc, step=epoch)

                # Hard-negative TAR@FAR (realistic)
                for far, tar in metrics.tar_at_far_hard.items():
                    mlflow.log_metric(f"val_TAR@FAR_hard_{far:.3f}", tar, step=epoch)
                for far, thr in metrics.thr_at_far_hard.items():
                    mlflow.log_metric(f"val_thr@FAR_hard_{far:.3f}", thr, step=epoch)

                # All-negative TAR@FAR (optimistic baseline)
                for far, tar in metrics.tar_at_far_all.items():
                    mlflow.log_metric(f"val_TAR@FAR_all_{far:.3f}", tar, step=epoch)
                for far, thr in metrics.thr_at_far_all.items():
                    mlflow.log_metric(f"val_thr@FAR_all_{far:.3f}", thr, step=epoch)

                # Adjacent-only TAR@FAR (domain-relevant, if adjacency inferred)
                for far, tar in metrics.tar_at_far_adj.items():
                    mlflow.log_metric(f"val_TAR@FAR_adj_{far:.3f}", tar, step=epoch)
                for far, thr in metrics.thr_at_far_adj.items():
                    mlflow.log_metric(f"val_thr@FAR_adj_{far:.3f}", thr, step=epoch)

                # Choose checkpoint score: prefer hard; else fall back to all
                if checkpoint_far in metrics.tar_at_far_hard:
                    ckpt_score = metrics.tar_at_far_hard[checkpoint_far]
                    ckpt_thr = metrics.thr_at_far_hard[checkpoint_far]
                    which = "hard"
                else:
                    ckpt_score = metrics.tar_at_far_all[checkpoint_far]
                    ckpt_thr = metrics.thr_at_far_all[checkpoint_far]
                    which = "all"

                mlflow.log_metric(f"checkpoint_score_TAR@FAR_{which}_{checkpoint_far:.3f}", ckpt_score, step=epoch)

                if ckpt_score > best_score:
                    best_score = ckpt_score
                    best_epoch = epoch

                    ckpt = {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "arcface_state": arcface.state_dict(),
                        "best_score": best_score,
                        "best_epoch": best_epoch,
                        "checkpoint_far": checkpoint_far,
                        "checkpoint_threshold": float(ckpt_thr),
                        "classes": class_names,
                        "meta": run_meta,
                    }
                    best_path = args.outdir / "best.pt"
                    torch.save(ckpt, best_path)
                    mlflow.log_artifact(str(best_path))

            dt = time.time() - t0
            mlflow.log_metric("epoch_time_sec", dt, step=epoch)
            print(
                f"[Epoch {epoch:03d}/{args.epochs}] "
                f"loss={train_loss:.4f} lr={lr_now:.2e} freeze={'Y' if do_freeze else 'N'} "
                f"time={dt:.1f}s {'(eval)' if do_eval else ''}"
            )

        # Final export artifacts: reload best checkpoint, recompute centroids + metrics, save bundle
        best_path = args.outdir / "best.pt"
        if best_path.exists():
            ckpt = torch.load(best_path, map_location="cpu")
            model.load_state_dict(ckpt["model_state"])
            arcface.load_state_dict(ckpt["arcface_state"])

        final_metrics = evaluate_verification(
            model=model,
            train_for_centroids=train_centroid_loader,
            val_loader=val_loader,
            num_classes=num_classes,
            class_names=class_names,
            device=device,
            far_targets=far_targets,
            hard_topk=args.hard_topk,
        )
        centroids = compute_centroids(model, train_centroid_loader, num_classes, device).cpu().numpy()

        centroids_path = args.outdir / "centroids.npy"
        np.save(centroids_path, centroids)
        mlflow.log_artifact(str(centroids_path))

        thresholds = {
            "hard": {f"{k:.3f}": float(v) for k, v in final_metrics.thr_at_far_hard.items()},
            "all": {f"{k:.3f}": float(v) for k, v in final_metrics.thr_at_far_all.items()},
            "adj": {f"{k:.3f}": float(v) for k, v in final_metrics.thr_at_far_adj.items()},
        }

        export_meta = {
            **run_meta,
            "best_epoch": best_epoch,
            "best_score": best_score,
            "far_targets": far_targets,
            "hard_topk": args.hard_topk,
            "thresholds": thresholds,
            "final_nearest_centroid_acc": final_metrics.nearest_centroid_acc,
            "final_TAR@FAR_hard": {f"{k:.3f}": float(v) for k, v in final_metrics.tar_at_far_hard.items()},
            "final_TAR@FAR_all": {f"{k:.3f}": float(v) for k, v in final_metrics.tar_at_far_all.items()},
            "final_TAR@FAR_adj": {f"{k:.3f}": float(v) for k, v in final_metrics.tar_at_far_adj.items()},
        }

        meta_path = args.outdir / "export_meta.json"
        meta_path.write_text(json.dumps(export_meta, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(meta_path))

        deploy_path = args.outdir / "deploy_model.pt"
        torch.save(
            {
                "model_state": model.state_dict(),
                "classes": class_names,
                "meta": export_meta,
            },
            deploy_path,
        )
        mlflow.log_artifact(str(deploy_path))

        print("\n=== Training complete ===")
        print(f"Best epoch: {best_epoch}")
        print(f"Best TAR@FAR({checkpoint_far:.3f}): {best_score:.4f}")
        print(f"Artifacts saved to: {args.outdir.resolve()}")
        print("Tip: launch MLflow UI with: mlflow ui")


if __name__ == "__main__":
    main()

