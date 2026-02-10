#!/usr/bin/env python3

"""
Generate an open-set centroid embedding for a die from a folder of images
using a trained Smart Die verification model (deploy_model.pt).

Example:
  python centroid_from_folder.py \
    --deploy /path/to/outdir/deploy_model.pt \
    --images /data/new_dies/die_24 \
    --out /path/to/outdir/open_centroids/die_24_centroid.npy \
    --device cpu

Notes:
- This is intended for dies NOT in training classes (open-set reference).
- It computes:
    1) per-image embedding = L2-normalized model output
    2) centroid = mean(embeddings)
    3) centroid = L2-normalize(centroid)
- Saves centroid .npy and a .json sidecar with metadata.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import timm
from PIL import Image
from torchvision import transforms


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def imagenet_normalize() -> Tuple[
    Tuple[float, float, float], Tuple[float, float, float]
]:
    return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / torch.clamp(torch.norm(x, dim=-1, keepdim=True), min=eps)


class EfficientNetEmbedding(nn.Module):
    def __init__(
        self, backbone_name: str, embedding_dim: int, pretrained: bool = False
    ):
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


def build_val_transform(input_size: int) -> transforms.Compose:
    mean, std = imagenet_normalize()
    return transforms.Compose(
        [
            transforms.Resize(int(input_size * 1.15)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def list_images(root: Path) -> List[Path]:
    if not root.exists():
        raise SystemExit(f"Images path does not exist: {root}")
    paths: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            paths.append(p)
    return sorted(paths)


def load_deploy_bundle(deploy_path: Path) -> dict:
    if not deploy_path.exists():
        raise SystemExit(f"Missing deploy model: {deploy_path}")
    bundle = torch.load(str(deploy_path), map_location="cpu")
    if not isinstance(bundle, dict) or "model_state" not in bundle:
        raise SystemExit(
            "deploy_model.pt does not look like the expected bundle (missing 'model_state')."
        )
    return bundle


@torch.no_grad()
def embed_images(
    model: nn.Module,
    tfm: transforms.Compose,
    image_paths: List[Path],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    if not image_paths:
        raise SystemExit(
            "No images found (supported extensions: jpg/jpeg/png/webp/bmp/tif/tiff)."
        )

    model.eval()
    all_embs: List[torch.Tensor] = []

    def load_one(path: Path) -> torch.Tensor:
        # Always convert to RGB for consistency
        with Image.open(path) as im:
            im = im.convert("RGB")
        return tfm(im)

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        xs = []
        for p in batch_paths:
            try:
                xs.append(load_one(p))
            except Exception as e:
                print(f"[WARN] Skipping unreadable image: {p} ({e})")

        if not xs:
            continue

        x = torch.stack(xs, dim=0).to(device)
        emb = model(x)  # (B,D) already L2-normalized
        all_embs.append(emb.detach().cpu())

    if not all_embs:
        raise SystemExit("All images failed to load; no embeddings produced.")

    return torch.cat(all_embs, dim=0)  # (N,D)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate centroid embedding from a folder of images (open-set die)."
    )
    ap.add_argument(
        "--deploy",
        type=Path,
        required=True,
        help="Path to deploy_model.pt from training output.",
    )
    ap.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Folder containing images for ONE die (recursive).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output centroid .npy path (parent dir will be created).",
    )
    ap.add_argument(
        "--device", choices=["cpu", "cuda"], default="cpu", help="Inference device."
    )
    ap.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for embedding inference."
    )
    ap.add_argument(
        "--die-id",
        type=str,
        default="",
        help="Optional: e.g. '24' or 'die_24' (stored in metadata only).",
    )
    args = ap.parse_args()

    bundle = load_deploy_bundle(args.deploy)
    meta = bundle.get("meta", {}) or {}

    backbone = meta.get("backbone", "efficientnet_b0")
    embedding_dim = int(meta.get("embedding_dim", 256))
    input_size = int(meta.get("input_size", 224))

    device = torch.device(
        "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    )

    model = EfficientNetEmbedding(
        backbone_name=backbone, embedding_dim=embedding_dim, pretrained=False
    )
    model.load_state_dict(bundle["model_state"], strict=True)
    model.to(device).eval()

    tfm = build_val_transform(input_size=input_size)
    image_paths = list_images(args.images)

    embs = embed_images(
        model=model,
        tfm=tfm,
        image_paths=image_paths,
        device=device,
        batch_size=max(1, int(args.batch_size)),
    )  # (N,D) on CPU

    # Centroid = mean of embeddings, then L2 normalize again
    centroid = embs.mean(dim=0, keepdim=True)  # (1,D)
    centroid = l2_normalize(centroid).squeeze(0)  # (D,)
    centroid_np = centroid.numpy().astype(np.float32)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(args.out), centroid_np)

    # Sidecar metadata
    meta_out = args.out.with_suffix(".json")
    sidecar = {
        "die_id": args.die_id,
        "images_root": str(args.images.resolve()),
        "num_images_found": len(image_paths),
        "num_images_embedded": int(embs.shape[0]),
        "embedding_dim": int(centroid_np.shape[0]),
        "model_backbone": backbone,
        "model_input_size": input_size,
        "deploy_model_path": str(args.deploy.resolve()),
    }
    meta_out.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

    print("=== Done ===")
    print(
        f"Embedded images: {sidecar['num_images_embedded']} / {sidecar['num_images_found']}"
    )
    print(f"Saved centroid:  {args.out.resolve()}")
    print(f"Saved metadata:  {meta_out.resolve()}")


if __name__ == "__main__":
    main()
