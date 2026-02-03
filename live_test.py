#!/usr/bin/env python3

"""
Live Smart Die Validation (session-based) with simple OpenCV UI.

What it does
- Loads your trained embedding model (deploy_model.pt) and centroids (centroids.npy)
- Reads a curated list of (recommended, installed) pairs from a JSON file
- Shows live camera feed + ArUco-warped ROI crop
- Press a key to trigger a short validation session (e.g., 3s), aggregates over frames,
  makes ACCEPT/REJECT/NO_DECISION, and logs one row per session for later analysis.

Dependencies
- pip install torch torchvision timm numpy opencv-contrib-python

Usage example
python live_validate.py \
  --artifacts /path/to/outdir \
  --pairs pairs_30.json \
  --camera 0 \
  --duration 3.0 \
  --interval-ms 150 \
  --threshold-source meta_adj_005 \
  --vote-p 0.60 \
  --min-valid 10 \
  --out live_results.jsonl

Pairs file format (JSON)
[
  {"recommended": "die_17", "installed": "die_19", "type": "hard_negative"},
  ...
]
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import timm
from torchvision import transforms


# -----------------------------
# ArUco warp (copied from capture_aruco_crop.py, minimal)
# -----------------------------
@dataclass(frozen=True)
class DetectionResult:
    quad: np.ndarray       # (4,2) float32 TL,TR,BR,BL
    warped: np.ndarray     # (out_size,out_size,3) BGR
    debug_poly: np.ndarray # polygon points int32 for drawing


def get_aruco_dict(name: str) -> cv2.aruco_Dictionary:
    if not hasattr(cv2.aruco, name):
        raise SystemExit(f"Unknown ArUco dictionary: {name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def polygon_area(pts: np.ndarray) -> float:
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def marker_area(c4: np.ndarray) -> float:
    return polygon_area(c4.astype(np.float32))


def choose_inner_corner(c4: np.ndarray, center_xy: Tuple[float, float]) -> np.ndarray:
    cx, cy = center_xy
    d2 = (c4[:, 0] - cx) ** 2 + (c4[:, 1] - cy) ** 2
    return c4[int(np.argmin(d2))]


def order_points_tl_tr_br_bl(pts: np.ndarray) -> np.ndarray:
    # pts: (4,2)
    s = pts.sum(axis=1)
    diff = pts[:, 1] - pts[:, 0]
    tl = pts[int(np.argmin(s))]
    br = pts[int(np.argmax(s))]
    tr = pts[int(np.argmin(diff))]
    bl = pts[int(np.argmax(diff))]
    return np.stack([tl, tr, br, bl], axis=0).astype(np.float32)


def detect_and_warp(
    frame_bgr: np.ndarray,
    aruco_dict: cv2.aruco_Dictionary,
    out_size: int,
    expected_ids: Optional[List[int]],
    min_area: float
) -> Optional[DetectionResult]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) < 4:
        return None

    ids = ids.flatten().tolist()

    valid: Dict[int, np.ndarray] = {}
    h, w = gray.shape[:2]
    center = (w * 0.5, h * 0.5)

    for c, mid in zip(corners, ids):
        c4 = c.reshape(4, 2).astype(np.float32)
        if marker_area(c4) >= min_area:
            valid[int(mid)] = c4

    if len(valid) < 4:
        return None

    if expected_ids is not None:
        if len(expected_ids) != 4:
            raise SystemExit("--ids must contain exactly 4 comma-separated IDs.")
        if not all(mid in valid for mid in expected_ids):
            return None
        chosen = {mid: valid[mid] for mid in expected_ids}
    else:
        items = sorted(valid.items(), key=lambda kv: marker_area(kv[1]), reverse=True)[:4]
        chosen = dict(items)

    inner_pts = []
    for _, c4 in chosen.items():
        inner_pts.append(choose_inner_corner(c4, center))
    inner_pts = np.stack(inner_pts, axis=0)

    quad = order_points_tl_tr_br_bl(inner_pts)
    if polygon_area(quad) < 1000.0:
        return None

    dst = np.array(
        [[0, 0], [out_size - 1, 0], [out_size - 1, out_size - 1], [0, out_size - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(frame_bgr, M, (out_size, out_size), flags=cv2.INTER_LINEAR)
    debug_poly = quad.astype(np.int32).reshape(-1, 1, 2)
    return DetectionResult(quad=quad, warped=warped, debug_poly=debug_poly)


# -----------------------------
# Model (compatible with train-script / eval_test_set)
# -----------------------------
def imagenet_normalize() -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    return mean, std


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / torch.clamp(torch.norm(x, dim=1, keepdim=True), min=eps)


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


def build_val_transform(input_size: int) -> transforms.Compose:
    mean, std = imagenet_normalize()
    return transforms.Compose([
        transforms.Resize(int(input_size * 1.15)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


# -----------------------------
# Live validation session
# -----------------------------
@dataclass
class Decision:
    decision: str                 # ACCEPT / REJECT / NO_DECISION
    valid_frames: int
    accept_frames: int
    accept_ratio: float
    mean_sim: float
    min_sim: float
    latency_ms: float


@torch.no_grad()
def run_session(
    cap: cv2.VideoCapture,
    aruco_dict: cv2.aruco_Dictionary,
    expected_ids: Optional[List[int]],
    roi_size: int,
    min_marker_area: float,
    model: nn.Module,
    centroids: torch.Tensor,            # (C,D) normalized
    class_to_idx: Dict[str, int],
    recommended: str,
    threshold: float,
    duration_s: float,
    interval_ms: int,
    vote_p: float,
    min_valid: int,
    device: torch.device,
    tfm: transforms.Compose,
    preview_windows: bool = True,
) -> Tuple[Decision, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Captures frames for duration_s, every interval_ms, whenever ROI is valid.
    Returns Decision + last full frame + last ROI.
    """
    t0 = time.time()

    sims: List[float] = []
    last_frame = None
    last_roi = None

    rec_idx = class_to_idx.get(recommended, None)
    if rec_idx is None:
        raise ValueError(f"Recommended class '{recommended}' not found in model classes.")

    next_sample = 0.0
    while (time.time() - t0) < duration_s:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        last_frame = frame

        det = detect_and_warp(
            frame_bgr=frame,
            aruco_dict=aruco_dict,
            out_size=roi_size,
            expected_ids=expected_ids,
            min_area=min_marker_area,
        )
        if det is not None:
            last_roi = det.warped

            # model expects RGB normalized tensor
            roi_rgb = cv2.cvtColor(det.warped, cv2.COLOR_BGR2RGB)
            x = tfm(roi_rgb).unsqueeze(0).to(device)  # (1,3,H,W)

            emb = model(x)                   # (1,D), normalized
            s = float((emb @ centroids.t())[0, rec_idx].item())
            sims.append(s)

        if preview_windows:
            # small live feedback during capture
            frame_vis = frame.copy()
            if det is not None:
                cv2.polylines(frame_vis, [det.debug_poly], True, (0, 255, 0), 2)
            cv2.imshow("Live Camera", frame_vis)
            if last_roi is not None:
                cv2.imshow("ROI Crop", last_roi)

        # pacing
        now = time.time()
        # simple sleep-based interval: wait until interval_ms elapsed since last loop tick
        dt = (interval_ms / 1000.0)
        if dt > 0:
            time.sleep(dt)

        # allow UI events
        if cv2.waitKey(1) & 0xFF == 27:
            break

    latency_ms = (time.time() - t0) * 1000.0

    k = len(sims)
    if k == 0 or k < min_valid:
        return Decision(
            decision="NO_DECISION",
            valid_frames=k,
            accept_frames=0,
            accept_ratio=0.0,
            mean_sim=float(np.mean(sims)) if sims else float("nan"),
            min_sim=float(np.min(sims)) if sims else float("nan"),
            latency_ms=latency_ms,
        ), last_frame, last_roi

    sims_np = np.array(sims, dtype=np.float32)
    accept_frames = int(np.sum(sims_np >= threshold))
    accept_ratio = float(accept_frames / max(1, k))

    decision = "ACCEPT" if accept_ratio >= vote_p else "REJECT"

    return Decision(
        decision=decision,
        valid_frames=k,
        accept_frames=accept_frames,
        accept_ratio=accept_ratio,
        mean_sim=float(np.mean(sims_np)),
        min_sim=float(np.min(sims_np)),
        latency_ms=latency_ms,
    ), last_frame, last_roi


# -----------------------------
# UI helpers
# -----------------------------
def draw_overlay(img: np.ndarray, lines: List[str], x: int = 10, y: int = 25, dy: int = 22) -> np.ndarray:
    out = img.copy()
    for i, line in enumerate(lines):
        cv2.putText(out, line, (x, y + i * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, line, (x, y + i * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live Smart Die Validation UI")
    p.add_argument("--artifacts", type=str, required=True,
                   help="Training output directory containing deploy_model.pt and centroids.npy (and optionally export_meta.json).")
    p.add_argument("--pairs", type=str, required=True,
                   help="JSON file containing list of die pairs to test.")
    p.add_argument("--out", type=str, default="live_results.jsonl",
                   help="Output log file (JSON lines).")

    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)

    p.add_argument("--aruco-dict", type=str, default="DICT_4X4_50",
                   help="OpenCV aruco dictionary name (must exist in cv2.aruco).")
    p.add_argument("--ids", type=str, default="",
                   help="Optional: 4 comma-separated ArUco IDs to lock onto (e.g., '0,1,2,3').")
    p.add_argument("--min-marker-area", type=float, default=900.0,
                   help="Min marker area in px^2 to accept a detected marker.")
    p.add_argument("--roi-size", type=int, default=224,
                   help="Warped ROI output size (should match training input size; default 224).")

    p.add_argument("--duration", type=float, default=3.0,
                   help="Capture window per validation trigger (seconds).")
    p.add_argument("--interval-ms", type=int, default=150,
                   help="Sampling interval in ms during capture window.")
    p.add_argument("--vote-p", type=float, default=0.60,
                   help="Session ACCEPT if accept_ratio >= vote_p.")
    p.add_argument("--min-valid", type=int, default=10,
                   help="Minimum valid ROI frames required; else NO_DECISION.")

    p.add_argument("--threshold", type=float, default=None,
                   help="Override threshold tau directly. If set, ignores --threshold-source.")
    p.add_argument("--threshold-source", type=str, default="meta_adj_005",
                   choices=["meta_adj_005", "meta_hard_005", "meta_all_005", "meta_adj_001", "meta_hard_001", "meta_all_001", "meta_adj_010", "meta_hard_010", "meta_all_010"],
                   help="Where to pull tau from export_meta.json if --threshold not set.")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                   help="Device to run inference on.")
    return p.parse_args()


def load_pairs(path: Path) -> List[Dict[str, Any]]:
    pairs = json.loads(path.read_text())
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("Pairs file must be a non-empty JSON list.")
    for i, r in enumerate(pairs):
        if "recommended" not in r or "installed" not in r:
            raise ValueError(f"Pairs[{i}] missing 'recommended' or 'installed'.")
    return pairs


def threshold_from_meta(meta: Dict[str, Any], source: str) -> float:
    """
    source examples:
      meta_adj_005 -> meta['thresholds']['adj']['0.050']
    """
    mapping = {
        "meta_adj_005": ("adj", "0.050"),
        "meta_hard_005": ("hard", "0.050"),
        "meta_all_005": ("all", "0.050"),
        "meta_adj_001": ("adj", "0.010"),
        "meta_hard_001": ("hard", "0.010"),
        "meta_all_001": ("all", "0.010"),
        "meta_adj_010": ("adj", "0.100"),
        "meta_hard_010": ("hard", "0.100"),
        "meta_all_010": ("all", "0.100"),
    }
    kind, far = mapping[source]
    try:
        return float(meta["thresholds"][kind][far])
    except Exception as e:
        raise ValueError(f"Could not read threshold from meta with source={source}: {e}")


def main() -> None:
    args = parse_args()

    artifacts = Path(args.artifacts)
    deploy_path = artifacts / "deploy_model.pt"
    centroids_path = artifacts / "centroids.npy"
    meta_path = artifacts / "export_meta.json"

    if not deploy_path.exists():
        raise SystemExit(f"Missing: {deploy_path}")
    if not centroids_path.exists():
        raise SystemExit(f"Missing: {centroids_path}")

    pairs = load_pairs(Path(args.pairs))

    # Load deploy bundle
    bundle = torch.load(str(deploy_path), map_location="cpu")
    classes: List[str] = bundle.get("classes", [])
    meta: Dict[str, Any] = bundle.get("meta", {}) or {}
    if not classes:
        # fallback: try meta
        classes = meta.get("classes", [])
    if not classes:
        raise SystemExit("Could not find 'classes' in deploy_model.pt bundle.")

    backbone = meta.get("backbone", "efficientnet_b0")
    embedding_dim = int(meta.get("embedding_dim", 256))
    input_size = int(meta.get("input_size", 224))

    # Threshold
    if args.threshold is not None:
        tau = float(args.threshold)
    else:
        # prefer export_meta.json if present, else bundle meta
        if meta_path.exists():
            meta_from_file = json.loads(meta_path.read_text())
        else:
            meta_from_file = meta
        tau = threshold_from_meta(meta_from_file, args.threshold_source)

    # Build model
    model = EfficientNetEmbedding(backbone_name=backbone, embedding_dim=embedding_dim, pretrained=False)
    model.load_state_dict(bundle["model_state"], strict=True)

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    model.to(device).eval()

    # Load centroids
    centroids_np = np.load(str(centroids_path))
    if centroids_np.ndim != 2:
        raise SystemExit("centroids.npy must be a 2D array (C,D).")
    centroids = torch.from_numpy(centroids_np).float()
    # ensure normalized
    centroids = l2_normalize(centroids)
    centroids = centroids.to(device)

    class_to_idx = {c: i for i, c in enumerate(classes)}

    # Validate pairs against known classes
    bad = []
    for i, r in enumerate(pairs):
        if r["recommended"] not in class_to_idx:
            bad.append((i, "recommended", r["recommended"]))
        if r["installed"] not in class_to_idx:
            bad.append((i, "installed", r["installed"]))
    if bad:
        msg = "\n".join([f"Pairs[{i}] {k}='{v}' not in model classes" for i, k, v in bad[:15]])
        raise SystemExit(f"Pairs file contains unknown class names:\n{msg}\nKnown classes: {classes}")

    # Transforms
    tfm = build_val_transform(input_size=input_size)

    # ArUco config
    expected_ids = None
    if args.ids.strip():
        expected_ids = [int(x.strip()) for x in args.ids.split(",") if x.strip()]
        if len(expected_ids) != 4:
            raise SystemExit("--ids must have exactly 4 IDs.")
    aruco_dict = get_aruco_dict(args.aruco_dict)

    # Camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(args.width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(args.height))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Controls:")
    print("  n : next pair")
    print("  b : previous pair")
    print("  v or SPACE : run validation session for current pair (captures for --duration)")
    print("  q or ESC : quit")
    print("")
    print(f"Model classes: {classes}")
    print(f"Threshold tau: {tau:.6f} (source: {'--threshold' if args.threshold is not None else args.threshold_source})")
    print(f"Logging to: {out_path.resolve()}")

    idx = 0
    last_decision: Optional[Decision] = None
    last_det: Optional[DetectionResult] = None

    # Main UI loop
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        det = detect_and_warp(
            frame_bgr=frame,
            aruco_dict=aruco_dict,
            out_size=args.roi_size,
            expected_ids=expected_ids,
            min_area=args.min_marker_area,
        )
        last_det = det

        pair = pairs[idx]
        recommended = pair["recommended"]
        installed = pair["installed"]
        ptype = pair.get("type", "")

        # Overlay camera view
        cam_vis = frame.copy()
        if det is not None:
            cv2.polylines(cam_vis, [det.debug_poly], True, (0, 255, 0), 2)

        lines = [
            f"Pair {idx+1}/{len(pairs)}  type={ptype}",
            f"Recommended: {recommended}",
            f"Installed:   {installed}",
            f"tau={tau:.4f} vote_p={args.vote_p:.2f} min_valid={args.min_valid}",
            "Keys: n/b navigate | v/SPACE validate | q quit",
        ]
        if last_decision is not None:
            lines += [
                f"Last: {last_decision.decision}  valid={last_decision.valid_frames}  "
                f"acc_frames={last_decision.accept_frames}  acc_ratio={last_decision.accept_ratio:.2f}",
                f"sim_mean={last_decision.mean_sim:.3f} sim_min={last_decision.min_sim:.3f} latency={last_decision.latency_ms:.0f}ms",
            ]

        cam_vis = draw_overlay(cam_vis, lines)
        cv2.imshow("Live Camera", cam_vis)

        if det is not None:
            cv2.imshow("ROI Crop", det.warped)
        else:
            # show a blank ROI window so layout stays stable
            blank = np.zeros((args.roi_size, args.roi_size, 3), dtype=np.uint8)
            cv2.putText(blank, "NO ROI", (30, args.roi_size // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.imshow("ROI Crop", blank)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # q or ESC
            break
        elif key == ord("n"):
            idx = (idx + 1) % len(pairs)
            last_decision = None
        elif key == ord("b"):
            idx = (idx - 1) % len(pairs)
            last_decision = None
        elif key in (ord("v"), 32):  # v or SPACE
            # Run validation session
            dec, last_frame, last_roi = run_session(
                cap=cap,
                aruco_dict=aruco_dict,
                expected_ids=expected_ids,
                roi_size=args.roi_size,
                min_marker_area=args.min_marker_area,
                model=model,
                centroids=centroids,
                class_to_idx=class_to_idx,
                recommended=recommended,
                threshold=tau,
                duration_s=args.duration,
                interval_ms=args.interval_ms,
                vote_p=args.vote_p,
                min_valid=args.min_valid,
                device=device,
                tfm=tfm,
                preview_windows=True,
            )
            last_decision = dec

            # Ground truth correctness (based on pair definition)
            is_correct = bool(recommended == installed)

            # Log one JSON line
            record = {
                "timestamp": time.time(),
                "pair_index": idx,
                "pair_type": ptype,
                "recommended": recommended,
                "installed": installed,
                "is_correct": is_correct,
                "decision": dec.decision,
                "valid_frames": dec.valid_frames,
                "accept_frames": dec.accept_frames,
                "accept_ratio": dec.accept_ratio,
                "mean_sim_to_recommended": dec.mean_sim,
                "min_sim_to_recommended": dec.min_sim,
                "latency_ms": dec.latency_ms,
                "tau": tau,
                "vote_p": args.vote_p,
                "min_valid": args.min_valid,
                "duration_s": args.duration,
                "interval_ms": args.interval_ms,
                "model_backbone": backbone,
                "embedding_dim": embedding_dim,
                "input_size": input_size,
                "classes": classes,
            }
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. Results saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()

