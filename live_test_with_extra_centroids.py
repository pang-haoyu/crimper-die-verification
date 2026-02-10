#!/usr/bin/env python3

"""
Live Smart Die Validation (PASS / FAIL / UNCERTAIN) with simple OpenCV UI.

Core idea (new logic)
- Only "valid frames" are those where the ArUco-based ROI warp succeeds.
- For each valid frame, compute cosine similarities to ALL die centroids.
  - PASS: recommended die similarity >= tau
  - FAIL: recommended die < tau, but some OTHER die >= tau (and is the best match)
          (optionally require a margin: best_other - recommended >= margin)
  - UNCERTAIN: no die reaches tau (max similarity < tau)
- Over the 3s validation window, take a majority vote over PASS/FAIL/UNCERTAIN.
  - If too few valid frames -> UNCERTAIN
  - Ties -> UNCERTAIN (conservative)

Outputs
- Logs one JSON line per validation trigger to --out (jsonl).
- UI shows live camera, ROI crop, current pair, last result, and per-session counts.

Dependencies
- pip install torch torchvision timm numpy opencv-contrib-python

Example
python live_validate_3state.py \
  --artifacts /path/to/outdir \
  --pairs pairs_30.json \
  --camera 0 \
  --duration 3.0 \
  --interval-ms 150 \
  --threshold-source meta_adj_005 \
  --margin 0.05 \
  --min-valid 10 \
  --out live_results_3state.jsonl
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
from PIL import Image


# -----------------------------
# ArUco warp (minimal, adapted from your capture_aruco_crop.py approach)
# -----------------------------
@dataclass(frozen=True)
class DetectionResult:
    quad: np.ndarray  # (4,2) float32 TL,TR,BR,BL
    warped: np.ndarray  # (out_size,out_size,3) BGR
    debug_poly: np.ndarray  # polygon points int32 for drawing


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
    min_area: float,
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
        items = sorted(valid.items(), key=lambda kv: marker_area(kv[1]), reverse=True)[
            :4
        ]
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
    warped = cv2.warpPerspective(
        frame_bgr, M, (out_size, out_size), flags=cv2.INTER_LINEAR
    )
    debug_poly = quad.astype(np.int32).reshape(-1, 1, 2)
    return DetectionResult(quad=quad, warped=warped, debug_poly=debug_poly)


# -----------------------------
# Model (embedding) + transforms
# -----------------------------
def imagenet_normalize() -> Tuple[
    Tuple[float, float, float], Tuple[float, float, float]
]:
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    return mean, std


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / torch.clamp(torch.norm(x, dim=1, keepdim=True), min=eps)


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


# -----------------------------
# 3-state decision logic
# -----------------------------
@dataclass
class SessionDecision:
    decision: str  # PASS / FAIL / UNCERTAIN
    valid_frames: int
    pass_frames: int
    fail_frames: int
    uncertain_frames: int
    pass_ratio: float
    fail_ratio: float
    uncertain_ratio: float
    mean_sim_rec: float
    min_sim_rec: float
    mean_sim_best: float
    min_sim_best: float
    top_wrong_class: Optional[str]  # most frequent wrong class for FAIL frames
    top_wrong_count: int
    latency_ms: float


def label_frame_3state(
    sims_all: np.ndarray,  # (C,)
    rec_idx: int,
    tau: float,
    margin: float,
) -> Tuple[str, int, float, float]:
    """
    Returns:
      label: PASS/FAIL/UNCERTAIN
      top1_idx: argmax over classes
      s_rec: similarity to recommended
      s_top1: max similarity
    """
    top1_idx = int(np.argmax(sims_all))
    s_top1 = float(sims_all[top1_idx])
    s_rec = float(sims_all[rec_idx])

    # PASS if recommended confidently matches
    if s_rec >= tau:
        return "PASS", top1_idx, s_rec, s_top1

    # If nothing matches confidently, it's UNCERTAIN
    if s_top1 < tau:
        return "UNCERTAIN", top1_idx, s_rec, s_top1

    # Otherwise some die matches confidently (>= tau), but not the recommended
    # Add margin to avoid flaky FAIL when recommended is close.
    if top1_idx != rec_idx and (s_top1 - s_rec) >= margin:
        return "FAIL", top1_idx, s_rec, s_top1

    # Conservative fallback: not enough separation, treat as UNCERTAIN
    return "UNCERTAIN", top1_idx, s_rec, s_top1


def majority_vote_3state(pass_c: int, fail_c: int, unc_c: int) -> str:
    """
    Conservative:
    - Winner must be strictly greater than the others.
    - Any tie => UNCERTAIN.
    """
    if pass_c > fail_c and pass_c > unc_c:
        return "PASS"
    if fail_c > pass_c and fail_c > unc_c:
        return "FAIL"
    if unc_c > pass_c and unc_c > fail_c:
        return "UNCERTAIN"
    return "UNCERTAIN"


@torch.no_grad()
def run_session(
    cap: cv2.VideoCapture,
    aruco_dict: cv2.aruco_Dictionary,
    expected_ids: Optional[List[int]],
    roi_size: int,
    min_marker_area: float,
    model: nn.Module,
    centroids: torch.Tensor,  # (C,D) normalized
    classes: List[str],
    class_to_idx: Dict[str, int],
    recommended: str,
    tau: float,
    margin: float,
    duration_s: float,
    interval_ms: int,
    min_valid: int,
    device: torch.device,
    tfm: transforms.Compose,
    preview_windows: bool = True,
) -> Tuple[SessionDecision, Optional[np.ndarray], Optional[np.ndarray]]:
    t0 = time.time()

    rec_idx = class_to_idx.get(recommended)
    if rec_idx is None:
        raise ValueError(
            f"Recommended class '{recommended}' not found in model classes."
        )

    # Counters
    pass_c = 0
    fail_c = 0
    unc_c = 0
    valid = 0

    # Diagnostics
    s_rec_list: List[float] = []
    s_top1_list: List[float] = []
    wrong_top1_idxs: List[int] = []

    last_frame = None
    last_roi = None

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
            valid += 1

            roi_rgb = cv2.cvtColor(det.warped, cv2.COLOR_BGR2RGB)
            roi_pil = Image.fromarray(roi_rgb)  # ✅ convert numpy -> PIL
            x = tfm(roi_pil).unsqueeze(0).to(device)  # now Resize/CenterCrop work

            emb = model(x)  # (1,D), normalized
            sims = (emb @ centroids.t())[0].detach().float().cpu().numpy()  # (C,)

            label, top1_idx, s_rec, s_top1 = label_frame_3state(
                sims_all=sims, rec_idx=rec_idx, tau=tau, margin=margin
            )

            s_rec_list.append(s_rec)
            s_top1_list.append(s_top1)

            if label == "PASS":
                pass_c += 1
            elif label == "FAIL":
                fail_c += 1
                wrong_top1_idxs.append(top1_idx)
            else:
                unc_c += 1

        if preview_windows:
            frame_vis = frame.copy()
            if det is not None:
                cv2.polylines(frame_vis, [det.debug_poly], True, (0, 255, 0), 2)
            cv2.imshow("Live Camera", frame_vis)
            if last_roi is not None:
                cv2.imshow("ROI Crop", last_roi)

        # pacing
        if interval_ms > 0:
            time.sleep(interval_ms / 1000.0)

        # allow UI events
        if cv2.waitKey(1) & 0xFF == 27:
            break

    latency_ms = (time.time() - t0) * 1000.0

    # Not enough evidence => UNCERTAIN session
    if valid < min_valid:
        decision = "UNCERTAIN"
    else:
        decision = majority_vote_3state(pass_c, fail_c, unc_c)

    # Most frequent wrong predicted class among FAIL frames
    top_wrong_class = None
    top_wrong_count = 0
    if wrong_top1_idxs:
        vals, counts = np.unique(
            np.array(wrong_top1_idxs, dtype=np.int32), return_counts=True
        )
        j = int(np.argmax(counts))
        top_wrong_idx = int(vals[j])
        top_wrong_count = int(counts[j])
        if 0 <= top_wrong_idx < len(classes):
            top_wrong_class = classes[top_wrong_idx]

    def safe_mean(x: List[float]) -> float:
        return float(np.mean(x)) if x else float("nan")

    def safe_min(x: List[float]) -> float:
        return float(np.min(x)) if x else float("nan")

    total = max(1, valid)
    return (
        SessionDecision(
            decision=decision,
            valid_frames=valid,
            pass_frames=pass_c,
            fail_frames=fail_c,
            uncertain_frames=unc_c,
            pass_ratio=float(pass_c / total),
            fail_ratio=float(fail_c / total),
            uncertain_ratio=float(unc_c / total),
            mean_sim_rec=safe_mean(s_rec_list),
            min_sim_rec=safe_min(s_rec_list),
            mean_sim_best=safe_mean(s_top1_list),
            min_sim_best=safe_min(s_top1_list),
            top_wrong_class=top_wrong_class,
            top_wrong_count=top_wrong_count,
            latency_ms=latency_ms,
        ),
        last_frame,
        last_roi,
    )


# -----------------------------
# UI helpers
# -----------------------------
def draw_overlay(
    img: np.ndarray, lines: List[str], x: int = 10, y: int = 25, dy: int = 22
) -> np.ndarray:
    out = img.copy()
    for i, line in enumerate(lines):
        cv2.putText(
            out,
            line,
            (x, y + i * dy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            line,
            (x, y + i * dy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return out


# -----------------------------
# CLI + loading
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Live Smart Die Validation UI (PASS/FAIL/UNCERTAIN)"
    )
    p.add_argument(
        "--artifacts",
        type=str,
        required=True,
        help="Training output directory containing deploy_model.pt and centroids.npy (and optionally export_meta.json).",
    )

    p.add_argument(
        "--extra-centroids-dir",
        type=str,
        default="",
        help=(
            "Optional directory containing extra centroid .npy files for open-set testing. "
            "Each file must contain a single 1D centroid vector (D,) and be named like "
            "'die_24_centroid.npy' or 'die_24.npy'. If provided, these centroids are appended "
            "to centroids.npy and their die names are appended to the class list at runtime. "
            "If omitted, behavior matches the original closed-set script."
        ),
    )
    p.add_argument(
        "--pairs",
        type=str,
        required=True,
        help="JSON file containing list of die pairs to test.",
    )
    p.add_argument(
        "--out",
        type=str,
        default="live_results_3state.jsonl",
        help="Output log file (JSON lines).",
    )

    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)

    p.add_argument(
        "--aruco-dict",
        type=str,
        default="DICT_4X4_50",
        help="OpenCV aruco dictionary name (must exist in cv2.aruco).",
    )
    p.add_argument(
        "--ids",
        type=str,
        default="",
        help="Optional: 4 comma-separated ArUco IDs to lock onto (e.g., '0,1,2,3').",
    )
    p.add_argument(
        "--min-marker-area",
        type=float,
        default=900.0,
        help="Min marker area in px^2 to accept a detected marker.",
    )
    p.add_argument(
        "--roi-size",
        type=int,
        default=224,
        help="Warped ROI output size (typically equals training input size; default 224).",
    )

    p.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="Capture window per validation trigger (seconds).",
    )
    p.add_argument(
        "--interval-ms",
        type=int,
        default=150,
        help="Sampling interval in ms during capture window.",
    )
    p.add_argument(
        "--min-valid",
        type=int,
        default=10,
        help="Minimum valid ROI frames required; else session is UNCERTAIN.",
    )

    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override threshold tau directly. If set, ignores --threshold-source.",
    )
    p.add_argument(
        "--threshold-source",
        type=str,
        default="meta_adj_005",
        choices=[
            "meta_adj_005",
            "meta_hard_005",
            "meta_all_005",
            "meta_adj_001",
            "meta_hard_001",
            "meta_all_001",
            "meta_adj_010",
            "meta_hard_010",
            "meta_all_010",
        ],
        help="Where to pull tau from export_meta.json if --threshold not set.",
    )

    p.add_argument(
        "--margin",
        type=float,
        default=0.05,
        help="FAIL margin: require (best_other - recommended) >= margin. Set 0.0 to disable.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run inference on.",
    )
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
        raise ValueError(
            f"Could not read threshold from meta with source={source}: {e}"
        )


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
        classes = meta.get("classes", [])
    if not classes:
        raise SystemExit("Could not find 'classes' in deploy_model.pt bundle.")

    backbone = meta.get("backbone", "efficientnet_b0")
    embedding_dim = int(meta.get("embedding_dim", 256))
    input_size = int(meta.get("input_size", 224))

    # Threshold
    if args.threshold is not None:
        tau = float(args.threshold)
        tau_source = "--threshold"
    else:
        meta_for_tau = json.loads(meta_path.read_text()) if meta_path.exists() else meta
        tau = threshold_from_meta(meta_for_tau, args.threshold_source)
        tau_source = args.threshold_source

    # Build model
    model = EfficientNetEmbedding(
        backbone_name=backbone, embedding_dim=embedding_dim, pretrained=False
    )
    model.load_state_dict(bundle["model_state"], strict=True)

    device = torch.device(
        "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    )
    model.to(device).eval()

    # Load centroids
    centroids_np = np.load(str(centroids_path))
    if centroids_np.ndim != 2:
        raise SystemExit("centroids.npy must be a 2D array (C,D).")
    centroids_np = centroids_np.astype(np.float32, copy=False)

    # Optionally append extra (open-set) centroids
    extra_classes: List[str] = []
    extra_centroids: List[np.ndarray] = []
    if args.extra_centroids_dir.strip():
        extra_dir = Path(args.extra_centroids_dir)
        if not extra_dir.exists():
            raise SystemExit(f"--extra-centroids-dir does not exist: {extra_dir}")

        for npy_path in sorted(extra_dir.glob("*.npy")):
            stem = npy_path.stem  # e.g. "die_24_centroid" or "die_24"
            die_name = stem[: -len("_centroid")] if stem.endswith("_centroid") else stem

            v = np.load(str(npy_path)).astype(np.float32, copy=False)
            if v.ndim != 1:
                raise SystemExit(
                    f"Extra centroid must be 1D (D,), got shape={v.shape} from {npy_path}"
                )
            if v.shape[0] != centroids_np.shape[1]:
                raise SystemExit(
                    f"Extra centroid dim mismatch: got {v.shape[0]} but base centroids have D={centroids_np.shape[1]} "
                    f"(file: {npy_path})"
                )
            if die_name in classes or die_name in extra_classes:
                raise SystemExit(
                    f"Duplicate die name '{die_name}' from extra centroid file {npy_path}. "
                    "Extra centroid names must not collide with existing classes."
                )

            extra_classes.append(die_name)
            extra_centroids.append(v)

        if extra_centroids:
            centroids_np = np.vstack([centroids_np, np.stack(extra_centroids, axis=0)])
            classes = list(classes) + extra_classes
        centroids = torch.from_numpy(centroids_np).float()
        centroids = l2_normalize(centroids).to(device)

        if centroids.shape[0] != len(classes):
            raise SystemExit(
                f"Centroids count ({centroids.shape[0]}) != number of classes ({len(classes)})."
            )

        class_to_idx = {c: i for i, c in enumerate(classes)}

        # Validate pairs against known classes
        for i, r in enumerate(pairs):
            if r["recommended"] not in class_to_idx:
                raise SystemExit(
                    f"Pairs[{i}].recommended='{r['recommended']}' not in model classes: {classes}"
                )
            if r["installed"] not in class_to_idx:
                raise SystemExit(
                    f"Pairs[{i}].installed='{r['installed']}' not in model classes: {classes}"
                )

        tfm = build_val_transform(input_size=input_size)

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
        print("  v or SPACE : run 3s validation for current pair")
        print("  q or ESC : quit")
        print("")
        print(f"Model classes: {classes}")
        if args.extra_centroids_dir.strip() and extra_classes:
            print(f"Extra centroids loaded: {extra_classes}")
        print(
            f"tau={tau:.6f} (source={tau_source})  margin={args.margin:.3f}  min_valid={args.min_valid}"
        )
        print(f"Logging to: {out_path.resolve()}")

        idx = 0
        last_session: Optional[SessionDecision] = None

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

            pair = pairs[idx]
            recommended = pair["recommended"]
            installed = pair["installed"]
            ptype = pair.get("type", "")

            cam_vis = frame.copy()
            if det is not None:
                cv2.polylines(cam_vis, [det.debug_poly], True, (0, 255, 0), 2)

            lines = [
                f"Pair {idx + 1}/{len(pairs)}  type={ptype}",
                f"Recommended: {recommended}",
                f"Installed:   {installed}",
                f"tau={tau:.4f} margin={args.margin:.3f} min_valid={args.min_valid}",
                "Keys: n/b navigate | v/SPACE validate | q quit",
            ]

            if last_session is not None:
                lines += [
                    f"Last session: {last_session.decision}  valid={last_session.valid_frames}  "
                    f"P/F/U={last_session.pass_frames}/{last_session.fail_frames}/{last_session.uncertain_frames}",
                    f"ratios P/F/U={last_session.pass_ratio:.2f}/{last_session.fail_ratio:.2f}/{last_session.uncertain_ratio:.2f}  "
                    f"latency={last_session.latency_ms:.0f}ms",
                ]
                if (
                    last_session.decision == "FAIL"
                    and last_session.top_wrong_class is not None
                ):
                    lines.append(
                        f"FAIL mostly matched: {last_session.top_wrong_class} (count={last_session.top_wrong_count})"
                    )

            cam_vis = draw_overlay(cam_vis, lines)
            cv2.imshow("Live Camera", cam_vis)

            if det is not None:
                cv2.imshow("ROI Crop", det.warped)
            else:
                blank = np.zeros((args.roi_size, args.roi_size, 3), dtype=np.uint8)
                cv2.putText(
                    blank,
                    "NO ROI",
                    (30, args.roi_size // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow("ROI Crop", blank)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("n"):
                idx = (idx + 1) % len(pairs)
                last_session = None
            elif key == ord("b"):
                idx = (idx - 1) % len(pairs)
                last_session = None
            elif key in (ord("v"), 32):
                session, _, _ = run_session(
                    cap=cap,
                    aruco_dict=aruco_dict,
                    expected_ids=expected_ids,
                    roi_size=args.roi_size,
                    min_marker_area=args.min_marker_area,
                    model=model,
                    centroids=centroids,
                    classes=classes,
                    class_to_idx=class_to_idx,
                    recommended=recommended,
                    tau=tau,
                    margin=args.margin,
                    duration_s=args.duration,
                    interval_ms=args.interval_ms,
                    min_valid=args.min_valid,
                    device=device,
                    tfm=tfm,
                    preview_windows=True,
                )
                last_session = session

                is_correct = bool(recommended == installed)

                record = {
                    "timestamp": time.time(),
                    "pair_index": idx,
                    "pair_type": ptype,
                    "recommended": recommended,
                    "installed": installed,
                    "is_correct": is_correct,
                    "decision": session.decision,
                    "valid_frames": session.valid_frames,
                    "pass_frames": session.pass_frames,
                    "fail_frames": session.fail_frames,
                    "uncertain_frames": session.uncertain_frames,
                    "pass_ratio": session.pass_ratio,
                    "fail_ratio": session.fail_ratio,
                    "uncertain_ratio": session.uncertain_ratio,
                    "mean_sim_recommended": session.mean_sim_rec,
                    "min_sim_recommended": session.min_sim_rec,
                    "mean_sim_best": session.mean_sim_best,
                    "min_sim_best": session.min_sim_best,
                    "top_wrong_class": session.top_wrong_class,
                    "top_wrong_count": session.top_wrong_count,
                    "latency_ms": session.latency_ms,
                    "tau": tau,
                    "tau_source": tau_source,
                    "margin": args.margin,
                    "min_valid": args.min_valid,
                    "duration_s": args.duration,
                    "interval_ms": args.interval_ms,
                    "model_backbone": backbone,
                    "embedding_dim": embedding_dim,
                    "input_size": input_size,
                    "classes": classes,
                    "extra_centroids_dir": args.extra_centroids_dir,
                    "extra_classes": extra_classes,
                }

                with out_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")

        cap.release()
        cv2.destroyAllWindows()
        print(f"Done. Results saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
