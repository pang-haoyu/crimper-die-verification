#!/usr/bin/env python3

"""
Live capture + ArUco-based die-set cropping (square warp).

- Detects 4 ArUco markers near the corners of the region of interest.
- Uses each marker's "inner corner" (closest to image center) to define a quadrilateral ROI.
- Warps ROI to a square image of size --out-size.
- Shows live preview with overlay polygon for the crop region.
- Press 'c' to capture; press 'q' or ESC to quit.

Tested conceptually for OpenCV 4.x with aruco (opencv-contrib-python).
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class DetectionResult:
    quad: np.ndarray          # (4,2) float32 ordered as TL,TR,BR,BL in input image coords
    warped: np.ndarray        # output square image
    debug_poly: np.ndarray    # polygon points int32 for drawing


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ArUco-guided square crop capture tool")
    p.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    p.add_argument("--width", type=int, default=1280, help="Requested capture width")
    p.add_argument("--height", type=int, default=720, help="Requested capture height")
    p.add_argument("--incoming", type=Path, default=Path("incoming"), help="Output folder for captures")
    p.add_argument("--out-size", type=int, default=640, help="Output square size in pixels")
    p.add_argument("--dict", type=str, default="DICT_4X4_50",
                   help="ArUco dictionary name, e.g. DICT_4X4_50, DICT_5X5_100, ...")
    p.add_argument("--ids", type=str, default="",
                   help="Optional comma-separated marker IDs expected (exactly 4). Example: 0,1,2,3. "
                        "If empty, any 4 detected markers are used.")
    p.add_argument("--save-raw", action="store_true", help="Also save raw frames on capture")
    p.add_argument("--min-area", type=float, default=500.0,
                   help="Reject detected markers with area smaller than this (default: 500)")
    return p.parse_args()


def get_aruco_dict(name: str) -> cv2.aruco_Dictionary:
    if not hasattr(cv2.aruco, name):
        raise SystemExit(f"Unknown ArUco dictionary: {name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def order_points_tl_tr_br_bl(pts: np.ndarray) -> np.ndarray:
    """
    Orders 4 points into TL, TR, BR, BL.
    pts shape: (4,2)
    """
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.stack([tl, tr, br, bl], axis=0).astype(np.float32)


def polygon_area(quad: np.ndarray) -> float:
    x = quad[:, 0]
    y = quad[:, 1]
    return 0.5 * float(np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def marker_area(corners_4x2: np.ndarray) -> float:
    return polygon_area(corners_4x2.astype(np.float32))


def choose_inner_corner(marker_corners: np.ndarray, img_center: Tuple[float, float]) -> np.ndarray:
    """
    marker_corners: (4,2) for a single marker in OpenCV order.
    Returns the corner point (2,) that is closest to image center.
    This approximates the "inner corner" when markers are on the perimeter.
    """
    cx, cy = img_center
    d = np.sum((marker_corners - np.array([cx, cy], dtype=np.float32)) ** 2, axis=1)
    return marker_corners[int(np.argmin(d))]


def detect_and_warp(
    frame_bgr: np.ndarray,
    aruco_dict: cv2.aruco_Dictionary,
    out_size: int,
    expected_ids: Optional[List[int]],
    min_area: float
) -> Optional[DetectionResult]:
    """
    Returns DetectionResult if 4 suitable markers are detected; otherwise None.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) < 4:
        return None

    ids = ids.flatten().tolist()

    # Filter markers by area threshold to avoid tiny false positives.
    valid: Dict[int, np.ndarray] = {}
    h, w = gray.shape[:2]
    center = (w * 0.5, h * 0.5)

    for c, mid in zip(corners, ids):
        # c shape from OpenCV: (1,4,2)
        c4 = c.reshape(4, 2).astype(np.float32)
        if marker_area(c4) >= min_area:
            valid[int(mid)] = c4

    if len(valid) < 4:
        return None

    # Select 4 markers:
    if expected_ids is not None:
        if len(expected_ids) != 4:
            raise SystemExit("--ids must contain exactly 4 comma-separated IDs.")
        if not all(mid in valid for mid in expected_ids):
            return None
        chosen = {mid: valid[mid] for mid in expected_ids}
    else:
        # Use the largest 4 by area (robust when more than 4 are visible)
        items = sorted(valid.items(), key=lambda kv: marker_area(kv[1]), reverse=True)[:4]
        chosen = dict(items)

    # For each marker, take the corner closest to image center (inner-ish corner)
    inner_pts = []
    for mid, c4 in chosen.items():
        inner_pts.append(choose_inner_corner(c4, center))

    inner_pts = np.stack(inner_pts, axis=0)  # (4,2)

    # Order points TL,TR,BR,BL and sanity-check polygon
    quad = order_points_tl_tr_br_bl(inner_pts)
    if polygon_area(quad) < 1000.0:  # additional safeguard
        return None

    # Perspective warp to square
    dst = np.array([
        [0, 0],
        [out_size - 1, 0],
        [out_size - 1, out_size - 1],
        [0, out_size - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(frame_bgr, M, (out_size, out_size), flags=cv2.INTER_LINEAR)

    debug_poly = quad.astype(np.int32).reshape(-1, 1, 2)
    return DetectionResult(quad=quad, warped=warped, debug_poly=debug_poly)


def main() -> None:
    args = parse_args()
    incoming: Path = args.incoming
    incoming.mkdir(parents=True, exist_ok=True)

    expected_ids = None
    if args.ids.strip():
        expected_ids = [int(x.strip()) for x in args.ids.split(",") if x.strip()]

    aruco_dict = get_aruco_dict(args.dict)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}")

    # Try to set resolution; some cameras ignore these.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(args.width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(args.height))

    print("Controls: 'c' capture | 'q' quit | ESC quit")

    last_warp: Optional[np.ndarray] = None
    last_has_roi = False

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("Frame read failed; exiting.")
            break

        det = detect_and_warp(
            frame_bgr=frame,
            aruco_dict=aruco_dict,
            out_size=args.out_size,
            expected_ids=expected_ids,
            min_area=args.min_area,
        )

        vis = frame.copy()
        if det is not None:
            # Draw polygon ROI
            cv2.polylines(vis, [det.debug_poly], isClosed=True, color=(0, 255, 0), thickness=3)
            cv2.putText(vis, "ROI: OK (press 'c' to capture)", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            last_warp = det.warped
            last_has_roi = True
        else:
            cv2.putText(vis, "ROI: NOT READY (need 4 markers)", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
            last_has_roi = False

        cv2.imshow("Live View (ArUco ROI Overlay)", vis)

        # Optional: show warped preview in another window when available
        if last_warp is not None:
            cv2.imshow("Warped Crop Preview (Square)", last_warp)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):  # ESC or q
            break

        if key == ord("c"):
            if not last_has_roi or last_warp is None:
                print("Capture requested but ROI not ready (4 markers not detected).")
                continue

            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            crop_path = incoming / f"crop_{ts}.jpg"
            ok_crop = cv2.imwrite(str(crop_path), last_warp)

            if not ok_crop:
                print(f"Failed to write {crop_path}")
                continue

            if args.save_raw:
                raw_path = incoming / f"raw_{ts}.jpg"
                cv2.imwrite(str(raw_path), frame)

            print(f"Saved: {crop_path.name}" + (" (and raw frame)" if args.save_raw else ""))

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

