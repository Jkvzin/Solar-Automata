"""
LuminaPV Computer Vision & Signal Processing Engine (Node 2 Worker)
===================================================================
Author: Principal Distributed Systems Engineer & Senior Computer Vision Architect
Purpose: Performs local contrast normalization (CLAHE), robust quadrilateral contour
         and corner detection with perspective homography rectification, adaptive
         projection-profile grid segmentation (supporting Full-Cell and Half-Cut
         photovoltaic topologies with geometric fallback), in-memory cell slicing,
         and electroluminescence (EL) defect classification.
"""

import base64
from enum import Enum
import hashlib
import logging
import math
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from pydantic import BaseModel, Field
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from ultralytics import YOLO

logger = logging.getLogger("cv_engine")

# Standard commercial PV module cell counts (Full-cell, Half-cut & 1x1 Single-Cell)
PLAUSIBLE_CELL_COUNTS = {1, 36, 48, 54, 60, 66, 72, 96, 108, 120, 132, 144}
STANDARD_FAMILIES = {60, 72, 120, 144}


class DefectType(str, Enum):
    DEFEITUOSA = "defeituosa"
    NONE = "none"
    MICRO_CRACK = "micro-crack"
    DEAD = "dead"
    HOT_SPOT = "hot-spot"
    DELAMINATION = "delamination"
    GRID_FAULT = "grid-fault"


class CellBoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int
    x_pct: float
    y_pct: float
    width_pct: float
    height_pct: float


class CellInspection(BaseModel):
    index: int
    row: int
    col: int
    is_defective: bool
    defect_type: DefectType
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: Optional[CellBoundingBox] = None


class ModuleResult(BaseModel):
    module_id: str
    filename: str
    status: str  # "FAIL" | "PASS"
    rows: int
    cols: int
    total_cells: int
    defective_cells_count: int
    module_damage_percentage: float
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    x_bounds: Optional[List[int]] = None
    y_bounds: Optional[List[int]] = None
    active_area: Optional[Dict[str, Any]] = None
    rectified_image: Optional[str] = None
    processed_by_node: Optional[str] = None
    model_sha256: Optional[str] = None
    content_sha256: Optional[str] = None
    replicated: bool = False
    consistency_model: str = "read-your-writes"
    cache_hit: bool = False
    cells: List[CellInspection]


# -----------------------------------------------------------------------------
# 1. Local Contrast Normalization (CLAHE) & Geometric Helpers
# -----------------------------------------------------------------------------
def apply_clahe_normalization(
    gray_img: np.ndarray,
    clip_limit: float = 2.5,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to normalize
    local contrast prior to binarization, edge detection, and structural profiling.
    Preserves uint8 data type and prevents losses under non-uniform EL illumination.
    """
    if gray_img is None or gray_img.size == 0:
        raise ValueError("Cannot apply CLAHE to an empty image.")
    if len(gray_img.shape) == 3:
        gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
    if gray_img.dtype != np.uint8:
        gray_img = np.clip(gray_img, 0, 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray_img)


def order_quad_points(pts: np.ndarray) -> np.ndarray:
    """
    Orders 4 quadrilateral coordinates in strict clockwise order:
    [top-left, top-right, bottom-right, bottom-left].
    Uses centroid polar-angle sorting combined with principal edge orientation
    to prevent self-intersecting (bowtie) quads under rotation up to +/- 45 deg.
    """
    pts_arr = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    center = np.mean(pts_arr, axis=0)

    # Sort vertices clockwise around centroid: atan2 in image coords (y down, x right)
    angles = np.arctan2(pts_arr[:, 1] - center[1], pts_arr[:, 0] - center[0])
    sort_idx = np.argsort(angles)
    cw_pts = pts_arr[sort_idx]

    # Identify top-left (TL) such that the edge TL -> TR is the uppermost horizontal-ish edge
    best_k = 0
    best_score = float("inf")
    for k in range(4):
        p0 = cw_pts[k]
        p1 = cw_pts[(k + 1) % 4]
        edge_vec = p1 - p0
        edge_angle = abs(math.degrees(math.atan2(float(edge_vec[1]), float(edge_vec[0]))))
        mid_y = 0.5 * (p0[1] + p1[1])
        # Top edge must point roughly rightward (angle < 45 deg) and lie above center
        if edge_angle <= 45.0 and mid_y <= center[1]:
            if edge_angle < best_score:
                best_score = edge_angle
                best_k = k

    if best_score == float("inf"):
        # Fallback to classical sum/diff heuristic for extreme aspect ratios
        rect = np.zeros((4, 2), dtype="float32")
        s = pts_arr.sum(axis=1)
        rect[0] = pts_arr[np.argmin(s)]
        rect[2] = pts_arr[np.argmax(s)]
        diff = np.diff(pts_arr, axis=1)
        rect[1] = pts_arr[np.argmin(diff)]
        rect[3] = pts_arr[np.argmax(diff)]
        return rect

    ordered = np.roll(cw_pts, -best_k, axis=0).astype(np.float32)
    return ordered


def _intersect_lines(
    line1: Tuple[float, float, float, float],
    line2: Tuple[float, float, float, float]
) -> Optional[np.ndarray]:
    """
    Computes the analytical intersection (x, y) of two 2D lines represented as
    (vx, vy, x0, y0) where (vx, vy) is a unit direction vector and (x0, y0) is a point.
    """
    vx1, vy1, x1, y1 = line1
    vx2, vy2, x2, y2 = line2
    det = vx1 * vy2 - vy1 * vx2
    if abs(det) < 1e-6:
        return None
    dx = x2 - x1
    dy = y2 - y1
    t = (dx * vy2 - dy * vx2) / det
    ix = x1 + t * vx1
    iy = y1 + t * vy1
    return np.array([ix, iy], dtype=np.float32)


def _fit_quad_from_hull_lines(hull: np.ndarray, img_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    """
    Reconstructs the exact 4-corner perspective quadrilateral from a Convex Hull
    by decomposing hull segments into the 4 dominant directional families
    (Top, Right, Bottom, Left) and intersecting the fitted boundary lines.

    This guarantees accurate corner recovery even when:
    - Corner solar cells are completely dark/dead (producing chamfered 5-8 vertex hulls),
    - The panel suffers trapezoidal perspective distortion, or
    - Minor aluminum frame reflections create localized bumps along an edge.
    """
    h, w = img_shape[:2]
    peri = cv2.arcLength(hull, True)
    if peri <= 0:
        return None

    # Fine polygonal approximation of convex hull to extract linear boundary segments
    approx = cv2.approxPolyDP(hull, 0.003 * peri, True).reshape(-1, 2).astype(np.float32)
    n_pts = len(approx)
    if n_pts < 4:
        return None

    center = np.mean(approx, axis=0)
    segments = []
    for i in range(n_pts):
        p0 = approx[i]
        p1 = approx[(i + 1) % n_pts]
        vec = p1 - p0
        length = float(np.linalg.norm(vec))
        if length < 4.0:
            continue
        angle = math.atan2(float(vec[1]), float(vec[0]))  # (-pi, pi]
        mid = 0.5 * (p0 + p1)
        segments.append((length, angle, mid, p0, p1))

    if len(segments) < 4:
        return None

    # Estimate dominant orientation modulo 90 degrees (pi/2) using 4x angle circular mean
    sin_sum = sum(seg[0] * math.sin(4.0 * seg[1]) for seg in segments)
    cos_sum = sum(seg[0] * math.cos(4.0 * seg[1]) for seg in segments)
    dom_angle = 0.25 * math.atan2(sin_sum, cos_sum)  # in (-pi/4, pi/4]

    cos_a = math.cos(-dom_angle)
    sin_a = math.sin(-dom_angle)

    # Group segments into 4 side buckets: top, right, bottom, left
    # Diagonal chamfers from missing dark corner cells (~45 deg) are naturally rejected
    buckets: Dict[str, List[Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]]] = {
        "top": [],
        "right": [],
        "bottom": [],
        "left": [],
    }

    for length, angle, mid, p0, p1 in segments:
        rel_angle = math.degrees(angle - dom_angle)
        rel_angle = (rel_angle + 180.0) % 360.0 - 180.0  # (-180, 180]

        rel_mid = mid - center
        rx = rel_mid[0] * cos_a - rel_mid[1] * sin_a
        ry = rel_mid[0] * sin_a + rel_mid[1] * cos_a

        abs_rel = abs(rel_angle)
        if abs_rel <= 28.0 or abs_rel >= 152.0:
            # Horizontal-family edge
            if ry < 0:
                buckets["top"].append((length, angle, mid, p0, p1))
            else:
                buckets["bottom"].append((length, angle, mid, p0, p1))
        elif 62.0 <= abs_rel <= 118.0:
            # Vertical-family edge
            if rx > 0:
                buckets["right"].append((length, angle, mid, p0, p1))
            else:
                buckets["left"].append((length, angle, mid, p0, p1))

    if any(len(b) == 0 for b in buckets.values()):
        return None

    fitted_lines: Dict[str, Tuple[float, float, float, float]] = {}
    for side, seg_list in buckets.items():
        # Sort by segment length descending to pick the primary collinear boundary
        seg_list.sort(key=lambda s: s[0], reverse=True)
        best_len, best_ang, best_mid, bp0, bp1 = seg_list[0]
        base_dir = (bp1 - bp0) / max(best_len, 1e-6)
        base_norm = np.array([-base_dir[1], base_dir[0]], dtype=np.float32)

        # Collect inlier points from collinear segments on the same side (RANSAC consensus)
        sampled_pts = []
        for s_len, s_ang, _, sp0, sp1 in seg_list:
            ang_diff = abs((math.degrees(s_ang - best_ang) + 180.0) % 360.0 - 180.0)
            if ang_diff > 180.0:
                ang_diff = 360.0 - ang_diff
            dist_p0 = abs(float(np.dot(sp0 - bp0, base_norm)))
            dist_p1 = abs(float(np.dot(sp1 - bp0, base_norm)))
            if (ang_diff <= 8.0 or abs(ang_diff - 180.0) <= 8.0) and max(dist_p0, dist_p1) <= 8.5:
                n_samples = max(2, int(s_len // 5))
                for t in np.linspace(0.0, 1.0, n_samples):
                    sampled_pts.append(sp0 * (1.0 - t) + sp1 * t)

        pts_array = np.array(sampled_pts, dtype=np.float32)
        if len(pts_array) >= 2:
            line_params = cv2.fitLine(pts_array, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
            fitted_lines[side] = (
                float(line_params[0]),
                float(line_params[1]),
                float(line_params[2]),
                float(line_params[3]),
            )
        else:
            fitted_lines[side] = (
                float(base_dir[0]),
                float(base_dir[1]),
                float(bp0[0]),
                bp0[1],
            )

    tl = _intersect_lines(fitted_lines["top"], fitted_lines["left"])
    tr = _intersect_lines(fitted_lines["top"], fitted_lines["right"])
    br = _intersect_lines(fitted_lines["bottom"], fitted_lines["right"])
    bl = _intersect_lines(fitted_lines["bottom"], fitted_lines["left"])

    if any(pt is None for pt in (tl, tr, br, bl)):
        return None

    quad = np.array([tl, tr, br, bl], dtype=np.float32)
    # Sanity check: intersections must lie within reasonable image margin
    margin_lim = 0.30 * max(w, h)
    if (
        np.any(quad[:, 0] < -margin_lim)
        or np.any(quad[:, 0] > w + margin_lim)
        or np.any(quad[:, 1] < -margin_lim)
        or np.any(quad[:, 1] > h + margin_lim)
    ):
        return None

    ordered = order_quad_points(quad)
    if not cv2.isContourConvex(ordered.reshape(-1, 1, 2)):
        return None

    return ordered


def _is_quad_already_orthogonal_full_frame(
    ordered_quad: np.ndarray,
    img_shape: Tuple[int, int]
) -> bool:
    """
    Checks whether a detected 4-point quadrilateral already occupies the full frame
    in an upright, orthogonal orientation (no significant rotation or trapezoidal skew).
    """
    h, w = img_shape[:2]
    tl, tr, br, bl = ordered_quad

    margin_thresh = max(35.0, float(min(w, h)) * 0.08)
    corners_near_frame = (
        tl[0] <= margin_thresh
        and tl[1] <= margin_thresh
        and (w - tr[0]) <= margin_thresh
        and tr[1] <= margin_thresh
        and (w - br[0]) <= margin_thresh
        and (h - br[1]) <= margin_thresh
        and bl[0] <= margin_thresh
        and (h - bl[1]) <= margin_thresh
    )
    if not corners_near_frame:
        return False

    # Check edge angles for rotation
    top_angle = abs(math.degrees(math.atan2(float(tr[1] - tl[1]), float(tr[0] - tl[0]))))
    bot_angle = abs(math.degrees(math.atan2(float(br[1] - bl[1]), float(br[0] - bl[0]))))
    left_angle = abs(math.degrees(math.atan2(float(bl[1] - tl[1]), float(bl[0] - tl[0]))) - 90.0)
    right_angle = abs(math.degrees(math.atan2(float(br[1] - tr[1]), float(br[0] - tr[0]))) - 90.0)

    if max(top_angle, bot_angle, left_angle, right_angle) > 0.65:
        return False

    # Check trapezoidal length symmetry
    w_top = float(np.linalg.norm(tr - tl))
    w_bot = float(np.linalg.norm(br - bl))
    h_left = float(np.linalg.norm(bl - tl))
    h_right = float(np.linalg.norm(br - tr))

    if abs(w_top - w_bot) > max(6.0, 0.015 * w) or abs(h_left - h_right) > max(6.0, 0.015 * h):
        return False

    return True


def detect_panel_quadrilateral(gray_img: np.ndarray) -> Optional[np.ndarray]:
    """
    Robustly detects the 4 outer corners [TL, TR, BR, BL] of the PV module using:
    1. Local contrast normalization (CLAHE) + bilateral/Gaussian smoothing.
    2. Dual-stage segmentation:
       - Substrate-aware foreground extraction (handles dark/dead peripheral cells
         and anodized module frames against dark test-bench backgrounds).
       - Adaptive Canny edge + CLAHE Otsu active matrix segmentation with
         morphological bridging and reflection/bench-noise suppression.
    3. Convex Hull + 4-side collinear line fitting & intersection to reconstruct
       missing corners caused by dead corner cells or perspective distortion.
    """
    h, w = gray_img.shape[:2]
    img_area = float(w * h)

    clahe_img = apply_clahe_normalization(gray_img, clip_limit=2.5, tile_grid_size=(8, 8))
    blurred_raw = cv2.GaussianBlur(gray_img, (5, 5), 0)
    blurred_clahe = cv2.GaussianBlur(clahe_img, (5, 5), 0)

    # Estimate test-bench background from outermost 2px perimeter
    top_b = blurred_raw[:2, :].ravel()
    bot_b = blurred_raw[-2:, :].ravel()
    left_b = blurred_raw[:, :2].ravel()
    right_b = blurred_raw[:, -2:].ravel()
    border_pixels = np.concatenate([top_b, bot_b, left_b, right_b])
    bg_p15 = float(np.percentile(border_pixels, 15))
    bg_median = float(np.median(border_pixels))

    k_close_size = max(15, int(min(w, h) * 0.045))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (k_close_size, k_close_size))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    candidate_hulls: List[np.ndarray] = []

    # Stage A: Substrate + frame segmentation when module sits on a darker test bench
    # Captures the entire panel board including dark/dead border cells (intensity ~15-35)
    if bg_p15 < 14.0:
        sub_thresh_val = max(bg_p15 + 6.0, min(bg_median + 5.0, 14.0))
        _, sub_mask = cv2.threshold(blurred_raw, int(sub_thresh_val), 255, cv2.THRESH_BINARY)
        sub_opened = cv2.morphologyEx(sub_mask, cv2.MORPH_OPEN, kernel_open)
        sub_closed = cv2.morphologyEx(sub_opened, cv2.MORPH_CLOSE, kernel_close)
        cnts_sub, _ = cv2.findContours(sub_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts_sub:
            largest_sub = max(cnts_sub, key=cv2.contourArea)
            if cv2.contourArea(largest_sub) > 0.15 * img_area:
                candidate_hulls.append(cv2.convexHull(largest_sub))

    # Stage B: CLAHE + Adaptive Otsu + Adaptive Canny Edges
    _, otsu_clahe = cv2.threshold(blurred_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    med_val = float(np.median(blurred_clahe))
    canny_low = int(max(15.0, 0.45 * med_val))
    canny_high = int(min(240.0, max(60.0, 1.35 * med_val)))
    edges = cv2.Canny(blurred_clahe, canny_low, canny_high)

    # Combine CLAHE binary mask with internal/boundary Canny edges, suppress bench noise
    combined_mask = cv2.bitwise_or(otsu_clahe, edges)
    cleaned_mask = cv2.morphologyEx(otsu_clahe, cv2.MORPH_OPEN, kernel_open)
    bridged_mask = cv2.morphologyEx(
        cv2.bitwise_or(cleaned_mask, edges),
        cv2.MORPH_CLOSE,
        kernel_close
    )

    cnts_clahe, _ = cv2.findContours(bridged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts_clahe:
        # Filter out small background bench noise blobs (< 0.4% of frame area)
        valid_cnts = [c for c in cnts_clahe if cv2.contourArea(c) > 0.004 * img_area]
        if valid_cnts:
            largest_clahe = max(valid_cnts, key=cv2.contourArea)
            if cv2.contourArea(largest_clahe) > 0.12 * img_area:
                candidate_hulls.append(cv2.convexHull(largest_clahe))

    # Stage C: Multi-Otsu active luminescent matrix isolation for external camera captures
    # Separates illuminated photovoltaic cells from dark/gray test-bench surroundings
    otsu_val, otsu_raw = cv2.threshold(blurred_raw, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg_pixels = blurred_raw[blurred_raw > max(12.0, float(otsu_val) * 0.55)]
    if fg_pixels.size > 100:
        multi_otsu_val, _ = cv2.threshold(fg_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        eff_thresh = int(max(otsu_val, min(multi_otsu_val * 0.85, np.percentile(blurred_raw, 72))))
    else:
        eff_thresh = int(otsu_val)

    _, active_otsu = cv2.threshold(blurred_raw, eff_thresh, 255, cv2.THRESH_BINARY)
    active_otsu = cv2.morphologyEx(active_otsu, cv2.MORPH_OPEN, kernel_open)
    active_otsu = cv2.morphologyEx(active_otsu, cv2.MORPH_CLOSE, kernel_close)
    cnts_otsu, _ = cv2.findContours(active_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts_otsu:
        valid_otsu = [c for c in cnts_otsu if cv2.contourArea(c) > 0.08 * img_area]
        if valid_otsu:
            largest_otsu = max(valid_otsu, key=cv2.contourArea)
            candidate_hulls.append(cv2.convexHull(largest_otsu))

    if not candidate_hulls:
        return None

    # Prefer the hull that yields a clean 4-line quad with largest valid area
    best_quad: Optional[np.ndarray] = None
    best_area = -1.0

    for hull in candidate_hulls:
        quad = _fit_quad_from_hull_lines(hull, (h, w))
        if quad is None:
            rect = cv2.minAreaRect(hull)
            quad = order_quad_points(cv2.boxPoints(rect).astype(np.float32))

        q_area = float(cv2.contourArea(quad.reshape(-1, 1, 2)))
        if q_area > 0.12 * img_area and q_area > best_area:
            best_area = q_area
            best_quad = quad

    return best_quad


def _normalize_low_gain_capture(gray_img: np.ndarray) -> np.ndarray:
    """
    Applies 1% to 99% percentile Min-Max contrast/gain stretching on underexposed
    external EL camera captures so that low global illumination is not mistaken
    for an inactive/dead silicon wafer.
    """
    p1 = float(np.percentile(gray_img, 1))
    p99 = float(np.percentile(gray_img, 99))
    p85 = float(np.percentile(gray_img, 85))

    # Only stretch if the capture has valid dynamic range but is globally dim (underexposed)
    if p99 < 145.0 and (p99 - p1) >= 18.0 and (p85 - p1) >= 8.0:
        clipped = np.clip(gray_img.astype(np.float32), p1, p99)
        stretched = 12.0 + ((clipped - p1) / max(p99 - p1, 1.0)) * 203.0
        return np.clip(stretched, 0, 255).astype(np.uint8)

    return gray_img


def normalize_perspective(gray_img: np.ndarray) -> np.ndarray:
    """
    Detects the outer boundary of the photovoltaic module using CLAHE contrast
    normalization, Otsu/multi-Otsu active matrix isolation, Canny edge / Convex Hull
    4-line intersection, and performs homography rectification via perspective transform
    if the module is rotated or trapezoidally distorted.
    Also applies 1%-99% Min-Max gain normalization on dark external captures.
    """
    if gray_img is None or gray_img.size == 0:
        raise ValueError("Input image to normalize_perspective is empty.")

    if len(gray_img.shape) == 3:
        gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)

    gray_img = _normalize_low_gain_capture(gray_img)

    h, w = gray_img.shape[:2]
    # Isolated single-cell crops (e.g., 300x300 ELPV cells) are already ortho-cropped
    if max(h, w) <= 360 and (0.82 <= (float(w) / float(max(1, h))) <= 1.22):
        return gray_img

    ordered = detect_panel_quadrilateral(gray_img)
    if ordered is None:
        return gray_img

    # Preserve original raster if already orthogonal and full-frame
    if _is_quad_already_orthogonal_full_frame(ordered, (h, w)):
        return gray_img

    tl, tr, br, bl = ordered

    width_a = float(np.linalg.norm(br - bl))
    width_b = float(np.linalg.norm(tr - tl))
    max_w = int(round(max(width_a, width_b)))

    height_a = float(np.linalg.norm(tr - br))
    height_b = float(np.linalg.norm(tl - bl))
    max_h = int(round(max(height_a, height_b)))

    if max_w > 100 and max_h > 100:
        dst = np.array([
            [0, 0],
            [max_w - 1, 0],
            [max_w - 1, max_h - 1],
            [0, max_h - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(ordered, dst)
        rectified = cv2.warpPerspective(
            gray_img,
            M,
            (max_w, max_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=int(np.min(gray_img))
        )
        return rectified

    return gray_img


# -----------------------------------------------------------------------------
# 2. Adaptive Projection-Profile Grid Segmentation & Geometric Fallback
# -----------------------------------------------------------------------------
def _geometric_grid_fallback(
    active_h: int,
    active_w: int,
    y_start: int,
    y_end: int,
    x_start: int,
    x_end: int
) -> Tuple[int, int, List[int], List[int]]:
    """
    Deterministic geometric fallback based on classical photovoltaic module
    aspect ratios when extreme noise or severe corruption prevents automatic
    projection profile peak resolution.
    """
    aspect_ratio = float(active_w) / float(max(1, active_h))

    if aspect_ratio >= 1.0:
        # Landscape orientation
        if aspect_ratio >= 2.35:
            rows, cols = 6, 20  # 120 half-cut landscape
        elif aspect_ratio >= 1.82:
            rows, cols = 6, 12  # 72 full-cell commercial
        else:
            rows, cols = 6, 10  # 60 full-cell standard
    else:
        # Portrait orientation
        inv_ar = 1.0 / max(aspect_ratio, 1e-6)
        if inv_ar >= 2.35:
            rows, cols = 20, 6
        elif inv_ar >= 1.82:
            rows, cols = 12, 6
        else:
            rows, cols = 10, 6

    logger.warning(
        "Applying geometric aspect-ratio fallback for grid segmentation "
        "(active_w=%d, active_h=%d, AR=%.2f -> fallback=%dx%d).",
        active_w, active_h, aspect_ratio, rows, cols
    )

    y_bounds = [int(round(y)) for y in np.linspace(y_start, y_end + 1, rows + 1)]
    x_bounds = [int(round(x)) for x in np.linspace(x_start, x_end + 1, cols + 1)]
    return rows, cols, y_bounds, x_bounds


def _detect_1d_dividers(
    active_sub: np.ndarray,
    axis: int,
    min_cells: int = 4,
    max_cells: int = 26
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    Detects cell dividers along a given axis (axis=1 for rows along Y, axis=0 for cols along X)
    using a hybrid of:
    - High-percentile & mean intensity projection valleys (resistant to dead cells),
    - Directional Sobel edge energy,
    - Continuous dark-trough coalescing (prevents splitting wide Half-Cut central junction
      gaps into double Sobel peaks), and
    - Harmonic pitch interpolation (recovers missing dividers between adjacent dead cells).

    Returns:
      (dividers, gap_spans) where dividers are relative coordinates in [0, L]
      and gap_spans are (gap_start, gap_end) intervals for each divider.
    """
    length = active_sub.shape[0] if axis == 1 else active_sub.shape[1]
    if length < 32:
        return [], []

    # 1. Projection profiles along perpendicular axis
    mean_prof = np.mean(active_sub, axis=axis).astype(np.float32)
    p80_prof = np.percentile(active_sub, 80, axis=axis).astype(np.float32)
    # Blending p80 with mean ensures inter-cell gaps stay dark while rows/cols
    # containing several dead cells remain bright in the profile
    intensity_prof = 0.50 * mean_prof + 0.50 * p80_prof

    sigma = max(2.0, length * 0.003)
    if axis == 1:
        sobel = np.abs(cv2.Sobel(active_sub, cv2.CV_32F, 0, 1, ksize=3))
    else:
        sobel = np.abs(cv2.Sobel(active_sub, cv2.CV_32F, 1, 0, ksize=3))

    grad_prof = gaussian_filter1d(np.mean(sobel, axis=axis), sigma=sigma)
    valley_prof = gaussian_filter1d(np.max(intensity_prof) - intensity_prof, sigma=sigma)
    valley_p80 = gaussian_filter1d(np.max(p80_prof) - p80_prof, sigma=sigma)

    # Check if the 1D projection profile has sufficient valley contrast (SNR) above random noise
    if float(np.ptp(valley_p80)) < 10.0 and float(np.ptp(valley_prof)) < 10.0:
        return [], []

    norm_g = (grad_prof - np.min(grad_prof)) / (np.ptp(grad_prof) + 1e-6)
    norm_v = (valley_prof - np.min(valley_prof)) / (np.ptp(valley_prof) + 1e-6)
    norm_vp80 = (valley_p80 - np.min(valley_p80)) / (np.ptp(valley_p80) + 1e-6)

    combined = 0.45 * norm_g + 0.55 * norm_v

    default_max_cells = 26 if axis == 1 else 17
    min_dist = max(20 if axis == 1 else 35, length // default_max_cells)
    peaks, _ = find_peaks(combined, distance=min_dist, prominence=0.08)
    margin = int(length * 0.035)
    # Require that any valid divider is a genuine dark valley in the P80 profile
    # (rejects thin internal cell fingers and single-cell half-dark grid faults)
    raw_divs = [
        int(p) for p in peaks
        if margin < p < (length - margin) and norm_vp80[p] >= 0.22
    ]

    if not raw_divs:
        return [], []

    # Reject internal cell busbars: true inter-cell gaps exhibit deep P80 valleys
    # (norm_vp80 >= 0.70), whereas internal cell busbars are significantly shallower.
    strong_divs = [d for d in raw_divs if norm_vp80[d] >= 0.70]
    if len(strong_divs) >= 3:
        s_pts = [0] + strong_divs + [length]
        s_diffs = np.diff(s_pts).astype(np.float32)
        s_cv = float(np.std(s_diffs) / (np.mean(s_diffs) + 1e-6))
        if s_cv < 0.28:
            min_valid_vp80 = 0.65 * float(np.median([norm_vp80[d] for d in strong_divs]))
            raw_divs = [d for d in raw_divs if norm_vp80[d] >= min_valid_vp80]

    # 2. Coalesce double-peaks caused by wide Half-Cut central junction gaps
    # If two adjacent candidate dividers bound a continuous dark trough (no bright cell body
    # between them), merge them into a single wide-gap divider
    dark_thresh = float(np.min(p80_prof) + 0.40 * np.ptp(p80_prof))
    coalesced_divs: List[int] = []
    coalesced_spans: List[Tuple[int, int]] = []

    i = 0
    while i < len(raw_divs):
        d_curr = raw_divs[i]
        # Measure local dark trough span around d_curr
        g_left = d_curr
        while g_left > 0 and p80_prof[g_left - 1] <= dark_thresh:
            g_left -= 1
        g_right = d_curr
        while g_right < length - 1 and p80_prof[g_right + 1] <= dark_thresh:
            g_right += 1

        if i + 1 < len(raw_divs):
            d_next = raw_divs[i + 1]
            span_between = p80_prof[d_curr:d_next + 1]
            # Check if the region between d_curr and d_next is continuously dark (a single wide gap)
            if len(span_between) > 0 and float(np.median(span_between)) <= dark_thresh * 1.15:
                merged_center = int(round(0.5 * (d_curr + d_next)))
                g_right_next = d_next
                while g_right_next < length - 1 and p80_prof[g_right_next + 1] <= dark_thresh:
                    g_right_next += 1
                coalesced_divs.append(merged_center)
                coalesced_spans.append((min(g_left, d_curr), max(g_right_next, d_next)))
                i += 2
                continue

        coalesced_divs.append(d_curr)
        coalesced_spans.append((g_left, g_right))
        i += 1

    # 3. Harmonic Pitch Regularization: interpolate missing dividers if adjacent dead cells
    # suppressed a valley, or remove spurious internal finger peaks
    if len(coalesced_divs) >= 2:
        pts_all = [0] + coalesced_divs + [length]
        diffs = np.diff(pts_all).astype(np.float32)
        median_pitch = float(np.median(diffs))

        if median_pitch > min_dist:
            refined_divs: List[int] = []
            refined_spans: List[Tuple[int, int]] = []
            prev_pos = 0
            for d_pos, d_span in zip(coalesced_divs, coalesced_spans):
                step = d_pos - prev_pos
                # If a gap is ~2x or ~3x the median cell pitch, interpolate the missing divider(s)
                n_missing = int(round(step / median_pitch)) - 1
                if 1 <= n_missing <= 2 and abs(step - (n_missing + 1) * median_pitch) <= 0.28 * median_pitch:
                    for m_idx in range(1, n_missing + 1):
                        interp_d = int(round(prev_pos + m_idx * (step / float(n_missing + 1))))
                        refined_divs.append(interp_d)
                        refined_spans.append((max(0, interp_d - 2), min(length - 1, interp_d + 2)))
                refined_divs.append(d_pos)
                refined_spans.append(d_span)
                prev_pos = d_pos

            # Also check trailing interval from last divider to `length`
            tail_step = length - prev_pos
            n_tail_missing = int(round(tail_step / median_pitch)) - 1
            if 1 <= n_tail_missing <= 2 and abs(tail_step - (n_tail_missing + 1) * median_pitch) <= 0.28 * median_pitch:
                for m_idx in range(1, n_tail_missing + 1):
                    interp_d = int(round(prev_pos + m_idx * (tail_step / float(n_tail_missing + 1))))
                    refined_divs.append(interp_d)
                    refined_spans.append((max(0, interp_d - 2), min(length - 1, interp_d + 2)))

            coalesced_divs = refined_divs
            coalesced_spans = refined_spans

    return coalesced_divs, coalesced_spans


def _locate_active_area(rectified_img: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Identifies the active cell matrix bounding region [y_start, y_end, x_start, x_end]
    using high-percentile (P88) + mean intensity projections so that border rows/columns
    with up to 85% dead/dark cells are still accurately included.
    """
    h, w = rectified_img.shape[:2]

    # Blend 88th percentile with mean: immune to up to 85% dead cells in a border row/col
    # while rejecting localized small reflection specks on the outer frame
    proj_y_mean = np.mean(rectified_img, axis=1)
    proj_y_p88 = np.percentile(rectified_img, 88, axis=1)
    prof_y = 0.40 * proj_y_mean + 0.60 * proj_y_p88
    th_y = (np.max(prof_y) + np.min(prof_y)) * 0.35
    act_y = np.where(prof_y > th_y)[0]
    y_start, y_end = (int(act_y[0]), int(act_y[-1])) if len(act_y) else (0, h - 1)

    proj_x_mean = np.mean(rectified_img, axis=0)
    proj_x_p88 = np.percentile(rectified_img, 88, axis=0)
    prof_x = 0.40 * proj_x_mean + 0.60 * proj_x_p88
    th_x = (np.max(prof_x) + np.min(prof_x)) * 0.35
    act_x = np.where(prof_x > th_x)[0]
    x_start, x_end = (int(act_x[0]), int(act_x[-1])) if len(act_x) else (0, w - 1)

    return y_start, y_end, x_start, x_end


def _recover_dark_border_bands(
    bounds: List[int],
    full_length: int
) -> List[int]:
    """
    Checks margin symmetry of the detected grid bounds `[b_0, b_1, ..., b_k]` inside
    `[0, full_length]`. If an entire border row or column of cells is 100% dead/dark,
    one margin (`b_0` or `full_length - b_k`) will be larger than the opposite frame
    margin by an integer multiple of the median cell pitch `T`.
    Prepends or appends the missing dark border row/col bounds automatically.
    """
    if len(bounds) < 3:
        return bounds

    pitches = np.diff(bounds).astype(np.float32)
    median_pitch = float(np.median(pitches))
    if median_pitch < 15.0:
        return bounds

    start_margin = bounds[0]
    end_margin = full_length - bounds[-1]
    updated = list(bounds)

    # Check if start_margin hides 1 or 2 completely dark border rows/cols compared to end_margin
    if start_margin - end_margin >= 0.78 * median_pitch:
        n_extra = int(round((start_margin - end_margin) / median_pitch))
        if 1 <= n_extra <= 2:
            for _ in range(n_extra):
                new_b0 = max(end_margin, int(round(updated[0] - median_pitch)))
                if updated[0] - new_b0 >= 0.70 * median_pitch:
                    updated.insert(0, new_b0)

    # Symmetrically check if end_margin hides 1 or 2 completely dark border rows/cols
    if end_margin - start_margin >= 0.78 * median_pitch:
        n_extra = int(round((end_margin - start_margin) / median_pitch))
        if 1 <= n_extra <= 2:
            for _ in range(n_extra):
                new_end = min(full_length - start_margin, int(round(updated[-1] + median_pitch)))
                if new_end - updated[-1] >= 0.70 * median_pitch:
                    updated.append(new_end)

    return updated


def detect_grid_topology(rectified_img: np.ndarray) -> Tuple[int, int, List[int], List[int]]:
    """
    Computes horizontal and vertical projection profiles to dynamically detect
    rows (R) and columns (C), along with precise pixel divider boundaries.
    Supports both Full-Cell (e.g. 6x10, 6x12) and Half-Cut / Split-Cell (e.g. 12x10, 12x12)
    modules, recovers dark peripheral rows/columns, validates cell count plausibility,
    and falls back to aspect-ratio geometry under extreme noise.
    """
    if rectified_img is None or rectified_img.size == 0:
        raise ValueError("Cannot detect grid topology on an empty image.")

    if len(rectified_img.shape) == 3:
        rectified_img = cv2.cvtColor(rectified_img, cv2.COLOR_BGR2GRAY)

    h, w = rectified_img.shape[:2]

    # 1. Identify active luminescent matrix area (resilient to dark border cells)
    y_start, y_end, x_start, x_end = _locate_active_area(rectified_img)
    active_sub = rectified_img[y_start:y_end + 1, x_start:x_end + 1]
    ah, aw = active_sub.shape[:2]

    if ah < 32 or aw < 32:
        return _geometric_grid_fallback(h, w, 0, h - 1, 0, w - 1)

    img_ar = float(w) / float(max(1, h))
    active_ar = float(aw) / float(max(1, ah))
    is_square_like = (0.82 <= img_ar <= 1.22) or (0.82 <= active_ar <= 1.22)

    # 2. Dynamic 1D projection valley & edge detection along Y (rows) and X (cols)
    row_divs, _ = _detect_1d_dividers(active_sub, axis=1, min_cells=4, max_cells=26)
    col_divs, _ = _detect_1d_dividers(active_sub, axis=0, min_cells=4, max_cells=26)

    y_bounds = [y_start] + [y_start + d for d in row_divs] + [y_end + 1]
    x_bounds = [x_start] + [x_start + d for d in col_divs] + [x_end + 1]

    # 3. Recover 100%-dead border rows or columns via frame-margin symmetry
    y_bounds = _recover_dark_border_bands(y_bounds, h)
    x_bounds = _recover_dark_border_bands(x_bounds, w)

    rows = len(y_bounds) - 1
    cols = len(x_bounds) - 1

    # Single-Cell (1x1) EL Image Detection:
    # When an isolated square/near-square cell image (e.g., 300x300 from ELPV/amostra) is uploaded,
    # it lacks a valid commercial multi-cell grid topology. Return 1x1 directly instead of slicing into 6x10.
    multi_cell_families = PLAUSIBLE_CELL_COUNTS - {1}
    if is_square_like and (
        max(h, w) <= 360
        or rows < 4
        or cols < 4
        or (rows * cols) not in multi_cell_families
    ):
        return 1, 1, [0, h], [0, w]

    # 4. Integrity & Plausibility Validation + Geometric Fallback on Severe Noise
    mean_cell_h = float(y_bounds[-1] - y_bounds[0]) / max(1, rows)
    mean_cell_w = float(x_bounds[-1] - x_bounds[0]) / max(1, cols)
    cell_ar = mean_cell_w / max(1.0, mean_cell_h)

    # Trigger geometric fallback if grid resolution failed or produced impossible cell aspect ratio
    if (
        rows < 4
        or cols < 4
        or rows > 26
        or cols > 26
        or cell_ar < 0.25
        or cell_ar > 4.0
    ):
        return _geometric_grid_fallback(ah, aw, y_start, y_end, x_start, x_end)

    total_cells = rows * cols
    if total_cells not in PLAUSIBLE_CELL_COUNTS:
        logger.warning(
            "Grid plausibility warning: detected %dx%d (%d cells), which is outside "
            "standard commercial PV families (60, 72, 120, 144).",
            rows, cols, total_cells
        )

    return rows, cols, y_bounds, x_bounds


def segment_panel_cells(
    rectified_img: np.ndarray,
    y_bounds: Optional[List[int]] = None,
    x_bounds: Optional[List[int]] = None
) -> List[Dict[str, Any]]:
    """
    Dynamically slices the rectified photovoltaic module image into individual cells
    using projection-profile divider bounds (handling wide Half-Cut central busbar gaps
    by trimming excess dark gap margins) and returns spatial traceability metadata:

        {
            "cell_id": int,
            "row": int,
            "col": int,
            "bbox": [x, y, w, h],
            "image": np.ndarray (uint8)
        }
    """
    if rectified_img is None or rectified_img.size == 0:
        raise ValueError("Cannot segment cells from an empty image.")

    if len(rectified_img.shape) == 3:
        rectified_img = cv2.cvtColor(rectified_img, cv2.COLOR_BGR2GRAY)

    if y_bounds is None or x_bounds is None:
        _, _, y_bounds, x_bounds = detect_grid_topology(rectified_img)

    rows = len(y_bounds) - 1
    cols = len(x_bounds) - 1

    # Detect if there is a wide central half-cut gap along Y or X so we can trim
    # the extra dark junction-box strip from adjacent cell crops
    y_trim_top = [0] * rows
    y_trim_bot = [0] * rows
    if rows >= 6 and rows % 2 == 0:
        mid_r = rows // 2
        center_div_y = y_bounds[mid_r]
        band_y0 = max(0, center_div_y - 25)
        band_y1 = min(rectified_img.shape[0], center_div_y + 25)
        prof_y = np.mean(rectified_img[band_y0:band_y1, x_bounds[0]:x_bounds[-1]], axis=1)
        if prof_y.size > 0:
            dark_th = float(np.min(prof_y) + 0.35 * np.ptp(prof_y))
            dark_idx = np.where(prof_y <= dark_th)[0]
            if len(dark_idx) >= 10:
                # Wide central half-cut gap detected
                half_gap = max(0, (int(dark_idx[-1] - dark_idx[0]) - 4) // 2)
                cell_h_above = y_bounds[mid_r] - y_bounds[mid_r - 1]
                cell_h_below = y_bounds[mid_r + 1] - y_bounds[mid_r]
                y_trim_bot[mid_r - 1] = min(half_gap, int(cell_h_above * 0.22))
                y_trim_top[mid_r] = min(half_gap, int(cell_h_below * 0.22))

    segmented_cells: List[Dict[str, Any]] = []
    for r in range(rows):
        y0 = int(y_bounds[r] + y_trim_top[r])
        y1 = int(y_bounds[r + 1] - y_trim_bot[r])
        if y1 <= y0:
            y0, y1 = int(y_bounds[r]), int(y_bounds[r + 1])

        for c in range(cols):
            x0 = int(x_bounds[c])
            x1 = int(x_bounds[c + 1])
            cell_id = r * cols + c
            w = int(x1 - x0)
            h = int(y1 - y0)
            cell_crop = np.ascontiguousarray(rectified_img[y0:y1, x0:x1], dtype=np.uint8)

            segmented_cells.append({
                "cell_id": cell_id,
                "row": r,
                "col": c,
                "bbox": [x0, y0, w, h],
                "image": cell_crop,
            })

    return segmented_cells


def preprocess_el_module(image_input: Union[bytes, np.ndarray]) -> Dict[str, Any]:
    """
    End-to-end modular preprocessing pipeline for electroluminescence (EL) PV panels:
    1. Decodes or validates grayscale uint8 raster.
    2. Applies CLAHE local contrast normalization & perspective homography rectification.
    3. Dynamically segments the cell grid (Full-cell or Half-cut) with fallback safety.
    4. Returns rectified raster, topology metadata, and spatially indexed cell crops.
    """
    if isinstance(image_input, (bytes, bytearray)):
        if len(image_input) < 32:
            raise ValueError("Empty or corrupted image byte stream.")
        np_buffer = np.frombuffer(image_input, dtype=np.uint8)
        gray_img = cv2.imdecode(np_buffer, cv2.IMREAD_GRAYSCALE)
        if gray_img is None or gray_img.size == 0:
            raise ValueError("OpenCV failed to decode image byte stream.")
    elif isinstance(image_input, np.ndarray):
        if image_input.size == 0:
            raise ValueError("Empty numpy array provided to preprocess_el_module.")
        gray_img = (
            cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
            if len(image_input.shape) == 3
            else image_input.copy()
        )
        if gray_img.dtype != np.uint8:
            gray_img = np.clip(gray_img, 0, 255).astype(np.uint8)
    else:
        raise TypeError("image_input must be bytes or numpy.ndarray.")

    rectified = normalize_perspective(gray_img)
    rows, cols, y_bounds, x_bounds = detect_grid_topology(rectified)
    cells = segment_panel_cells(rectified, y_bounds=y_bounds, x_bounds=x_bounds)
    total_cells = rows * cols

    mean_cell_h = float(y_bounds[-1] - y_bounds[0]) / max(1, rows)
    mean_cell_w = float(x_bounds[-1] - x_bounds[0]) / max(1, cols)
    cell_aspect = mean_cell_w / max(1.0, mean_cell_h)
    is_half_cut = (
        total_cells >= 108
        or rows >= 12
        or cols >= 20
        or cell_aspect >= 1.45
        or cell_aspect <= 0.68
    )

    return {
        "rectified_image": rectified,
        "rows": rows,
        "cols": cols,
        "total_cells": total_cells,
        "panel_type": "half-cut" if is_half_cut else "full-cell",
        "is_plausible_topology": total_cells in PLAUSIBLE_CELL_COUNTS,
        "x_bounds": x_bounds,
        "y_bounds": y_bounds,
        "cells": cells,
    }


def inspect_cell_region(
    cell_img: np.ndarray,
    module_median_int: float,
    module_std_int: float
) -> Tuple[bool, DefectType, float]:
    """
    Analyzes an in-memory cropped cell for electroluminescence anomalies:
    intensity drops, hot-spots, micro-cracks, delamination, and grid/busbar faults.
    Returns: (is_defective, defect_type, confidence)
    """
    ch, cw = cell_img.shape[:2]
    if ch < 4 or cw < 4:
        return False, DefectType.NONE, 0.95

    # Crop inner active area to avoid divider borders & frame shadows (10% border margin)
    my = max(2, int(ch * 0.10))
    mx = max(2, int(cw * 0.10))
    inner = cell_img[my:ch - my, mx:cw - mx]
    if inner.size == 0:
        inner = cell_img

    cell_mean = float(np.mean(inner))
    cell_std = float(np.std(inner))
    cell_min = float(np.min(inner))
    cell_max = float(np.max(inner))

    # Reference ratio compared to healthy module luminescence
    ref_int = max(module_median_int, 20.0)
    rel_brightness = cell_mean / ref_int

    # 1. DEAD CELL: Complete disconnection or inactive crystal (< 40% of module baseline)
    if rel_brightness < 0.40:
        confidence = float(min(0.99, max(0.88, 1.0 - rel_brightness * 0.7)))
        return True, DefectType.DEAD, round(confidence, 2)

    # 2. HOT-SPOT: Localized current crowding / shunting causing abnormal high luminescence
    if rel_brightness > 1.25 or (cell_max > 235 and (cell_max - cell_mean) > 35):
        confidence = float(min(0.98, max(0.82, 0.75 + (cell_max / 255.0) * 0.2)))
        return True, DefectType.HOT_SPOT, round(confidence, 2)

    # 3. GRID-FAULT / BUSBAR FAULT:
    # Asymmetry where half of the cell (or a distinct quadrant) is deactivated due to busbar interruption
    mid_x = inner.shape[1] // 2
    left_half = inner[:, :mid_x]
    right_half = inner[:, mid_x:]
    mean_left = float(np.mean(left_half)) if left_half.size else cell_mean
    mean_right = float(np.mean(right_half)) if right_half.size else cell_mean
    asymmetry_h = abs(mean_left - mean_right) / (max(mean_left, mean_right) + 1e-5)

    mid_y = inner.shape[0] // 2
    top_half = inner[:mid_y, :]
    bot_half = inner[mid_y:, :]
    mean_top = float(np.mean(top_half)) if top_half.size else cell_mean
    mean_bot = float(np.mean(bot_half)) if bot_half.size else cell_mean
    asymmetry_v = abs(mean_top - mean_bot) / (max(mean_top, mean_bot) + 1e-5)

    if max(asymmetry_h, asymmetry_v) > 0.42:
        confidence = float(min(0.96, max(0.75, 0.70 + max(asymmetry_h, asymmetry_v) * 0.35)))
        return True, DefectType.GRID_FAULT, round(confidence, 2)

    # 4. MICRO-CRACK:
    # High-frequency dark lines traversing the crystalline silicon matrix
    laplacian_var = float(cv2.Laplacian(inner, cv2.CV_32F).var())
    crack_thresh = max(5.0, cell_mean - 1.8 * cell_std)
    dark_mask = (inner < crack_thresh).astype(np.uint8) * 255
    kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dark_lines = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel_line)
    crack_pixel_ratio = float(np.count_nonzero(dark_lines)) / float(inner.size)

    if (laplacian_var > 350.0 and crack_pixel_ratio > 0.015) or (crack_pixel_ratio > 0.025 and cell_std > 18.0):
        confidence = float(min(0.97, max(0.68, 0.65 + crack_pixel_ratio * 4.0)))
        return True, DefectType.MICRO_CRACK, round(confidence, 2)

    # 5. DELAMINATION:
    # Moisture ingress / encapsulant decay causing hazy contrast drops at edges/corners
    corner_size = max(2, int(min(inner.shape) * 0.25))
    c_tl = np.mean(inner[:corner_size, :corner_size])
    c_tr = np.mean(inner[:corner_size, -corner_size:])
    c_bl = np.mean(inner[-corner_size:, :corner_size])
    c_br = np.mean(inner[-corner_size:, -corner_size:])
    min_corner = min(c_tl, c_tr, c_bl, c_br)
    corner_drop = (cell_mean - min_corner) / (cell_mean + 1e-5)

    if corner_drop > 0.35 and rel_brightness < 0.90:
        confidence = float(min(0.94, max(0.65, 0.60 + corner_drop * 0.5)))
        return True, DefectType.DELAMINATION, round(confidence, 2)

    # 6. NOMINAL / HEALTHY CELL
    # Homogeneous luminescence with expected busbar pattern
    confidence = float(min(0.99, max(0.85, 1.0 - abs(rel_brightness - 1.0) * 0.25)))
    return False, DefectType.NONE, round(confidence, 2)


# -----------------------------------------------------------------------------
# YOLOv8-cls Classification Engine (best.pt) & Model Hash Verification
# -----------------------------------------------------------------------------
_YOLO_CLASSIFIER: Optional[YOLO] = None
_MODEL_SHA256: Optional[str] = None


def resolve_model_path(weights_path: Optional[str] = None) -> str:
    """
    Localiza o arquivo de pesos 'best.pt' nos diretórios candidatos padrão.
    """
    target_path = weights_path or os.getenv("MODEL_PATH")
    if not target_path or not os.path.exists(target_path):
        candidate_paths = [
            Path(__file__).resolve().parent / "models" / "best.pt",
            Path(__file__).resolve().parent.parent / "models" / "best.pt",
            Path(__file__).resolve().parent.parent / "weights" / "best.pt",
            Path(__file__).resolve().parent.parent / "model" / "best.pt",
            Path("models/best.pt"),
            Path("weights/best.pt"),
            Path("model/best.pt"),
        ]
        for p in candidate_paths:
            if p.is_file():
                target_path = str(p)
                break

    if not target_path or not os.path.exists(target_path):
        raise FileNotFoundError(
            "Arquivo de pesos 'best.pt' não encontrado nos caminhos padrão."
        )
    return target_path


def get_model_sha256(weights_path: Optional[str] = None) -> str:
    """
    Calcula e mantém em cache o hash SHA-256 do arquivo de pesos 'best.pt'
    para verificação de integridade e consistência entre workers distribuídos.
    """
    global _MODEL_SHA256
    if _MODEL_SHA256 is not None and weights_path is None:
        return _MODEL_SHA256

    try:
        path = resolve_model_path(weights_path)
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha.update(chunk)
        digest = sha.hexdigest()
        if weights_path is None:
            _MODEL_SHA256 = digest
        return digest
    except Exception as exc:
        logger.warning(f"Não foi possível calcular SHA-256 do modelo: {exc}")
        return "heuristic-fallback"


def get_yolo_classifier(weights_path: Optional[str] = None) -> YOLO:
    """
    Carrega e mantém em cache o modelo YOLOv8-cls (best.pt).
    Busca no caminho explícito, variável de ambiente MODEL_PATH ou diretórios padrão.
    """
    global _YOLO_CLASSIFIER
    if _YOLO_CLASSIFIER is not None and weights_path is None:
        return _YOLO_CLASSIFIER

    target_path = resolve_model_path(weights_path)
    logger.info(f"Carregando modelo YOLOv8-cls a partir de: {target_path}")
    _YOLO_CLASSIFIER = YOLO(target_path)
    return _YOLO_CLASSIFIER


def _trim_cell_gap_margins(cell_crop: np.ndarray, trim_ratio: float = 0.05) -> np.ndarray:
    """
    Remove a margem periférica correspondente às frestas inter-células (cell gaps)
    e sombras de moldura (~4 a 5 pixels), garantindo que apenas a área útil
    do wafer de silício seja enviada ao classificador YOLOv8-cls.
    """
    h, w = cell_crop.shape[:2]
    if h < 16 or w < 16:
        return cell_crop
    ty = 4 if h <= 100 else max(4, min(6, int(round(h * trim_ratio))))
    tx = 4 if w <= 100 else max(4, min(6, int(round(w * trim_ratio))))
    if (h - 2 * ty) < 8 or (w - 2 * tx) < 8:
        return cell_crop
    return np.ascontiguousarray(cell_crop[ty:h - ty, tx:w - tx])


def _format_cell_crop_bgr(cell_crop: np.ndarray, trim_border: bool = True) -> np.ndarray:
    """
    Garante que o crop da célula tenha as margens de fresta aparadas, possua 3 canais (BGR)
    e tipo uint8 para entrada no YOLOv8-cls.
    """
    if trim_border:
        cell_crop = _trim_cell_gap_margins(cell_crop)
    if cell_crop.dtype != np.uint8:
        cell_crop = np.clip(cell_crop, 0, 255).astype(np.uint8)
    if len(cell_crop.shape) == 2:
        return cv2.cvtColor(cell_crop, cv2.COLOR_GRAY2BGR)
    if cell_crop.shape[2] == 4:
        return cv2.cvtColor(cell_crop, cv2.COLOR_BGRA2BGR)
    return cell_crop


def classify_cell_yolo(
    cell_crop: np.ndarray,
    model: Optional[YOLO] = None
) -> Tuple[bool, DefectType, float]:
    """
    Classifica uma célula solar cortada (crop) utilizando a API de classificação YOLOv8-cls.
    Extrai:
      - Predição da classe: results[0].names[results[0].probs.top1]
      - Confiança: float(results[0].probs.top1conf)

    Retorna:
      Tuple[bool, DefectType, float]: (is_defective, defect_type, confidence)
    """
    if model is None:
        model = get_yolo_classifier()

    cell_input = _format_cell_crop_bgr(cell_crop)

    # Inferência Ultralytics YOLOv8-cls
    results = model(cell_input, verbose=False)
    probs = results[0].probs
    top1_idx = probs.top1
    pred_label = str(results[0].names[top1_idx]).strip().lower()
    confidence = float(probs.top1conf)

    # Identificação da classe: 'boa' vs 'defeituosa'
    is_defective = (pred_label in ["defeituosa", "defeito", "defect", "defective"])
    defect_type = DefectType.DEFEITUOSA if is_defective else DefectType.NONE

    return is_defective, defect_type, round(confidence, 4)


def classify_cells_batch_yolo(
    cell_crops: List[np.ndarray],
    model: Optional[YOLO] = None
) -> List[Tuple[bool, DefectType, float]]:
    """
    Realiza inferência vetorizada em lote (Batch Inference) sobre todas as células
    recortadas de um módulo fotovoltaico em uma única chamada ao motor YOLOv8-cls,
    reduzindo significativamente o overhead de invocação por célula em sistemas distribuídos.
    """
    if not cell_crops:
        return []

    if model is None:
        model = get_yolo_classifier()

    batch_inputs = [_format_cell_crop_bgr(crop) for crop in cell_crops]
    results = model(batch_inputs, verbose=False)

    predictions: List[Tuple[bool, DefectType, float]] = []
    for res in results:
        probs = res.probs
        top1_idx = probs.top1
        pred_label = str(res.names[top1_idx]).strip().lower()
        confidence = float(probs.top1conf)
        is_defective = (pred_label in ["defeituosa", "defeito", "defect", "defective"])
        defect_type = DefectType.DEFEITUOSA if is_defective else DefectType.NONE
        predictions.append((is_defective, defect_type, round(confidence, 4)))

    return predictions


def classify_single_cell_tta_yolo(
    cell_crop: np.ndarray,
    model: Optional[YOLO] = None,
    defect_threshold: float = 0.55
) -> Tuple[bool, DefectType, float]:
    """
    Executa inferência com Test-Time Augmentation (TTA: original + espelhamento
    horizontal + espelhamento vertical) para células unitárias (1x1).
    Consolida a probabilidade média da classe defeituosa, aumentando a robustez
    contra artefatos de orientação ou iluminação em imagens de células isoladas.
    """
    if model is None:
        model = get_yolo_classifier()

    orig_bgr = _format_cell_crop_bgr(cell_crop, trim_border=False)
    hflip_bgr = _format_cell_crop_bgr(cv2.flip(cell_crop, 1), trim_border=False)
    vflip_bgr = _format_cell_crop_bgr(cv2.flip(cell_crop, 0), trim_border=False)

    results = model([orig_bgr, hflip_bgr, vflip_bgr], verbose=False)
    def_probs: List[float] = []
    for res in results:
        probs = res.probs
        # Localiza o índice da classe defeituosa no dicionário de nomes do modelo
        def_idx = next(
            (k for k, v in res.names.items() if str(v).strip().lower() in ["defeituosa", "defeito", "defect", "defective"]),
            None
        )
        if def_idx is not None:
            def_probs.append(float(probs.data[def_idx].item()))
        else:
            top1_idx = probs.top1
            pred_label = str(res.names[top1_idx]).strip().lower()
            conf = float(probs.top1conf)
            def_probs.append(conf if pred_label in ["defeituosa", "defeito", "defect", "defective"] else (1.0 - conf))

    mean_def_prob = float(np.mean(def_probs)) if def_probs else 0.0
    is_defective = mean_def_prob >= defect_threshold
    defect_type = DefectType.DEFEITUOSA if is_defective else DefectType.NONE
    confidence = mean_def_prob if is_defective else (1.0 - mean_def_prob)
    return is_defective, defect_type, round(confidence, 4)


def draw_defect_annotations(
    image: np.ndarray,
    cells: List[CellInspection],
    highlight_color: Tuple[int, int, int] = (0, 0, 255),  # Vermelho BGR para defeituosas
    box_thickness: int = 2,
    draw_banner: bool = False
) -> np.ndarray:
    """
    Desenha retângulos visuais de destaque demarcando as células defeituosas
    e adiciona um overlay translúcido para rápida identificação visual.
    """
    annotated = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
    overlay = annotated.copy()

    defective_cells = [c for c in cells if c.is_defective and c.bbox]
    healthy_cells = [c for c in cells if not c.is_defective and c.bbox]

    # Contorno sutil semitransparente (#3a4454 / BGR 84, 68, 58) nas células sadias (PASS) em painéis multi-células
    if len(cells) > 1 and healthy_cells:
        grid_layer = annotated.copy()
        for cell in healthy_cells:
            bbox = cell.bbox
            cv2.rectangle(
                grid_layer,
                (bbox.x, bbox.y),
                (bbox.x + bbox.width, bbox.y + bbox.height),
                (84, 68, 58),
                1
            )
        cv2.addWeighted(grid_layer, 0.45, annotated, 0.55, 0, annotated)
        overlay = annotated.copy()

    # Preenchimento translúcido das células com defeito
    for cell in defective_cells:
        bbox = cell.bbox
        cv2.rectangle(
            overlay,
            (bbox.x, bbox.y),
            (bbox.x + bbox.width, bbox.y + bbox.height),
            highlight_color,
            -1
        )

    # Mescla o preenchimento com 25% de opacidade
    cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0, annotated)

    # Contorno sólido e rótulo de confiança
    for cell in defective_cells:
        bbox = cell.bbox
        x, y, w, h = bbox.x, bbox.y, bbox.width, bbox.height
        cv2.rectangle(annotated, (x, y), (x + w, y + h), highlight_color, box_thickness)

        label = f"{cell.confidence:.0%}"
        font_scale = max(0.35, min(w, h) / 130.0)
        cv2.putText(
            annotated,
            label,
            (x + 4, y + max(12, int(14 * (font_scale / 0.35)))),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            highlight_color,
            1,
            cv2.LINE_AA
        )

    if draw_banner:
        total = len(cells)
        defective = len(defective_cells)
        damage_pct = round((defective / float(total)) * 100.0, 1) if total > 0 else 0.0
        status_text = "REPROVADO (FAIL)" if damage_pct >= 5.0 else "APROVADO (PASS)"

        banner_h = 36
        banner = np.zeros((banner_h, annotated.shape[1], 3), dtype=np.uint8)
        header_text = f"Dano: {damage_pct:.1f}% ({defective}/{total} defeituosas) | Status: {status_text}"
        cv2.putText(banner, header_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        annotated = np.vstack([banner, annotated])

    return annotated


def generate_module_id(filename: str, fallback_idx: int = 1) -> str:
    """
    Generates or extracts a standardized module ID (e.g. MOD-2026-001).
    """
    match = re.search(r'(MOD-\d{4}-\d{3,4})', filename, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Check for scan_XX or panel_XX pattern
    match_seq = re.search(r'(?:scan|panel|module|scan_module)[_-](\d{1,4})', filename, re.IGNORECASE)
    if match_seq:
        num = int(match_seq.group(1))
        return f"MOD-2026-{num:03d}"

    # Use deterministic hash of filename to guarantee distinct IDs across different files
    h = (abs(hash(filename)) % 900) + 100
    return f"MOD-2026-{h:03d}"


def evaluate_module(
    image_bytes: bytes,
    filename: str,
    fallback_idx: int = 1,
    use_yolo: bool = True,
    model: Optional[YOLO] = None,
    node_id: Optional[str] = None
) -> ModuleResult:
    """
    Complete end-to-end computer vision processing pipeline for a single PV module:
    1. Memory decoding & validation
    2. Perspective normalization (CLAHE + Convex Hull 4-Line Homography Rectification)
    3. Dynamic grid topology detection (arbitrary R rows x C cols, Full-Cell & Half-Cut)
    4. Adaptive in-memory cell slicing with spatial traceability metadata
    5. Vectorized batch YOLOv8-cls classification per cell (best.pt)
    6. Module damage % computation and FAIL/PASS classification
    7. Visual reconstruction with defect bounding boxes
    """
    if not image_bytes or len(image_bytes) < 32:
        raise ValueError("Empty or corrupted image byte stream.")

    content_hash = hashlib.sha256(image_bytes).hexdigest()

    np_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(np_buffer, cv2.IMREAD_GRAYSCALE)
    if img is None or img.size == 0:
        raise ValueError(f"OpenCV failed to decode image '{filename}' as a valid graphical raster.")

    # 1. Perspective Normalization
    rectified = normalize_perspective(img)
    h, w = rectified.shape[:2]

    # 2. Dynamic Grid Topology Detection (R rows x C cols)
    rows, cols, y_bounds, x_bounds = detect_grid_topology(rectified)
    total_cells = rows * cols

    x_start = x_bounds[0]
    x_end = x_bounds[-1]
    y_start = y_bounds[0]
    y_end = y_bounds[-1]
    active_area = {
        "x": int(x_start),
        "y": int(y_start),
        "width": int(x_end - x_start),
        "height": int(y_end - y_start),
        "x_pct": round((x_start / float(w)) * 100.0, 2),
        "y_pct": round((y_start / float(h)) * 100.0, 2),
        "width_pct": round(((x_end - x_start) / float(w)) * 100.0, 2),
        "height_pct": round(((y_end - y_start) / float(h)) * 100.0, 2)
    }

    # Baseline statistics across active cells for fallback heuristic
    active_crop = rectified[y_start:y_end, x_start:x_end]
    if active_crop.size > 0:
        # Use upper-half luminescence to avoid bias when many border cells are dead
        bright_pixels = active_crop[active_crop > np.percentile(active_crop, 35)]
        module_median_int = float(np.median(bright_pixels)) if bright_pixels.size > 0 else float(np.median(rectified))
        module_std_int = float(np.std(active_crop))
    else:
        module_median_int = float(np.median(rectified))
        module_std_int = float(np.std(rectified))

    # 3. Adaptive Cell Slicing and Batch YOLOv8-cls Inspection
    cells: List[CellInspection] = []
    defective_count = 0
    yolo_model: Optional[YOLO] = model

    if use_yolo and yolo_model is None:
        try:
            yolo_model = get_yolo_classifier()
        except Exception as exc:
            logger.warning(
                f"YOLOv8-cls model could not be loaded ({exc}). Falling back to heuristic inspection."
            )
            yolo_model = None

    segmented_cells = segment_panel_cells(rectified, y_bounds=y_bounds, x_bounds=x_bounds)

    if yolo_model is not None and segmented_cells:
        if total_cells == 1:
            batch_preds = [
                classify_single_cell_tta_yolo(segmented_cells[0]["image"], model=yolo_model)
            ]
        else:
            raw_yolo_preds = classify_cells_batch_yolo(
                [item["image"] for item in segmented_cells],
                model=yolo_model
            )
            inner_stds = [
                float(np.std(_trim_cell_gap_margins(it["image"], trim_ratio=0.10)))
                for it in segmented_cells
            ]
            median_cell_std = float(np.median(inner_stds)) if inner_stds else 15.0

            batch_preds = []
            for item, (y_def, y_type, y_conf) in zip(segmented_cells, raw_yolo_preds):
                cell_img = item["image"]
                h_def, h_type, h_conf = inspect_cell_region(cell_img, module_median_int, module_std_int)
                inner = _trim_cell_gap_margins(cell_img, trim_ratio=0.10)
                c_mean = float(np.mean(inner)) if inner.size > 0 else float(np.mean(cell_img))
                c_std = float(np.std(inner)) if inner.size > 0 else float(np.std(cell_img))
                rel_b = c_mean / max(module_median_int, 20.0)

                if y_def:
                    # Cross-validate with radiometric & gradient consistency to suppress false alarms
                    # caused by perspective warp interpolation, half-cut aspect ratios, or borderline scores
                    if not h_def and c_std < 9.5 and (0.87 <= rel_b <= 1.13):
                        batch_preds.append((False, DefectType.NONE, h_conf))
                    elif not h_def and y_conf < 0.72 and (0.90 <= rel_b <= 1.10):
                        batch_preds.append((False, DefectType.NONE, h_conf))
                    else:
                        batch_preds.append((True, y_type, y_conf))
                else:
                    # Recover subtle & severe physical defects (dead cells, grid-faults, delamination, micro-cracks)
                    if h_def and (
                        h_type in (DefectType.DEAD, DefectType.GRID_FAULT, DefectType.DELAMINATION)
                        or rel_b < 0.84
                        or (median_cell_std < 13.5 and h_conf >= 0.65)
                        or (c_std < 17.5 and h_conf >= 0.65)
                    ):
                        batch_preds.append((True, DefectType.DEFEITUOSA, h_conf))
                    else:
                        batch_preds.append((False, DefectType.NONE, y_conf))
    else:
        batch_preds = [
            inspect_cell_region(item["image"], module_median_int, module_std_int)
            for item in segmented_cells
        ]

    for cell_meta, (is_defective, defect_type, confidence) in zip(segmented_cells, batch_preds):
        if is_defective:
            defective_count += 1

        idx = int(cell_meta["cell_id"])
        r = int(cell_meta["row"])
        c = int(cell_meta["col"])
        x0, y0, cell_w, cell_h = [int(v) for v in cell_meta["bbox"]]

        bbox = CellBoundingBox(
            x=x0,
            y=y0,
            width=cell_w,
            height=cell_h,
            x_pct=round((x0 / float(w)) * 100.0, 3),
            y_pct=round((y0 / float(h)) * 100.0, 3),
            width_pct=round((cell_w / float(w)) * 100.0, 3),
            height_pct=round((cell_h / float(h)) * 100.0, 3)
        )

        cells.append(
            CellInspection(
                index=idx,
                row=r,
                col=c,
                is_defective=is_defective,
                defect_type=defect_type,
                confidence=confidence,
                bbox=bbox
            )
        )

    # 4. Classification & Metrics
    module_damage_pct = round((defective_count / float(total_cells)) * 100.0, 1) if total_cells > 0 else 0.0
    status = "FAIL" if module_damage_pct >= 5.0 else "PASS"
    module_id = generate_module_id(filename, fallback_idx)
    resolved_node_id = node_id or os.getenv("NODE_ID", "worker-1")
    model_hash = get_model_sha256()

    # Reconstructed / visual raster with bounding boxes highlighting defective cells
    annotated_panel = draw_defect_annotations(rectified, cells, draw_banner=False)
    _, buf = cv2.imencode('.jpg', annotated_panel, [cv2.IMWRITE_JPEG_QUALITY, 85])
    rectified_image = f"data:image/jpeg;base64,{base64.b64encode(buf).decode('ascii')}"

    return ModuleResult(
        module_id=module_id,
        filename=filename,
        status=status,
        rows=rows,
        cols=cols,
        total_cells=total_cells,
        defective_cells_count=defective_count,
        module_damage_percentage=module_damage_pct,
        image_width=int(w),
        image_height=int(h),
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        active_area=active_area,
        rectified_image=rectified_image,
        processed_by_node=resolved_node_id,
        model_sha256=model_hash,
        content_sha256=content_hash,
        replicated=False,
        consistency_model="read-your-writes",
        cache_hit=False,
        cells=cells
    )
