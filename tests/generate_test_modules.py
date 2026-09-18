"""
Synthetic Electroluminescence (EL) PV Module Generator
======================================================
Generates realistic electroluminescence images of solar modules with:
- Configurable arbitrary grid topologies (6x10, 6x12, 12x10, etc.)
- Physical inter-cell busbars, ribbons, and dark division gaps
- Controllable defect injection: dead cells, micro-cracks, hot-spots, delamination, grid-faults
- Optional perspective tilt for homography rectification testing
"""

import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def create_synthetic_el_cell(
    width: int,
    height: int,
    base_luminescence: int = 175,
    defect_type: str = "none"
) -> np.ndarray:
    """
    Creates a single solar cell raster with realistic EL texture and defect signatures.
    """
    # Base emissive background with subtle silicon grain noise
    cell = np.full((height, width), base_luminescence, dtype=np.float32)
    noise = np.random.normal(0, 3.5, (height, width))
    cell = np.clip(cell + noise, 0, 255)

    # Add vertical finger / busbar texture (thin faint vertical stripes)
    n_fingers = max(4, width // 15)
    for i in range(1, n_fingers):
        fx = int(i * width / n_fingers)
        cell[:, max(0, fx-1):min(width, fx+1)] -= 15.0

    # Defect simulation
    if defect_type == "dead":
        # Completely disconnected or severely inactive cell
        cell = np.random.normal(25, 5, (height, width))

    elif defect_type == "hot-spot":
        # Intense localized current concentration / overheating
        cx, cy = width // 2, height // 2
        y, x = np.ogrid[:height, :width]
        dist_sq = (x - cx)**2 + (y - cy)**2
        radius_sq = (min(width, height) * 0.35)**2
        hot_mask = np.exp(-dist_sq / (2 * (radius_sq / 4)))
        cell = cell + hot_mask * 110.0

    elif defect_type == "grid-fault":
        # Busbar / finger interruption causing one side to be dark
        split_x = width // 2
        cell[:, split_x:] = cell[:, split_x:] * 0.20 + np.random.normal(15, 3, (height, width - split_x))

    elif defect_type == "micro-crack":
        # High-frequency branching dark fissure line
        pts = []
        curr_x, curr_y = int(width * 0.2), int(height * 0.25)
        for _ in range(8):
            pts.append((curr_x, curr_y))
            curr_x += int(np.random.randint(width // 10, width // 6))
            curr_y += int(np.random.randint(-height // 10, height // 10))
            curr_x = min(width - 5, max(5, curr_x))
            curr_y = min(height - 5, max(5, curr_y))
        for i in range(len(pts) - 1):
            cv2.line(cell, pts[i], pts[i+1], color=20.0, thickness=2)
            # Add micro-branch
            bx = pts[i][0] + np.random.randint(-5, 5)
            by = pts[i][1] + np.random.randint(10, 20)
            cv2.line(cell, pts[i], (bx, min(height-2, by)), color=25.0, thickness=1)

    elif defect_type == "delamination":
        # Moisture ingress causing contrast drop at edges/corners
        for y in range(height):
            for x in range(width):
                dist_corner = math.hypot(x, y)
                if dist_corner < min(width, height) * 0.7:
                    decay = (1.0 - (dist_corner / (min(width, height) * 0.7)))
                    cell[y, x] *= (1.0 - decay * 0.65)

    return np.clip(cell, 0, 255).astype(np.uint8)


def generate_module_image(
    rows: int,
    cols: int,
    cell_w: int = 70,
    cell_h: int = 70,
    gap_px: int = 6,
    outer_border_px: int = 24,
    defect_map: Optional[Dict[Tuple[int, int], str]] = None,
    apply_perspective_tilt: bool = False
) -> np.ndarray:
    """
    Assembles a full PV module image with R rows x C cols, physical gaps,
    outer frame, and specified defect locations.
    """
    if defect_map is None:
        defect_map = {}

    active_w = cols * cell_w + (cols - 1) * gap_px
    active_h = rows * cell_h + (rows - 1) * gap_px

    total_w = active_w + outer_border_px * 2
    total_h = active_h + outer_border_px * 2

    # Module background (dark anodized aluminum frame & gap backing)
    module_img = np.full((total_h, total_w), 18, dtype=np.uint8)

    # Place individual cells into the active area
    for r in range(rows):
        y_start = outer_border_px + r * (cell_h + gap_px)
        y_end = y_start + cell_h
        for c in range(cols):
            x_start = outer_border_px + c * (cell_w + gap_px)
            x_end = x_start + cell_w

            dtype = defect_map.get((r, c), "none")
            cell_raster = create_synthetic_el_cell(cell_w, cell_h, defect_type=dtype)
            module_img[y_start:y_end, x_start:x_end] = cell_raster

    # If perspective tilt is requested, place on canvas and apply 4-point projective warp
    if apply_perspective_tilt:
        pad = 60
        canvas_w = total_w + pad * 2
        canvas_h = total_h + pad * 2
        canvas = np.full((canvas_h, canvas_w), 5, dtype=np.uint8)
        canvas[pad:pad+total_h, pad:pad+total_w] = module_img

        src_pts = np.array([
            [pad, pad],
            [pad + total_w, pad],
            [pad + total_w, pad + total_h],
            [pad, pad + total_h]
        ], dtype=np.float32)

        # Mild trapezoidal tilt (5 degrees perspective angle)
        dst_pts = np.array([
            [pad + 30, pad + 20],
            [pad + total_w - 35, pad + 10],
            [pad + total_w - 10, pad + total_h - 15],
            [pad + 15, pad + total_h - 25]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        module_img = cv2.warpPerspective(canvas, M, (canvas_w, canvas_h))

    return module_img


def generate_test_dataset(output_dir: Path):
    """
    Generates a suite of test module images covering various topologies and defects.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Standard 60-cell module (6 rows x 10 cols) - PASS (1 single defect, < 5% damage)
    # Damage: 1 / 60 = 1.7% -> PASS
    img_6x10_pass = generate_module_image(
        rows=6, cols=10,
        defect_map={(2, 4): "hot-spot"}
    )
    cv2.imwrite(str(output_dir / "scan_module_6x10_pass.png"), img_6x10_pass)

    # 2. 72-cell commercial module (6 rows x 12 cols) - FAIL (multiple defects, > 5% damage)
    # Damage: 6 / 72 = 8.3% -> FAIL
    img_6x12_fail = generate_module_image(
        rows=6, cols=12,
        defect_map={
            (0, 1): "micro-crack",
            (1, 3): "dead",
            (2, 6): "grid-fault",
            (3, 8): "delamination",
            (4, 10): "hot-spot",
            (5, 2): "micro-crack"
        }
    )
    cv2.imwrite(str(output_dir / "scan_module_6x12_fail.png"), img_6x12_fail)

    # 3. Half-cut 120-cell module (12 rows x 10 cols) - FAIL (8 defects, 8/120 = 6.7%)
    img_12x10_fail = generate_module_image(
        rows=12, cols=10, cell_w=60, cell_h=35, gap_px=4,
        defect_map={
            (0, 0): "dead",
            (2, 3): "micro-crack",
            (4, 7): "grid-fault",
            (6, 1): "hot-spot",
            (7, 5): "dead",
            (9, 8): "delamination",
            (11, 2): "micro-crack",
            (11, 9): "grid-fault"
        }
    )
    cv2.imwrite(str(output_dir / "scan_module_12x10_fail.png"), img_12x10_fail)

    # 4. Standard 60-cell module with perspective tilt (homography test)
    img_tilt = generate_module_image(
        rows=6, cols=10,
        defect_map={(1, 2): "micro-crack", (3, 5): "dead"},
        apply_perspective_tilt=True
    )
    cv2.imwrite(str(output_dir / "scan_module_tilted_6x10.png"), img_tilt)

    print(f"Generated 4 test PV module images in: {output_dir}")


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "fixtures"
    generate_test_dataset(out)
