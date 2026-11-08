"""
Unit Tests for Computer Vision & Signal Processing Engine (Node 2)
==================================================================
Covers:
- Legacy & standard module fixtures (6x10 PASS, 6x12 FAIL, 12x10 Half-Cut, Tilted)
- Mild rotations (+/- 10 degrees) with bench noise & frame reflections
- Trapezoidal perspective distortion (non-orthogonal camera homography)
- Panels with dark/dead border and corner cells (including full dark border row)
- Standard Full-Cell vs. Half-Cut (Split-Cell with wide central junction gap)
- Spatial traceability metadata contract: {"cell_id", "row", "col", "bbox", "image"}
- Geometric aspect-ratio fallback under severe noise & plausibility logging
- Strict preservation of YOLOv8-cls (best.pt) model input contract
"""

import logging
import os
from pathlib import Path
import sys
from typing import Dict, Optional, Tuple
import unittest

import cv2
import numpy as np

# Add node2_worker and tests to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "node2_worker"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cv_engine import (
    DefectType,
    _trim_cell_gap_margins,
    apply_clahe_normalization,
    classify_cell_yolo,
    classify_cells_batch_yolo,
    classify_single_cell_tta_yolo,
    detect_grid_topology,
    detect_panel_quadrilateral,
    draw_defect_annotations,
    evaluate_module,
    get_model_sha256,
    get_yolo_classifier,
    inspect_cell_region,
    normalize_perspective,
    preprocess_el_module,
    segment_panel_cells,
)
from generate_test_modules import create_synthetic_el_cell, generate_module_image


def _generate_half_cut_with_central_gap(
    rows: int = 12,
    cols: int = 10,
    cell_w: int = 60,
    cell_h: int = 35,
    gap_px: int = 4,
    central_gap_px: int = 22,
    outer_border_px: int = 24,
    defect_map: Optional[Dict[Tuple[int, int], str]] = None,
) -> np.ndarray:
    """
    Helper to synthesize a realistic Half-Cut (Split-Cell) PV module with a wider
    central horizontal gap where the central junction box / inter-string busbar sits.
    """
    if defect_map is None:
        defect_map = {}

    mid_row = rows // 2
    active_w = cols * cell_w + (cols - 1) * gap_px
    active_h = rows * cell_h + (rows - 2) * gap_px + central_gap_px

    total_w = active_w + outer_border_px * 2
    total_h = active_h + outer_border_px * 2

    module_img = np.full((total_h, total_w), 18, dtype=np.uint8)

    y_cursor = outer_border_px
    for r in range(rows):
        y_start = y_cursor
        y_end = y_start + cell_h
        for c in range(cols):
            x_start = outer_border_px + c * (cell_w + gap_px)
            x_end = x_start + cell_w
            dtype = defect_map.get((r, c), "none")
            module_img[y_start:y_end, x_start:x_end] = create_synthetic_el_cell(
                cell_w, cell_h, defect_type=dtype
            )
        y_cursor = y_end + (central_gap_px if (r + 1) == mid_row else gap_px)

    return module_img


def _rotate_module_on_canvas(module_img: np.ndarray, angle_deg: float, add_bench_noise: bool = True) -> np.ndarray:
    """
    Places a PV module on a dark test-bench canvas and rotates it by `angle_deg` degrees,
    optionally injecting small background bench noise blobs.
    """
    h, w = module_img.shape[:2]
    pad = int(max(h, w) * 0.25)
    canvas_h = h + pad * 2
    canvas_w = w + pad * 2
    canvas = np.full((canvas_h, canvas_w), 5, dtype=np.uint8)
    canvas[pad:pad + h, pad:pad + w] = module_img

    center = (canvas_w / 2.0, canvas_h / 2.0)
    rot_mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(
        canvas,
        rot_mat,
        (canvas_w, canvas_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=5,
    )

    if add_bench_noise:
        # Inject isolated bench reflection specks in the outer corners
        rotated[10:14, 10:15] = 190
        rotated[canvas_h - 15:canvas_h - 11, canvas_w - 16:canvas_w - 11] = 210

    return rotated


class TestCVEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures_dir = Path(__file__).resolve().parent / "fixtures"

    def test_corrupted_image_handling(self):
        """Verify that invalid/corrupted bytes raise ValueError cleanly."""
        with self.assertRaises(ValueError):
            evaluate_module(b"not an image", filename="corrupt.png")

        with self.assertRaises(ValueError):
            evaluate_module(b"", filename="empty.png")

    def test_evaluate_6x10_pass_module(self):
        """Verify dynamic detection and evaluation of 6x10 module (PASS)."""
        img_path = self.fixtures_dir / "scan_module_6x10_pass.png"
        self.assertTrue(img_path.is_file(), f"Fixture missing: {img_path}")

        with open(img_path, "rb") as f:
            data = f.read()

        result = evaluate_module(
            data,
            filename="scan_module_6x10_pass.png",
            use_yolo=False,
            node_id="worker-1"
        )

        self.assertEqual(result.rows, 6)
        self.assertEqual(result.cols, 10)
        self.assertEqual(result.total_cells, 60)
        self.assertEqual(len(result.cells), 60)
        self.assertEqual(result.processed_by_node, "worker-1")
        self.assertIsNotNone(result.model_sha256)
        self.assertEqual(len(result.model_sha256), 64)
        self.assertIsNotNone(result.content_sha256)
        self.assertEqual(len(result.content_sha256), 64)
        self.assertEqual(result.consistency_model, "read-your-writes")
        # Verify strict enum types
        valid_enums = {e.value for e in DefectType}
        for cell in result.cells:
            self.assertIn(cell.defect_type.value, valid_enums)
            self.assertGreaterEqual(cell.confidence, 0.0)
            self.assertLessEqual(cell.confidence, 1.0)

        # Damage should be under 5% -> PASS
        self.assertLess(result.module_damage_percentage, 5.0)
        self.assertEqual(result.status, "PASS")

    def test_evaluate_6x12_fail_module(self):
        """Verify dynamic detection and evaluation of 6x12 module (FAIL)."""
        img_path = self.fixtures_dir / "scan_module_6x12_fail.png"
        self.assertTrue(img_path.is_file(), f"Fixture missing: {img_path}")

        with open(img_path, "rb") as f:
            data = f.read()

        result = evaluate_module(data, filename="scan_module_6x12_fail.png")

        self.assertEqual(result.rows, 6)
        self.assertEqual(result.cols, 12)
        self.assertEqual(result.total_cells, 72)
        self.assertGreaterEqual(result.defective_cells_count, 4)
        self.assertGreaterEqual(result.module_damage_percentage, 5.0)
        self.assertEqual(result.status, "FAIL")

    def test_evaluate_12x10_half_cut_module(self):
        """Verify dynamic detection on half-cut 12x10 module."""
        img_path = self.fixtures_dir / "scan_module_12x10_fail.png"
        self.assertTrue(img_path.is_file(), f"Fixture missing: {img_path}")

        with open(img_path, "rb") as f:
            data = f.read()

        result = evaluate_module(data, filename="scan_module_12x10_fail.png")

        self.assertEqual(result.rows, 12)
        self.assertEqual(result.cols, 10)
        self.assertEqual(result.total_cells, 120)
        self.assertGreater(result.defective_cells_count, 0)
        self.assertEqual(result.status, "FAIL")

    def test_perspective_homography_tilted(self):
        """Verify perspective rectification on tilted scan."""
        img_path = self.fixtures_dir / "scan_module_tilted_6x10.png"
        self.assertTrue(img_path.is_file(), f"Fixture missing: {img_path}")

        with open(img_path, "rb") as f:
            data = f.read()

        result = evaluate_module(data, filename="scan_module_tilted_6x10.png", use_yolo=False)
        # Should rectify and resolve the grid
        self.assertEqual(result.rows, 6)
        self.assertEqual(result.cols, 10)
        self.assertEqual(result.total_cells, 60)

    def test_rotation_plus_and_minus_10_degrees(self):
        """Verify homography rectification and grid segmentation under +/- 10 degree rotations."""
        base_panel = generate_module_image(
            rows=6,
            cols=10,
            defect_map={(1, 3): "dead", (4, 7): "hot-spot"},
            apply_perspective_tilt=False,
        )

        for angle in (10.0, -10.0):
            with self.subTest(angle_deg=angle):
                rotated = _rotate_module_on_canvas(base_panel, angle_deg=angle, add_bench_noise=True)
                _, buf = cv2.imencode(".png", rotated)
                result = evaluate_module(
                    buf.tobytes(),
                    filename=f"scan_rot_{int(angle)}.png",
                    use_yolo=False,
                )
                self.assertEqual(result.rows, 6)
                self.assertEqual(result.cols, 10)
                self.assertEqual(result.total_cells, 60)
                self.assertEqual(len(result.cells), 60)

    def test_trapezoidal_perspective_distortion(self):
        """Verify rectification of severe trapezoidal perspective distortion (non-orthogonal camera)."""
        base_panel = generate_module_image(
            rows=6,
            cols=10,
            defect_map={(2, 2): "dead", (3, 8): "micro-crack"},
            apply_perspective_tilt=False,
        )
        ph, pw = base_panel.shape[:2]
        pad = 90
        canvas_w = pw + pad * 2
        canvas_h = ph + pad * 2
        canvas = np.full((canvas_h, canvas_w), 5, dtype=np.uint8)
        canvas[pad:pad + ph, pad:pad + pw] = base_panel

        src_pts = np.array([
            [pad, pad],
            [pad + pw, pad],
            [pad + pw, pad + ph],
            [pad, pad + ph],
        ], dtype=np.float32)

        # Strong trapezoidal foreshortening: top edge narrowed by 110px, bottom widened, vertical skew
        dst_pts = np.array([
            [pad + 60, pad + 45],
            [pad + pw - 55, pad + 30],
            [pad + pw + 15, pad + ph - 10],
            [pad - 15, pad + ph - 25],
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        trapezoid_img = cv2.warpPerspective(canvas, M, (canvas_w, canvas_h), borderValue=5)

        preproc = preprocess_el_module(trapezoid_img)
        self.assertEqual(preproc["rows"], 6)
        self.assertEqual(preproc["cols"], 10)
        self.assertEqual(preproc["total_cells"], 60)
        self.assertTrue(preproc["is_plausible_topology"])
        self.assertEqual(len(preproc["cells"]), 60)

    def test_panel_with_dark_border_and_corner_cells(self):
        """
        Verify that panels with dead/dark peripheral and corner cells (even when tilted
        or when an entire border row is dead) do not lose outer rows or columns.
        """
        # Case A: All 4 corners and multiple border cells are dead on a tilted module
        dark_border_defects = {
            (0, 0): "dead",
            (0, 1): "dead",
            (0, 9): "dead",
            (5, 0): "dead",
            (5, 9): "dead",
            (2, 0): "dead",
            (3, 9): "dead",
        }
        panel_a = generate_module_image(
            rows=6,
            cols=10,
            defect_map=dark_border_defects,
            apply_perspective_tilt=True,
        )
        # Even if corner (0,0) is completely blacked out to canvas level (0), 4-line hull
        # intersection reconstructs the missing corner
        preproc_a = preprocess_el_module(panel_a)
        self.assertEqual(preproc_a["rows"], 6)
        self.assertEqual(preproc_a["cols"], 10)
        self.assertEqual(preproc_a["total_cells"], 60)

        # Case B: Unframed tilted active grid where corner (0,0) and (5,9) are completely black (intensity 5)
        unframed = generate_module_image(
            rows=6,
            cols=10,
            outer_border_px=0,
            defect_map={(0, 0): "dead", (5, 9): "dead"},
            apply_perspective_tilt=False,
        )
        # Force dead corners to exact canvas background intensity (5)
        unframed[:70, :70] = 5
        unframed[-70:, -70:] = 5
        rotated_unframed = _rotate_module_on_canvas(unframed, angle_deg=7.5, add_bench_noise=True)
        quad = detect_panel_quadrilateral(rotated_unframed)
        self.assertIsNotNone(quad)
        self.assertEqual(quad.shape, (4, 2))
        rectified_unframed = normalize_perspective(rotated_unframed)
        r_u, c_u, _, _ = detect_grid_topology(rectified_unframed)
        self.assertEqual(r_u, 6)
        self.assertEqual(c_u, 10)

        # Case C: Entire top border row (all 10 cells in row 0) is 100% dead/dark
        all_top_row_dead = {(0, c): "dead" for c in range(10)}
        panel_c = generate_module_image(
            rows=6,
            cols=10,
            defect_map=all_top_row_dead,
            apply_perspective_tilt=False,
        )
        r_c, c_c, y_b, x_b = detect_grid_topology(panel_c)
        self.assertEqual(r_c, 6)
        self.assertEqual(c_c, 10)
        self.assertEqual(len(y_b), 7)
        self.assertEqual(len(x_b), 11)

    def test_standard_full_cell_vs_half_cut_with_central_gap(self):
        """
        Verify dynamic segmentation of Full-Cell (6x10) vs. Half-Cut (12x10 and 12x12)
        modules with a wide central horizontal junction-box gap, checking spatial metadata.
        """
        # 1. Standard Full-Cell 6x10
        full_cell_img = generate_module_image(rows=6, cols=10)
        res_full = preprocess_el_module(full_cell_img)
        self.assertEqual(res_full["rows"], 6)
        self.assertEqual(res_full["cols"], 10)
        self.assertEqual(res_full["total_cells"], 60)
        self.assertEqual(res_full["panel_type"], "full-cell")

        # 2. Half-Cut 12x10 (120 half-cells) with wide 22px central horizontal gap
        half_cut_120 = _generate_half_cut_with_central_gap(
            rows=12,
            cols=10,
            cell_w=60,
            cell_h=35,
            gap_px=4,
            central_gap_px=22,
            defect_map={(0, 2): "dead", (5, 4): "hot-spot", (6, 4): "micro-crack", (11, 9): "dead"},
        )
        res_hc120 = preprocess_el_module(half_cut_120)
        self.assertEqual(res_hc120["rows"], 12)
        self.assertEqual(res_hc120["cols"], 10)
        self.assertEqual(res_hc120["total_cells"], 120)
        self.assertEqual(res_hc120["panel_type"], "half-cut")
        self.assertTrue(res_hc120["is_plausible_topology"])
        self.assertEqual(len(res_hc120["cells"]), 120)

        # 3. Half-Cut 12x12 (144 half-cells) with wide 22px central horizontal gap
        half_cut_144 = _generate_half_cut_with_central_gap(
            rows=12,
            cols=12,
            cell_w=56,
            cell_h=34,
            gap_px=4,
            central_gap_px=22,
        )
        res_hc144 = preprocess_el_module(half_cut_144)
        self.assertEqual(res_hc144["rows"], 12)
        self.assertEqual(res_hc144["cols"], 12)
        self.assertEqual(res_hc144["total_cells"], 144)
        self.assertEqual(res_hc144["panel_type"], "half-cut")
        self.assertTrue(res_hc144["is_plausible_topology"])

        # Validate spatial traceability metadata dictionary structure on every cell
        for expected_id, cell_meta in enumerate(res_hc120["cells"]):
            self.assertEqual(set(cell_meta.keys()), {"cell_id", "row", "col", "bbox", "image"})
            self.assertEqual(cell_meta["cell_id"], expected_id)
            self.assertEqual(cell_meta["row"], expected_id // 10)
            self.assertEqual(cell_meta["col"], expected_id % 10)
            bbox = cell_meta["bbox"]
            self.assertEqual(len(bbox), 4)
            self.assertGreater(bbox[2], 0)
            self.assertGreater(bbox[3], 0)
            crop = cell_meta["image"]
            self.assertIsInstance(crop, np.ndarray)
            self.assertEqual(crop.dtype, np.uint8)
            self.assertEqual(crop.shape, (bbox[3], bbox[2]))

    def test_geometric_fallback_on_severe_noise(self):
        """
        Verify that when an image contains severe noise without discernible cell gaps,
        detect_grid_topology falls back cleanly to standard aspect-ratio geometry.
        """
        rng = np.random.default_rng(42)
        # Landscape ~1.67 aspect ratio noisy image -> should fallback to 6x10 (60 cells)
        noisy_6x10 = rng.integers(90, 140, size=(480, 800), dtype=np.uint8)
        rows, cols, y_b, x_b = detect_grid_topology(noisy_6x10)
        self.assertEqual(rows, 6)
        self.assertEqual(cols, 10)
        self.assertEqual(len(y_b), 7)
        self.assertEqual(len(x_b), 11)

        # Landscape ~2.0 aspect ratio noisy image -> should fallback to 6x12 (72 cells)
        noisy_6x12 = rng.integers(90, 140, size=(450, 900), dtype=np.uint8)
        rows_72, cols_72, _, _ = detect_grid_topology(noisy_6x12)
        self.assertEqual(rows_72, 6)
        self.assertEqual(cols_72, 12)

    def test_yolo_classifier_loading_and_inference(self):
        """Verify YOLOv8-cls model loads and classifies single & batch cell crops correctly."""
        model = get_yolo_classifier()
        self.assertIsNotNone(model)
        self.assertEqual(model.task, "classify")
        self.assertIn("boa", model.names.values())
        self.assertIn("defeituosa", model.names.values())

        model_hash = get_model_sha256()
        self.assertEqual(len(model_hash), 64)

        # Verify preprocessed cell crops feed directly into YOLO without contract changes
        sample_panel = generate_module_image(rows=6, cols=10, apply_perspective_tilt=True)
        preproc = preprocess_el_module(sample_panel)
        first_crop = preproc["cells"][0]["image"]
        self.assertEqual(first_crop.dtype, np.uint8)

        is_defective, defect_type, confidence = classify_cell_yolo(first_crop, model=model)
        self.assertIsInstance(is_defective, bool)
        self.assertIsInstance(defect_type, DefectType)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

        # Test vectorized batch inference on multiple cell crops
        batch_crops = [c["image"] for c in preproc["cells"][:4]]
        batch_preds = classify_cells_batch_yolo(batch_crops, model=model)
        self.assertEqual(len(batch_preds), 4)
        for pred_def, pred_type, pred_conf in batch_preds:
            self.assertIsInstance(pred_def, bool)
            self.assertIsInstance(pred_type, DefectType)
            self.assertGreaterEqual(pred_conf, 0.0)
            self.assertLessEqual(pred_conf, 1.0)

    def test_draw_defect_annotations(self):
        """Verify drawing bounding boxes on defective cells."""
        img = np.full((200, 200, 3), 128, dtype=np.uint8)
        from cv_engine import CellInspection, CellBoundingBox
        cells = [
            CellInspection(
                index=0, row=0, col=0, is_defective=True,
                defect_type=DefectType.DEFEITUOSA, confidence=0.92,
                bbox=CellBoundingBox(x=10, y=10, width=80, height=80, x_pct=5.0, y_pct=5.0, width_pct=40.0, height_pct=40.0)
            ),
            CellInspection(
                index=1, row=0, col=1, is_defective=False,
                defect_type=DefectType.NONE, confidence=0.98,
                bbox=CellBoundingBox(x=100, y=10, width=80, height=80, x_pct=50.0, y_pct=5.0, width_pct=40.0, height_pct=40.0)
            )
        ]
        annotated = draw_defect_annotations(img, cells, draw_banner=True)
        self.assertIsNotNone(annotated)
        self.assertEqual(len(annotated.shape), 3)
        # Should have banner height added
        self.assertGreater(annotated.shape[0], 200)

    def test_trim_cell_gap_margins_preserves_central_wafer(self):
        """
        Verify that _trim_cell_gap_margins strips dark inter-cell gap borders
        while preserving the central active silicon wafer.
        """
        # Synthesize a 64x64 cell with a 3px dark inter-cell gap border (intensity 18)
        # surrounding a bright uniform silicon wafer (intensity 175)
        cell_with_gap = np.full((64, 64), 175, dtype=np.uint8)
        cell_with_gap[:3, :] = 18
        cell_with_gap[-3:, :] = 18
        cell_with_gap[:, :3] = 18
        cell_with_gap[:, -3:] = 18

        trimmed = _trim_cell_gap_margins(cell_with_gap, trim_ratio=0.05)
        self.assertLess(trimmed.shape[0], 64)
        self.assertLess(trimmed.shape[1], 64)
        self.assertGreaterEqual(trimmed.shape[0], 50)
        self.assertGreaterEqual(trimmed.shape[1], 50)
        # All dark border pixels (18) must be removed, leaving only the 175 wafer
        self.assertEqual(int(trimmed.min()), 175)
        self.assertEqual(int(trimmed.max()), 175)

    def test_grid_topology_ignores_internal_cell_busbars(self):
        """
        Verify that internal horizontal busbars inside EL cells do not cause
        _detect_1d_dividers to double the row count (e.g., 6x10 mistaken for 12x10).
        """
        rows, cols = 6, 10
        cell_w, cell_h = 80, 80
        gap_px, border_px = 6, 20
        total_w = cols * cell_w + (cols - 1) * gap_px + 2 * border_px
        total_h = rows * cell_h + (rows - 1) * gap_px + 2 * border_px
        panel = np.full((total_h, total_w), 14, dtype=np.uint8)

        rng = np.random.default_rng(123)
        for r in range(rows):
            for c in range(cols):
                y0 = border_px + r * (cell_h + gap_px)
                x0 = border_px + c * (cell_w + gap_px)
                wafer = rng.normal(172, 9, size=(cell_h, cell_w)).clip(120, 220).astype(np.uint8)
                # Add 3 horizontal internal busbars inside each cell (shallower than deep inter-cell gaps)
                for frac in (0.25, 0.50, 0.75):
                    by = int(cell_h * frac)
                    wafer[max(0, by - 1):min(cell_h, by + 2), :] = 95
                panel[y0:y0 + cell_h, x0:x0 + cell_w] = wafer

        det_rows, det_cols, y_bounds, x_bounds = detect_grid_topology(panel)
        self.assertEqual(det_rows, 6)
        self.assertEqual(det_cols, 10)
        self.assertEqual(len(y_bounds), 7)
        self.assertEqual(len(x_bounds), 11)

    def test_evaluate_module_with_yolo_false_alarm_resilience(self):
        """
        Verify that evaluate_module(..., use_yolo=True) suppresses false positives on
        clean/tilted synthetic scans (PASS) while accurately flagging defective modules (FAIL).
        """
        pass_fixtures = [
            ("scan_module_6x10_pass.png", 6, 10),
            ("scan_module_tilted_6x10.png", 6, 10),
        ]
        for fname, exp_r, exp_c in pass_fixtures:
            with self.subTest(fixture=fname):
                fpath = self.fixtures_dir / fname
                with open(fpath, "rb") as f:
                    data = f.read()
                res = evaluate_module(data, filename=fname, use_yolo=True)
                self.assertEqual(res.rows, exp_r)
                self.assertEqual(res.cols, exp_c)
                self.assertLess(res.module_damage_percentage, 5.0)
                self.assertEqual(res.status, "PASS")

        fail_fixtures = [
            ("scan_module_6x12_fail.png", 6, 12),
            ("scan_module_12x10_fail.png", 12, 10),
        ]
        for fname, exp_r, exp_c in fail_fixtures:
            with self.subTest(fixture=fname):
                fpath = self.fixtures_dir / fname
                with open(fpath, "rb") as f:
                    data = f.read()
                res = evaluate_module(data, filename=fname, use_yolo=True)
                self.assertEqual(res.rows, exp_r)
                self.assertEqual(res.cols, exp_c)
                self.assertGreaterEqual(res.module_damage_percentage, 5.0)
                self.assertEqual(res.status, "FAIL")

    def test_single_cell_1x1_topology_and_tta_evaluation(self):
        """
        Verify that an isolated square/near-square EL cell image (e.g., 300x300 single cell)
        is detected as rows=1, cols=1 (total_cells=1) without triggering spurious 6x10 slicing,
        and is evaluated via TTA with 0.0% (PASS) or 100.0% (FAIL) damage percentage.
        """
        single_good_cell = create_synthetic_el_cell(300, 300, defect_type="none")
        # Add 3 horizontal busbars typical of a 300x300 single EL cell
        for frac in (0.25, 0.50, 0.75):
            by = int(300 * frac)
            single_good_cell[by - 1:by + 2, :] = 85

        rows, cols, y_b, x_b = detect_grid_topology(single_good_cell)
        self.assertEqual(rows, 1)
        self.assertEqual(cols, 1)
        self.assertEqual(y_b, [0, 300])
        self.assertEqual(x_b, [0, 300])

        _, buf_good = cv2.imencode(".png", single_good_cell)
        res_good = evaluate_module(buf_good.tobytes(), filename="cell_good_300x300.png", use_yolo=True)
        self.assertEqual(res_good.rows, 1)
        self.assertEqual(res_good.cols, 1)
        self.assertEqual(res_good.total_cells, 1)
        self.assertIn(res_good.module_damage_percentage, (0.0, 100.0))
        self.assertEqual(res_good.status, "FAIL" if res_good.defective_cells_count == 1 else "PASS")

    def test_half_cut_6x24_landscape_module_with_central_busbar(self):
        """
        Verify that a landscape 6x24 Half-Cut EL module (144 half-cells divided by a thick
        central vertical busbar) preserves its native landscape orientation, detects the exact
        6x24 grid, and marks only genuinely cracked cells instead of the entire panel.
        """
        webp_path = self.fixtures_dir / "Electroluminescence+(EL)+Testing+for+PV+Modules.webp"
        if not webp_path.is_file():
            self.skipTest("6x24 WebP fixture not present.")
        with open(webp_path, "rb") as f:
            raw_bytes = f.read()

        res = evaluate_module(raw_bytes, filename=webp_path.name, use_yolo=True)
        self.assertEqual(res.rows, 6)
        self.assertEqual(res.cols, 24)
        self.assertEqual(res.total_cells, 144)
        self.assertGreater(res.image_width, res.image_height)
        self.assertEqual(res.status, "FAIL")
        self.assertGreaterEqual(res.defective_cells_count, 15)
        self.assertLessEqual(res.defective_cells_count, 45)


if __name__ == "__main__":
    unittest.main()
