"""
Unit Tests for Computer Vision & Signal Processing Engine (Node 2)
==================================================================
"""

import os
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

# Add node2_worker to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "node2_worker"))

from cv_engine import (
    DefectType,
    evaluate_module,
    normalize_perspective,
    detect_grid_topology,
    inspect_cell_region,
    classify_cell_yolo,
    draw_defect_annotations,
    get_yolo_classifier,
    get_model_sha256
)


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

    def test_yolo_classifier_loading_and_inference(self):
        """Verify YOLOv8-cls model loads and classifies cell crops correctly."""
        model = get_yolo_classifier()
        self.assertIsNotNone(model)
        self.assertEqual(model.task, "classify")
        self.assertIn("boa", model.names.values())
        self.assertIn("defeituosa", model.names.values())

        model_hash = get_model_sha256()
        self.assertEqual(len(model_hash), 64)

        # Test inference on a 64x64 cell crop
        dummy_cell = np.full((64, 64), 180, dtype=np.uint8)
        is_defective, defect_type, confidence = classify_cell_yolo(dummy_cell, model=model)
        self.assertIsInstance(is_defective, bool)
        self.assertIsInstance(defect_type, DefectType)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

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


if __name__ == "__main__":
    unittest.main()
