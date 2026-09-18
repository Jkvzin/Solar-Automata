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
    inspect_cell_region
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

        result = evaluate_module(data, filename="scan_module_6x10_pass.png")

        self.assertEqual(result.rows, 6)
        self.assertEqual(result.cols, 10)
        self.assertEqual(result.total_cells, 60)
        self.assertEqual(len(result.cells), 60)
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

        result = evaluate_module(data, filename="scan_module_tilted_6x10.png")
        # Should rectify and resolve the grid
        self.assertEqual(result.rows, 6)
        self.assertEqual(result.cols, 10)
        self.assertEqual(result.total_cells, 60)


if __name__ == "__main__":
    unittest.main()
