"""
Integration Tests for Node 1 (Gateway) and Node 2 (Worker)
==========================================================
Verifies:
- Node 2 standalone endpoints (/health, /api/v1/process-module, error responses)
- Node 1 static and console serving (/ and /fixtures/...)
- Node 1 bulk orchestrator (/api/v1/analyze/bulk) with concurrent dispatch
- Telemetry aggregation calculations and JSON contract conformity
"""

import asyncio
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
import httpx

# Add paths to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "node1_gateway"))
sys.path.insert(0, str(ROOT_DIR / "node2_worker"))

import node1_gateway.main as gateway_module
from node1_gateway.main import app as gateway_app
from node2_worker.main import app as worker_app


class TestDistributedAPIIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures_dir = ROOT_DIR / "tests" / "fixtures"
        cls.worker_client = TestClient(worker_app)
        cls.gateway_client = TestClient(gateway_app)

    # -------------------------------------------------------------------------
    # Node 2 (Worker) Tests
    # -------------------------------------------------------------------------
    def test_worker_health(self):
        """Worker /health probe returns 200 and healthy status."""
        response = self.worker_client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["service"], "node2_worker")

    def test_worker_process_module_valid(self):
        """Worker processes valid module scan and returns contract conforming JSON."""
        img_path = self.fixtures_dir / "scan_module_6x10_pass.png"
        with open(img_path, "rb") as f:
            files = {"file": ("scan_module_6x10_pass.png", f.read(), "image/png")}

        response = self.worker_client.post("/api/v1/process-module", files=files)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn("module_id", data)
        self.assertEqual(data["filename"], "scan_module_6x10_pass.png")
        self.assertEqual(data["rows"], 6)
        self.assertEqual(data["cols"], 10)
        self.assertEqual(data["total_cells"], 60)
        self.assertEqual(data["status"], "PASS")
        self.assertEqual(len(data["cells"]), 60)

    def test_worker_corrupted_image_returns_422(self):
        """Worker returns HTTP 422 with structured detail when image is corrupt."""
        files = {"file": ("corrupt.png", b"this is completely corrupted bytes", "image/png")}
        response = self.worker_client.post("/api/v1/process-module", files=files)
        self.assertEqual(response.status_code, 422)
        data = response.json()
        self.assertIn("detail", data)

    # -------------------------------------------------------------------------
    # Node 1 (Gateway) Tests
    # -------------------------------------------------------------------------
    def test_gateway_serves_console_html(self):
        """Gateway GET / serves frontend/index.html."""
        response = self.gateway_client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers.get("content-type", ""))
        self.assertIn("LuminaPV", response.text)
        self.assertIn("POST /api/v1/analyze/bulk", response.text)
        self.assertIn("css/variables.css", response.text)
        self.assertIn("js/app.js", response.text)

    def test_gateway_serves_modular_frontend_assets(self):
        """Gateway serves mounted static CSS and JS modules."""
        css_resp = self.gateway_client.get("/css/variables.css")
        self.assertEqual(css_resp.status_code, 200)
        self.assertIn("--defect", css_resp.text)

        layout_resp = self.gateway_client.get("/css/layout.css")
        self.assertEqual(layout_resp.status_code, 200)

        comp_resp = self.gateway_client.get("/css/components.css")
        self.assertEqual(comp_resp.status_code, 200)

        js_resp = self.gateway_client.get("/js/app.js")
        self.assertEqual(js_resp.status_code, 200)
        self.assertIn("LuminaPV", js_resp.text)

    def test_gateway_serves_legacy_console(self):
        """Gateway GET /legacy serves backwards-compatible monolithic console."""
        response = self.gateway_client.get("/legacy")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers.get("content-type", ""))
        self.assertIn("LuminaPV", response.text)

    def test_gateway_serves_fixtures(self):
        """Gateway GET /fixtures/luminapv-batch-sample.json serves sample JSON."""
        response = self.gateway_client.get("/fixtures/luminapv-batch-sample.json")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("batch_summary", data)
        self.assertIn("modules", data)

    def test_gateway_bulk_analysis_end_to_end_mocked(self):
        """
        Tests Gateway POST /api/v1/analyze/bulk orchestrating parallel requests,
        verifying aggregated batch metrics according to specifications:
        total_modules = len(modules)
        defective_modules_count = count(status == 'FAIL')
        batch_failure_percentage = (defective_modules_count / total_modules) * 100
        average_cell_degradation_percentage = mean(module_damage_percentage)
        """
        # Mock responses from worker for 2 modules (1 pass, 1 fail)
        mock_module_pass = {
            "module_id": "MOD-2026-001",
            "filename": "scan_01.png",
            "status": "PASS",
            "rows": 6,
            "cols": 10,
            "total_cells": 60,
            "defective_cells_count": 1,
            "module_damage_percentage": 1.7,
            "cells": []
        }
        mock_module_fail = {
            "module_id": "MOD-2026-002",
            "filename": "scan_02.png",
            "status": "FAIL",
            "rows": 6,
            "cols": 12,
            "total_cells": 72,
            "defective_cells_count": 12,
            "module_damage_percentage": 16.7,
            "cells": []
        }

        async def mock_dispatch(client, file_bytes, filename, content_type):
            if "01" in filename:
                return mock_module_pass
            return mock_module_fail

        with patch("node1_gateway.main.dispatch_module_to_worker", side_effect=mock_dispatch):
            files = [
                ("files", ("scan_01.png", b"fake_bytes_1", "image/png")),
                ("files", ("scan_02.png", b"fake_bytes_2", "image/png"))
            ]
            response = self.gateway_client.post("/api/v1/analyze/bulk", files=files)

            self.assertEqual(response.status_code, 200)
            data = response.json()

            # Verify Batch Summary contract
            self.assertIn("batch_summary", data)
            summary = data["batch_summary"]
            self.assertEqual(summary["total_modules"], 2)
            self.assertEqual(summary["defective_modules_count"], 1)
            self.assertEqual(summary["batch_failure_percentage"], 50.0)
            # Average degradation: (1.7 + 16.7) / 2 = 9.2%
            self.assertEqual(summary["average_cell_degradation_percentage"], 9.2)

            # Verify modules list
            self.assertEqual(len(data["modules"]), 2)
            self.assertEqual(data["modules"][0]["module_id"], "MOD-2026-001")
            self.assertEqual(data["modules"][1]["module_id"], "MOD-2026-002")

    def test_gateway_bulk_analysis_with_real_worker_direct(self):
        """
        Tests Gateway calling the real Worker pipeline by routing httpx calls
        directly through the worker test client.
        """
        img1_path = self.fixtures_dir / "scan_module_6x10_pass.png"
        img2_path = self.fixtures_dir / "scan_module_6x12_fail.png"

        with open(img1_path, "rb") as f1, open(img2_path, "rb") as f2:
            f1_bytes = f1.read()
            f2_bytes = f2.read()

        async def direct_worker_dispatch(client, file_bytes, filename, content_type):
            # Call worker app directly in-memory
            from cv_engine import evaluate_module
            res = evaluate_module(file_bytes, filename=filename)
            return json.loads(res.model_dump_json())

        with patch("node1_gateway.main.dispatch_module_to_worker", side_effect=direct_worker_dispatch):
            files = [
                ("files", ("scan_module_6x10_pass.png", f1_bytes, "image/png")),
                ("files", ("scan_module_6x12_fail.png", f2_bytes, "image/png"))
            ]
            response = self.gateway_client.post("/api/v1/analyze/bulk", files=files)
            self.assertEqual(response.status_code, 200)
            data = response.json()

            summary = data["batch_summary"]
            self.assertEqual(summary["total_modules"], 2)
            self.assertEqual(summary["defective_modules_count"], 1)
            self.assertEqual(summary["batch_failure_percentage"], 50.0)
            self.assertGreater(summary["average_cell_degradation_percentage"], 0.0)

            modules = data["modules"]
            self.assertEqual(len(modules), 2)
            self.assertEqual(modules[0]["status"], "PASS")
            self.assertEqual(modules[1]["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
