"""
Integration Tests for Node 1 (Gateway) and Node 2 (Worker)
==========================================================
Verifies:
- Node 2 standalone endpoints (/health, /api/v1/registry/node, /api/v1/replicas/stats, /api/v1/process-module)
- Node 2 service discovery heartbeat registration (Redis + memory fallback)
- Node 2 Primary-Backup result replication, Read-Your-Writes distributed cache hit, and model consistency
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
import node2_worker.main as worker_module
from node2_worker.main import app as worker_app


class TestDistributedAPIIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures_dir = ROOT_DIR / "tests" / "fixtures"
        # Mock AWS S3 & SQS clients to prevent network calls/timeouts in test environments
        patcher_s3 = patch.object(gateway_module, "s3_client")
        patcher_sqs = patch.object(gateway_module, "sqs_client")
        cls.mock_s3 = patcher_s3.start()
        cls.mock_sqs = patcher_sqs.start()
        cls.addClassCleanup(patcher_s3.stop)
        cls.addClassCleanup(patcher_sqs.stop)

        cls.worker_client = TestClient(worker_app)
        cls.gateway_client = TestClient(gateway_app)

    # -------------------------------------------------------------------------
    # Node 2 (Worker), Service Discovery & Replication Tests
    # -------------------------------------------------------------------------
    def test_worker_health(self):
        """Worker /health probe returns 200, healthy status, naming, and consistency metadata."""
        response = self.worker_client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["service"], "node2_worker")
        self.assertIn("node_id", data)
        self.assertIn("service_name", data)
        self.assertIn("dns_name", data)
        self.assertIn("model_sha256", data)
        self.assertEqual(len(data["model_sha256"]), 64)
        self.assertEqual(data["consistency_model"], "read-your-writes")
        self.assertTrue(data["model_consistent"])

    def test_worker_registry_node_endpoint_and_redis_heartbeat(self):
        """Worker /api/v1/registry/node exposes naming metadata and supports Redis heartbeat."""
        response = self.worker_client.get("/api/v1/registry/node")
        self.assertEqual(response.status_code, 200)
        meta = response.json()
        self.assertEqual(meta["service_name"], "service://luminapv-vision-worker")
        self.assertTrue(meta["dns_name"].endswith(".luminapv.local"))
        self.assertEqual(meta["status"], "ONLINE")

        # Verify Redis heartbeat registration and clean shutdown (srem) with mocked Redis client
        mock_redis = AsyncMock()
        worker_module.registry.redis_client = mock_redis
        try:
            hb_meta = asyncio.run(worker_module.registry.register_heartbeat())
            self.assertEqual(hb_meta["registry_backend"], "redis")
            mock_redis.set.assert_awaited_once()
            mock_redis.sadd.assert_awaited_once()

            # Verify stop() removes node key and cleans node_id from Redis Set
            asyncio.run(worker_module.registry.stop())
            mock_redis.delete.assert_awaited_once_with(
                f"{worker_module.REGISTRY_KEY_PREFIX}:{worker_module.registry.node_id}"
            )
            mock_redis.srem.assert_awaited_once_with(
                f"{worker_module.REGISTRY_KEY_PREFIX}:nodes",
                worker_module.registry.node_id,
            )
            mock_redis.close.assert_awaited_once()
        finally:
            worker_module.registry.redis_client = None

    def test_worker_process_module_valid(self):
        """Worker processes valid module scan and returns contract conforming JSON with replication metadata."""
        worker_module._LOCAL_REPLICA_STORE.clear()
        img_path = self.fixtures_dir / "scan_module_6x10_pass.png"
        with open(img_path, "rb") as f:
            raw_bytes = f.read()

        files = {"file": ("scan_module_6x10_pass.png", raw_bytes, "image/png")}
        response = self.worker_client.post("/api/v1/process-module?use_yolo=false&use_cache=false", files=files)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn("module_id", data)
        self.assertEqual(data["filename"], "scan_module_6x10_pass.png")
        self.assertEqual(data["rows"], 6)
        self.assertEqual(data["cols"], 10)
        self.assertEqual(data["total_cells"], 60)
        self.assertEqual(data["status"], "PASS")
        self.assertIn("processed_by_node", data)
        self.assertIn("model_sha256", data)
        self.assertIn("content_sha256", data)
        self.assertTrue(data["replicated"])
        self.assertFalse(data["cache_hit"])
        self.assertEqual(data["consistency_model"], "read-your-writes")
        self.assertEqual(len(data["cells"]), 60)

    def test_worker_distributed_replication_and_read_your_writes_cache(self):
        """Worker replicates ModuleResult and serves Read-Your-Writes cache hit on subsequent requests."""
        worker_module._LOCAL_REPLICA_STORE.clear()
        img_path = self.fixtures_dir / "scan_module_6x10_pass.png"
        with open(img_path, "rb") as f:
            raw_bytes = f.read()

        # 1st request: cache miss -> evaluates and replicates
        res1 = self.worker_client.post(
            "/api/v1/process-module?use_yolo=false&use_cache=true",
            files={"file": ("scan_01.png", raw_bytes, "image/png")}
        )
        self.assertEqual(res1.status_code, 200)
        d1 = res1.json()
        self.assertTrue(d1["replicated"])
        self.assertFalse(d1["cache_hit"])
        content_hash = d1["content_sha256"]

        # 2nd request with same image content: Read-Your-Writes cache HIT
        res2 = self.worker_client.post(
            "/api/v1/process-module?use_yolo=false&use_cache=true",
            files={"file": ("scan_02.png", raw_bytes, "image/png")}
        )
        self.assertEqual(res2.status_code, 200)
        d2 = res2.json()
        self.assertTrue(d2["replicated"])
        self.assertTrue(d2["cache_hit"])
        self.assertEqual(d2["content_sha256"], content_hash)
        self.assertEqual(d2["filename"], "scan_02.png")

        # Verify GET /api/v1/replicas/{content_sha256}
        replica_resp = self.worker_client.get(f"/api/v1/replicas/{content_hash}")
        self.assertEqual(replica_resp.status_code, 200)
        self.assertEqual(replica_resp.json()["content_sha256"], content_hash)

        # Verify GET /api/v1/replicas/stats
        stats_resp = self.worker_client.get("/api/v1/replicas/stats")
        self.assertEqual(stats_resp.status_code, 200)
        stats = stats_resp.json()
        self.assertEqual(stats["consistency_model"], "read-your-writes")
        self.assertEqual(stats["replication_strategy"], "primary-backup")
        self.assertTrue(stats["model_consistent"])
        self.assertGreaterEqual(stats["total_replicated_modules"], 1)
        self.assertGreaterEqual(stats["cache_hits"], 1)

    def test_worker_local_lru_replica_store_capacity_eviction(self):
        """Local LRUReplicaStore evicts least recently used items when capacity is exceeded."""
        lru = worker_module.LRUReplicaStore(max_size=3)
        lru["hash_1"] = {"id": 1}
        lru["hash_2"] = {"id": 2}
        lru["hash_3"] = {"id": 3}
        self.assertEqual(len(lru), 3)

        # Access hash_1 so it becomes most recently used (hash_2 becomes oldest)
        _ = lru.get("hash_1")

        # Insert hash_4 -> should evict hash_2
        lru["hash_4"] = {"id": 4}
        self.assertEqual(len(lru), 3)
        self.assertNotIn("hash_2", lru)
        self.assertIn("hash_1", lru)
        self.assertIn("hash_3", lru)
        self.assertIn("hash_4", lru)

    def test_worker_cache_invalid_schema_degrades_to_cache_miss(self):
        """get_cached_result safely degrades to cache miss when cached payload has incompatible schema."""
        worker_module._LOCAL_REPLICA_STORE.clear()
        img_path = self.fixtures_dir / "scan_module_6x10_pass.png"
        with open(img_path, "rb") as f:
            raw_bytes = f.read()

        import hashlib
        content_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        content_key = f"{content_sha256}:yolo=0"

        # Inject corrupted/incompatible schema payload into local replica store
        worker_module._LOCAL_REPLICA_STORE[content_key] = {
            "filename": "corrupted_schema.png",
            "model_sha256": worker_module.get_model_sha256(),
            "rows": "invalid_type_not_an_int",
            # missing required fields like cols, total_cells, status, etc.
        }

        misses_before = worker_module.replicator.cache_misses
        cached = asyncio.run(
            worker_module.replicator.get_cached_result(
                raw_bytes,
                filename="scan_module_6x10_pass.png",
                use_yolo=False,
            )
        )
        self.assertIsNone(cached)
        self.assertEqual(worker_module.replicator.cache_misses, misses_before + 1)
        self.assertNotIn(content_key, worker_module._LOCAL_REPLICA_STORE)

        # Full endpoint request should recompute cleanly (HTTP 200) after schema cache miss
        worker_module._LOCAL_REPLICA_STORE[content_key] = {
            "filename": "corrupted_schema.png",
            "model_sha256": worker_module.get_model_sha256(),
        }
        res = self.worker_client.post(
            "/api/v1/process-module?use_yolo=false&use_cache=true",
            files={"file": ("scan_module_6x10_pass.png", raw_bytes, "image/png")},
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertFalse(data["cache_hit"])
        self.assertTrue(data["replicated"])

    def test_worker_replication_stats_purges_expired_redis_hashes(self):
        """get_replication_stats purges expired hashes from REPLICA_INDEX_KEY in Redis."""
        worker_module._LOCAL_REPLICA_STORE.clear()
        mock_redis = AsyncMock()
        mock_redis.get.return_value = worker_module.get_model_sha256()
        mock_redis.smembers.return_value = {"valid_hash_aaa", "expired_hash_bbb"}
        mock_redis.exists.side_effect = lambda key: 1 if "valid_hash_aaa" in key else 0

        worker_module.registry.redis_client = mock_redis
        try:
            stats = asyncio.run(worker_module.replicator.get_replication_stats())
            self.assertEqual(stats["total_replicated_modules"], 1)
            self.assertEqual(stats["replicated_keys"], ["valid_hash_aaa"])
            mock_redis.srem.assert_awaited_once_with(
                worker_module.REPLICA_INDEX_KEY, "expired_hash_bbb"
            )
        finally:
            worker_module.registry.redis_client = None

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
            "processed_by_node": "worker-1",
            "replicated": True,
            "consistency_model": "read-your-writes",
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
            "processed_by_node": "worker-2",
            "replicated": True,
            "consistency_model": "read-your-writes",
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
            # Call worker app directly in-memory with deterministic heuristic for synthetic fixtures
            from cv_engine import evaluate_module
            res = evaluate_module(file_bytes, filename=filename, use_yolo=False)
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

    # -------------------------------------------------------------------------
    # Issue #3: Leader Election & Coordination Tests
    # -------------------------------------------------------------------------
    def test_worker_health_leader_election_telemetry(self):
        """Worker /health probe exposes leader election metadata: role, term, and leader_id."""
        response = self.worker_client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("role", data)
        self.assertIn("term", data)
        self.assertIn("leader_id", data)
        self.assertIn(data["role"], ["LEADER", "FOLLOWER"])
        self.assertIsInstance(data["term"], int)
        self.assertGreaterEqual(data["term"], 1)

    def test_worker_cluster_election_step_down_and_reelect(self):
        """Worker POST /api/v1/cluster/election supports step_down simulation and re-election."""
        # 1. Step down
        resp_step = self.worker_client.post("/api/v1/cluster/election?action=step_down")
        self.assertEqual(resp_step.status_code, 200)
        step_data = resp_step.json()
        self.assertEqual(step_data["status"], "election_triggered")
        self.assertEqual(step_data["role"], "FOLLOWER")

        # 2. Re-elect
        resp_elect = self.worker_client.post("/api/v1/cluster/election?action=reelect")
        self.assertEqual(resp_elect.status_code, 200)
        elect_data = resp_elect.json()
        self.assertEqual(elect_data["status"], "election_evaluated")
        self.assertIn(elect_data["role"], ["LEADER", "FOLLOWER"])

    def test_worker_leader_election_with_mocked_redis(self):
        """Worker LeaderElectionManager acquires atomic lease (SET ... NX EX 8) and updates term in Redis."""
        mock_redis = AsyncMock()
        mock_redis.get.return_value = None  # No current leader
        mock_redis.incr.return_value = 2    # Term incremented
        mock_redis.set.return_value = True  # Lease acquired

        worker_module.registry.redis_client = mock_redis
        try:
            res = asyncio.run(worker_module.election_manager.elect_or_renew())
            self.assertEqual(res["role"], "LEADER")
            self.assertEqual(res["term"], 2)
            self.assertEqual(res["leader_id"], worker_module.registry.node_id)
            mock_redis.set.assert_awaited()
        finally:
            worker_module.registry.redis_client = None

    # -------------------------------------------------------------------------
    # Issue #3: Distributed Mutual Exclusion (DistributedLock) Tests
    # -------------------------------------------------------------------------
    def test_distributed_lock_acquisition_release_and_collision(self):
        """DistributedLock acquires lock, tracks active locks, and blocks conflicting concurrent acquires."""
        lock1 = gateway_module.lock_manager.create_lock("lock:module:test_hash_abc", ttl_seconds=10)
        lock2 = gateway_module.lock_manager.create_lock("lock:module:test_hash_abc", ttl_seconds=10)

        # 1. First acquire succeeds
        acquired1 = asyncio.run(lock1.acquire())
        self.assertTrue(acquired1)

        # 2. Verify active locks tracking
        active = asyncio.run(gateway_module.lock_manager.get_active_locks())
        self.assertTrue(any(item["resource"] == "lock:module:test_hash_abc" for item in active))

        # 3. Second acquire for the same resource fails while held
        acquired2 = asyncio.run(lock2.acquire(timeout=0.1))
        self.assertFalse(acquired2)

        # 4. Release lock 1
        released1 = asyncio.run(lock1.release())
        self.assertTrue(released1)

        # 5. Lock 2 can now acquire
        acquired2_after = asyncio.run(lock2.acquire(timeout=0.5))
        self.assertTrue(acquired2_after)
        asyncio.run(lock2.release())

    def test_distributed_lock_redis_lua_integration(self):
        """DistributedLock interacts with Redis SET NX and Lua atomic script correctly."""
        mock_redis = AsyncMock()
        mock_redis.set.return_value = True
        mock_redis.eval.return_value = 1

        lock = gateway_module.DistributedLock(
            resource="lock:module:redis_test_123",
            ttl_seconds=30,
            redis_client=mock_redis
        )

        acq = asyncio.run(lock.acquire())
        self.assertTrue(acq)
        mock_redis.set.assert_awaited_once()

        rel = asyncio.run(lock.release())
        self.assertTrue(rel)
        mock_redis.eval.assert_awaited_once()

    # -------------------------------------------------------------------------
    # Issue #3: AWS S3/SQS Ingestion Resilience Tests
    # -------------------------------------------------------------------------
    def test_gateway_bulk_analysis_aws_failure_resilience(self):
        """Gateway analyze_bulk does not fail with HTTP 500 if AWS S3/SQS encounters an error."""
        # Force mock S3 and SQS to raise exceptions to simulate missing AWS credentials / network outage
        self.mock_s3.upload_fileobj.side_effect = Exception("AWS Access Denied / Invalid Credentials")
        self.mock_sqs.send_message.side_effect = Exception("AWS SQS Unreachable")

        img1_path = self.fixtures_dir / "scan_module_6x10_pass.png"
        with open(img1_path, "rb") as f1:
            f1_bytes = f1.read()

        async def dummy_worker_dispatch(*args, **kwargs):
            return {
                "module_id": "MOD-2026-999",
                "filename": "scan_module_6x10_pass.png",
                "status": "PASS",
                "rows": 6,
                "cols": 10,
                "total_cells": 60,
                "defective_cells_count": 0,
                "module_damage_percentage": 0.0,
                "processed_by_node": "worker-1",
                "cells": []
            }

        with patch("node1_gateway.main.dispatch_module_to_worker", side_effect=dummy_worker_dispatch):
            files = [("files", ("scan_module_6x10_pass.png", f1_bytes, "image/png"))]
            # Should succeed with HTTP 200, safely catching the AWS error
            response = self.gateway_client.post("/api/v1/analyze/bulk", files=files)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["batch_summary"]["total_modules"], 1)

        # Reset mocks side effect for subsequent tests
        self.mock_s3.upload_fileobj.side_effect = None
        self.mock_sqs.send_message.side_effect = None

    # -------------------------------------------------------------------------
    # Issue #4: Dynamic Name Resolution, Load Balancing & Failover Tests
    # -------------------------------------------------------------------------
    def test_service_catalog_resolver_dynamic_names(self):
        """ServiceCatalogResolver resolves logical service name to active healthy endpoints from Redis."""
        mock_redis = AsyncMock()
        mock_redis.smembers.return_value = {"worker-1", "worker-2"}
        node1_data = json.dumps({
            "node_id": "worker-1",
            "endpoint": "http://worker-1.luminapv.local:8001",
            "status": "ONLINE",
            "role": "LEADER"
        })
        node2_data = json.dumps({
            "node_id": "worker-2",
            "endpoint": "http://worker-2.luminapv.local:8002",
            "status": "ONLINE",
            "role": "FOLLOWER"
        })
        mock_redis.get.side_effect = lambda key: node1_data if "worker-1" in key else node2_data

        resolver = gateway_module.ServiceCatalogResolver(
            redis_url="redis://localhost:6379",
            default_url="http://localhost:8001",
            service_name="service://luminapv-vision-worker"
        )
        resolver.redis_client = mock_redis

        endpoints = asyncio.run(resolver.get_healthy_endpoints())
        self.assertEqual(len(endpoints), 2)
        self.assertIn("http://worker-1.luminapv.local:8001", endpoints)
        self.assertIn("http://worker-2.luminapv.local:8002", endpoints)

    def test_gateway_automatic_failover(self):
        """dispatch_module_to_worker performs automatic failover if the primary worker is unreachable."""
        calls = []

        async def mock_post(url, files):
            calls.append(url)
            if "worker-1" in url:
                # Primary fails with connection error
                raise httpx.ConnectError("Connection refused to worker-1")
            # Failover worker succeeds
            mock_resp = unittest.mock.MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "module_id": "MOD-FAILOVER",
                "filename": "scan.png",
                "status": "PASS",
                "processed_by_node": "worker-2"
            }
            return mock_resp

        mock_client = AsyncMock()
        mock_client.post.side_effect = mock_post

        result = asyncio.run(gateway_module.dispatch_module_to_worker(
            client=mock_client,
            file_bytes=b"fake_bytes",
            filename="scan.png",
            content_type="image/png",
            target_worker_url="http://worker-1.luminapv.local:8001",
            candidate_worker_urls=["http://worker-1.luminapv.local:8001", "http://worker-2.luminapv.local:8002"]
        ))

        self.assertEqual(result["module_id"], "MOD-FAILOVER")
        self.assertEqual(result["processed_by_node"], "worker-2")
        self.assertEqual(len(calls), 2)
        self.assertIn("worker-1.luminapv.local:8001", calls[0])
        self.assertIn("worker-2.luminapv.local:8002", calls[1])

    def test_gateway_round_robin_balancing(self):
        """analyze_bulk balances module requests across workers in Round-Robin order."""
        dispatched_targets = []

        async def tracking_dispatch(client, file_bytes, filename, content_type, target_worker_url=None, candidate_worker_urls=None):
            dispatched_targets.append(target_worker_url)
            return {
                "module_id": f"MOD-{filename}",
                "filename": filename,
                "status": "PASS",
                "rows": 6,
                "cols": 10,
                "total_cells": 60,
                "defective_cells_count": 0,
                "module_damage_percentage": 0.0,
                "processed_by_node": "worker",
                "cells": []
            }

        # Mock resolver to return two workers
        with patch.object(
            gateway_module.resolver,
            "get_healthy_endpoints",
            AsyncMock(return_value=["http://worker-1:8001", "http://worker-2:8002"])
        ):
            with patch("node1_gateway.main.dispatch_module_to_worker", side_effect=tracking_dispatch):
                files = [
                    ("files", ("scan_a.png", b"bytes_a", "image/png")),
                    ("files", ("scan_b.png", b"bytes_b", "image/png")),
                    ("files", ("scan_c.png", b"bytes_c", "image/png")),
                    ("files", ("scan_d.png", b"bytes_d", "image/png"))
                ]
                gateway_module.resolver.rr_index = 0
                resp = self.gateway_client.post("/api/v1/analyze/bulk", files=files)
                self.assertEqual(resp.status_code, 200)

                # Expect alternating Round-Robin assignment
                self.assertEqual(len(dispatched_targets), 4)
                self.assertEqual(dispatched_targets[0], "http://worker-1:8001")
                self.assertEqual(dispatched_targets[1], "http://worker-2:8002")
                self.assertEqual(dispatched_targets[2], "http://worker-1:8001")
                self.assertEqual(dispatched_targets[3], "http://worker-2:8002")

    # -------------------------------------------------------------------------
    # Issue #4: Cluster Telemetry Endpoint Tests
    # -------------------------------------------------------------------------
    def test_gateway_cluster_status_endpoint(self):
        """Gateway GET /api/v1/cluster/status exposes complete distributed telemetry."""
        response = self.gateway_client.get("/api/v1/cluster/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["status"], "healthy")
        self.assertIn("leader_id", data)
        self.assertIn("term", data)
        self.assertIn("naming_scheme", data)
        self.assertEqual(data["naming_scheme"]["service_name"], "service://luminapv-vision-worker")
        self.assertEqual(data["naming_scheme"]["dns_domain"], "luminapv.local")
        self.assertIn("consistency_model", data)
        self.assertIn("nodes", data)
        self.assertIsInstance(data["nodes"], list)
        self.assertIn("active_locks", data)
        self.assertIsInstance(data["active_locks"], list)
        self.assertIn("replication_stats", data)

    def test_gateway_cluster_election_endpoint(self):
        """Gateway POST /api/v1/cluster/election triggers re-election simulation."""
        response = self.gateway_client.post("/api/v1/cluster/election?action=step_down")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "election_triggered")
        self.assertEqual(data["action"], "step_down")
        self.assertIn("message", data)

    # -------------------------------------------------------------------------
    # Issue #15: Authentication System & S3 Repository Adapter Tests
    # -------------------------------------------------------------------------
    def test_auth_login_default_operator_and_verify_me(self):
        """POST /api/auth/login authenticates default seeded operator and GET /api/auth/me validates JWT."""
        login_resp = self.gateway_client.post(
            "/api/auth/login",
            json={"email": "operador@luminapv.local", "password": "LuminaPV@2026"},
        )
        self.assertEqual(login_resp.status_code, 200)
        payload = login_resp.json()
        self.assertIn("access_token", payload)
        self.assertEqual(payload["token_type"], "Bearer")
        self.assertEqual(payload["user"]["email"], "operador@luminapv.local")

        token = payload["access_token"]
        me_resp = self.gateway_client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(me_resp.status_code, 200)
        me_data = me_resp.json()
        self.assertEqual(me_data["email"], "operador@luminapv.local")

    def test_auth_register_duplicate_and_logout_revocation(self):
        """POST /api/auth/register creates new operator, rejects duplicate email (409), and POST /api/auth/logout revokes JWT."""
        reg_resp = self.gateway_client.post(
            "/api/auth/register",
            json={
                "name": "Engenheira Solar",
                "email": "eng.solar@luminapv.local",
                "password": "StrongPassword!123",
                "role": "engineer",
            },
        )
        self.assertEqual(reg_resp.status_code, 201)
        reg_data = reg_resp.json()
        token = reg_data["access_token"]
        self.assertEqual(reg_data["user"]["email"], "eng.solar@luminapv.local")
        self.assertEqual(reg_data["user"]["role"], "engineer")

        # Duplicate registration must return 409 Conflict
        dup_resp = self.gateway_client.post(
            "/api/auth/register",
            json={
                "name": "Engenheira Solar Dup",
                "email": "eng.solar@luminapv.local",
                "password": "StrongPassword!123",
                "role": "engineer",
            },
        )
        self.assertEqual(dup_resp.status_code, 409)

        # Logout revokes token jti
        logout_resp = self.gateway_client.post(
            "/api/auth/logout",
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(logout_resp.status_code, 200)
        self.assertEqual(logout_resp.json()["status"], "logged_out")

        # Subsequent /api/auth/me with revoked token must return 401
        me_after_logout = self.gateway_client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(me_after_logout.status_code, 401)

    def test_s3_user_repository_adapter_contract(self):
        """S3UserRepository persists and retrieves user records via S3 object keys and falls back gracefully."""
        from node1_gateway.auth import S3UserRepository, InMemoryUserRepository, UserRecord, hash_password
        import io

        stored_objects = {}

        mock_s3 = unittest.mock.MagicMock()

        def fake_put_object(Bucket, Key, Body, ContentType):
            stored_objects[(Bucket, Key)] = Body

        def fake_get_object(Bucket, Key):
            if (Bucket, Key) not in stored_objects:
                raise Exception("NoSuchKey")
            return {"Body": io.BytesIO(stored_objects[(Bucket, Key)])}

        mock_s3.put_object.side_effect = fake_put_object
        mock_s3.get_object.side_effect = fake_get_object

        repo = S3UserRepository(
            bucket_name="luminapv-auth-bucket",
            prefix="auth",
            s3_client=mock_s3,
            fallback_repo=InMemoryUserRepository(seed_default=False),
        )

        record = UserRecord(
            user_id="usr-test-s3-001",
            email="s3test@luminapv.local",
            name="S3 Test Operator",
            role="admin",
            password_hash=hash_password("S3SecretPass!99"),
            created_at="2026-04-17T12:00:00+00:00",
            updated_at="2026-04-17T12:00:00+00:00",
        )
        asyncio.run(repo.create_user(record))

        # Verify both primary object and email index object were written to S3
        self.assertTrue(
            any(k.endswith("auth/users/by-id/usr-test-s3-001.json") for (_, k) in stored_objects.keys())
        )
        self.assertTrue(
            any("auth/users/by-email/" in k for (_, k) in stored_objects.keys())
        )

        fetched_by_email = asyncio.run(repo.get_user_by_email("s3test@luminapv.local"))
        self.assertIsNotNone(fetched_by_email)
        self.assertEqual(fetched_by_email.user_id, "usr-test-s3-001")
        self.assertEqual(fetched_by_email.role, "admin")

    # -------------------------------------------------------------------------
    # User Analysis History & S3 Image Gallery Tests
    # -------------------------------------------------------------------------
    def test_user_bulk_upload_partitions_by_user_id(self):
        """Confirms S3 put_object is called with keys starting with users/<user_id>/ when authenticated, and anonymous/ when unauthenticated."""
        asyncio.run(gateway_module.history_service.memory_repo.clear())
        self.mock_s3.put_object.reset_mock()
        self.mock_s3.put_object.side_effect = None

        login_resp = self.gateway_client.post(
            "/api/auth/login",
            json={"email": "operador@luminapv.local", "password": "LuminaPV@2026"},
        )
        self.assertEqual(login_resp.status_code, 200)
        login_data = login_resp.json()
        token = login_data["access_token"]
        user_id = login_data["user"]["user_id"]

        async def mock_worker_dispatch(client, file_bytes, filename, content_type, **kwargs):
            return {
                "module_id": "MOD-PART-001",
                "filename": filename,
                "status": "FAIL",
                "rows": 6,
                "cols": 10,
                "total_cells": 60,
                "defective_cells_count": 3,
                "module_damage_percentage": 5.0,
                "processed_by_node": "worker-1",
                "cells": [{"index": 0, "row": 0, "col": 0, "is_defective": True, "defect_type": "micro-crack", "confidence": 0.92}],
            }

        with patch("node1_gateway.main.dispatch_module_to_worker", side_effect=mock_worker_dispatch):
            files = [("files", ("module_el_01.png", b"raw_image_bytes_01", "image/png"))]
            resp = self.gateway_client.post(
                "/api/v1/analyze/bulk",
                files=files,
                headers={"Authorization": f"Bearer {token}"},
            )
            self.assertEqual(resp.status_code, 200)
            body = resp.json()
            self.assertEqual(len(body["modules"]), 1)
            mod = body["modules"][0]
            self.assertTrue(mod["s3_image_key"].startswith(f"users/{user_id}/scans/"))
            self.assertTrue(mod["s3_report_key"].startswith(f"users/{user_id}/reports/"))

            put_keys = [
                call.kwargs.get("Key") or (call.args[1] if len(call.args) > 1 else "")
                for call in self.mock_s3.put_object.call_args_list
            ]
            self.assertTrue(any(k.startswith(f"users/{user_id}/scans/") and k.endswith("_module_el_01.png") for k in put_keys))
            self.assertTrue(any(k.startswith(f"users/{user_id}/reports/") and k.endswith("_MOD-PART-001.json") for k in put_keys))

            # Verify unauthenticated upload is NOT persisted to S3 (kept local-only)
            self.mock_s3.put_object.reset_mock()
            anon_resp = self.gateway_client.post(
                "/api/v1/analyze/bulk",
                files=[("files", ("anon_scan.png", b"anon_bytes", "image/png"))],
            )
            self.assertEqual(anon_resp.status_code, 200)
            anon_mod = anon_resp.json()["modules"][0]
            self.assertFalse(anon_mod.get("persisted_in_s3"))
            self.assertEqual(anon_mod.get("storage_scope"), "local")
            self.assertIsNone(anon_mod.get("s3_image_key"))
            self.assertIsNone(anon_mod.get("s3_report_key"))
            self.assertEqual(self.mock_s3.put_object.call_count, 0)

            # Verify syncing local unauthenticated modules upon login saves them to S3 without reprocessing
            sync_resp = self.gateway_client.post(
                "/api/v1/user/history/sync",
                json={"modules": [anon_mod]},
                headers={"Authorization": f"Bearer {token}"},
            )
            self.assertEqual(sync_resp.status_code, 200)
            sync_keys = [
                call.kwargs.get("Key") or (call.args[1] if len(call.args) > 1 else "")
                for call in self.mock_s3.put_object.call_args_list
            ]
            self.assertTrue(any(k.startswith(f"users/{user_id}/scans/") and k.endswith("_anon_scan.png") for k in sync_keys))
            self.assertTrue(any(k.startswith(f"users/{user_id}/reports/") and k.endswith(".json") for k in sync_keys))

    def test_get_user_history_requires_auth(self):
        """Confirms HTTP 401 when accessing /api/v1/user/history without a valid Bearer JWT token."""
        resp_list = self.gateway_client.get("/api/v1/user/history")
        self.assertEqual(resp_list.status_code, 401)

        resp_detail = self.gateway_client.get("/api/v1/user/history/MOD-2026-001")
        self.assertEqual(resp_detail.status_code, 401)

        resp_delete = self.gateway_client.delete("/api/v1/user/history/MOD-2026-001")
        self.assertEqual(resp_delete.status_code, 401)

    def test_get_user_history_returns_user_scans(self):
        """Verifies that history returns only the scans belonging to the logged-in operator, with pre-signed S3 URLs, full detail reload, and deletion."""
        asyncio.run(gateway_module.history_service.memory_repo.clear())

        # Authenticate Operator 1 (default operator)
        login_op1 = self.gateway_client.post(
            "/api/auth/login",
            json={"email": "operador@luminapv.local", "password": "LuminaPV@2026"},
        )
        self.assertEqual(login_op1.status_code, 200)
        token_op1 = login_op1.json()["access_token"]
        uid_op1 = login_op1.json()["user"]["user_id"]

        # Register & authenticate Operator 2
        reg_op2 = self.gateway_client.post(
            "/api/auth/register",
            json={
                "name": "Operador Dois",
                "email": "op2.history@luminapv.local",
                "password": "Op2Password!2026",
                "role": "operator",
            },
        )
        if reg_op2.status_code == 409:
            reg_op2 = self.gateway_client.post(
                "/api/auth/login",
                json={"email": "op2.history@luminapv.local", "password": "Op2Password!2026"},
            )
        self.assertIn(reg_op2.status_code, (200, 201))
        token_op2 = reg_op2.json()["access_token"]
        uid_op2 = reg_op2.json()["user"]["user_id"]

        self.mock_s3.generate_presigned_url.side_effect = (
            lambda method, Params, ExpiresIn=3600: f"https://{Params['Bucket']}.s3.us-east-2.amazonaws.com/{Params['Key']}?X-Amz-Expires={ExpiresIn}&sig=mock"
        )

        async def mock_worker_dispatch(client, file_bytes, filename, content_type, **kwargs):
            mod_id = "MOD-OP1-SCAN" if "op1" in filename else "MOD-OP2-SCAN"
            return {
                "module_id": mod_id,
                "filename": filename,
                "status": "FAIL" if "op1" in filename else "PASS",
                "rows": 6,
                "cols": 10,
                "total_cells": 60,
                "defective_cells_count": 4 if "op1" in filename else 0,
                "module_damage_percentage": 6.7 if "op1" in filename else 0.0,
                "processed_by_node": "worker-1",
                "cells": [
                    {
                        "index": 12,
                        "row": 1,
                        "col": 2,
                        "is_defective": True,
                        "defect_type": "micro-crack",
                        "confidence": 0.94,
                        "bbox": {"x": 10, "y": 20, "width": 50, "height": 50, "x_pct": 20.0, "y_pct": 16.6, "width_pct": 10.0, "height_pct": 16.6},
                    }
                ] if "op1" in filename else [],
            }

        with patch("node1_gateway.main.dispatch_module_to_worker", side_effect=mock_worker_dispatch):
            # Upload scan as Operator 1
            up1 = self.gateway_client.post(
                "/api/v1/analyze/bulk",
                files=[("files", ("op1_panel.png", b"op1_img_bytes", "image/png"))],
                headers={"Authorization": f"Bearer {token_op1}"},
            )
            self.assertEqual(up1.status_code, 200)

            # Upload scan as Operator 2
            up2 = self.gateway_client.post(
                "/api/v1/analyze/bulk",
                files=[("files", ("op2_panel.png", b"op2_img_bytes", "image/png"))],
                headers={"Authorization": f"Bearer {token_op2}"},
            )
            self.assertEqual(up2.status_code, 200)

        # Operator 1 fetches their history -> must only see MOD-OP1-SCAN
        hist_op1 = self.gateway_client.get(
            "/api/v1/user/history?limit=50",
            headers={"Authorization": f"Bearer {token_op1}"},
        )
        self.assertEqual(hist_op1.status_code, 200)
        hist_data_1 = hist_op1.json()
        self.assertEqual(hist_data_1["user_id"], uid_op1)
        self.assertEqual(len(hist_data_1["items"]), 1)
        item1 = hist_data_1["items"][0]
        self.assertEqual(item1["module_id"], "MOD-OP1-SCAN")
        self.assertEqual(item1["filename"], "op1_panel.png")
        self.assertTrue(item1["s3_image_key"].startswith(f"users/{uid_op1}/scans/"))
        self.assertIn(f"users/{uid_op1}/scans/", item1["image_url"])
        self.assertIn("X-Amz-Expires=3600", item1["presigned_url"])

        # Operator 1 fetches full module detail for interactive canvas reload
        detail_op1 = self.gateway_client.get(
            "/api/v1/user/history/MOD-OP1-SCAN",
            headers={"Authorization": f"Bearer {token_op1}"},
        )
        self.assertEqual(detail_op1.status_code, 200)
        detail_json = detail_op1.json()
        self.assertEqual(detail_json["module_id"], "MOD-OP1-SCAN")
        self.assertEqual(len(detail_json["cells"]), 1)
        self.assertEqual(detail_json["cells"][0]["defect_type"], "micro-crack")

        # Operator 1 cannot access Operator 2's module detail
        forbidden_detail = self.gateway_client.get(
            "/api/v1/user/history/MOD-OP2-SCAN",
            headers={"Authorization": f"Bearer {token_op1}"},
        )
        self.assertEqual(forbidden_detail.status_code, 404)

        # Operator 1 deletes their scan
        del_resp = self.gateway_client.delete(
            "/api/v1/user/history/MOD-OP1-SCAN",
            headers={"Authorization": f"Bearer {token_op1}"},
        )
        self.assertEqual(del_resp.status_code, 200)
        self.assertEqual(del_resp.json()["status"], "deleted")

        # History for Operator 1 is now empty
        hist_after_del = self.gateway_client.get(
            "/api/v1/user/history",
            headers={"Authorization": f"Bearer {token_op1}"},
        )
        self.assertEqual(hist_after_del.status_code, 200)
        self.assertEqual(len(hist_after_del.json()["items"]), 0)

        self.mock_s3.generate_presigned_url.side_effect = None


if __name__ == "__main__":
    unittest.main()
