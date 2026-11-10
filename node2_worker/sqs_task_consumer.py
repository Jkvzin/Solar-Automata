"""
LuminaPV Worker Work-Queue Task Consumer (Node 2 Worker)
========================================================
Implements:
- Competing Consumer pattern for asynchronous module inspection work queues
- AWS SQS Long Polling (WaitTimeSeconds=20, MaxNumberOfMessages=1)
- Seamless offline/local fallback via Redis List ('luminapv:queue:tasks') and shared memory queue
- Distributed cache check (LRU & Redis) before running YOLOv8-cls inference
- In-memory/S3 raw image fetching ('raw_uploads/{batch_id}/{scan_id}.png')
- Perspective homography, grid slicing & YOLOv8-cls execution via cv_engine
- S3 annotated image persistence ('inspections/{batch_id}/{scan_id}_annotated.png')
- Atomic distributed batch status update in Redis ('luminapv:batch:{batch_id}')
- Message deletion via SQS ReceiptHandle or Redis LPOP to ensure queue health
"""

import asyncio
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
import sys
from typing import Any, Callable, Dict, List, Optional

# Ensure identical module identity across different sys.path import roots
if __name__ == "node2_worker.sqs_task_consumer" and "sqs_task_consumer" not in sys.modules:
    sys.modules["sqs_task_consumer"] = sys.modules[__name__]
elif __name__ == "sqs_task_consumer" and "node2_worker.sqs_task_consumer" not in sys.modules:
    sys.modules["node2_worker.sqs_task_consumer"] = sys.modules[__name__]

import redis.asyncio as aioredis

from cv_engine import evaluate_module, get_model_sha256, ModuleResult

logger = logging.getLogger("node2_worker.task_consumer")

# Shared in-memory fallback storage for unit testing or single-process execution
_SHARED_LOCAL_RAW_STORAGE: Dict[str, bytes] = {}
_SHARED_LOCAL_TASK_QUEUE: List[Dict[str, Any]] = []
_SHARED_LOCAL_BATCH_STORE: Dict[str, Dict[str, Any]] = {}
_SHARED_LOCK = asyncio.Lock()


class WorkerTaskConsumer:
    """
    Competing Consumer service that polls inspection task messages from AWS SQS
    or the Redis fallback task queue, coordinates cache verification, executes
    neural optical inspection, and atomically updates batch state.
    """

    def __init__(
        self,
        node_id: str,
        redis_url: str = "redis://state.luminapv.local:6379/0",
        queue_url: Optional[str] = None,
        bucket_name: Optional[str] = None,
        aws_region: str = "us-east-2",
        aws_ingestion_enabled: bool = False,
        replicator: Optional[Any] = None,
        poll_interval: float = 0.5
    ) -> None:
        self.node_id = node_id
        self.redis_url = redis_url
        self.queue_url = queue_url or os.getenv(
            "AWS_SQS_QUEUE_URL",
            "https://sqs.us-east-2.amazonaws.com/058835009582/fila-imagens-paineis-solares"
        )
        self.bucket_name = bucket_name or os.getenv(
            "AWS_S3_BUCKET",
            "dataset-paineis-solares-joaoguilherme-joaovictor-gustavo-2026"
        )
        self.aws_region = aws_region
        self.aws_ingestion_enabled = aws_ingestion_enabled
        self.replicator = replicator
        self.poll_interval = poll_interval

        self.redis_client: Optional[aioredis.Redis] = None
        self._s3_client: Optional[Any] = None
        self._sqs_client: Optional[Any] = None

        self._running: bool = False
        self._task: Optional[asyncio.Task] = None
        self.tasks_processed: int = 0
        self.tasks_failed: int = 0
        self.cache_hits: int = 0

        self._init_aws_clients()

    def _init_aws_clients(self) -> None:
        if self.aws_ingestion_enabled:
            try:
                import boto3
                self._s3_client = boto3.client("s3", region_name=self.aws_region)
                self._sqs_client = boto3.client("sqs", region_name=self.aws_region)
                logger.info(f"[{self.node_id}] AWS S3 & SQS clients initialized for worker consumption.")
            except Exception as e:
                logger.warning(f"[{self.node_id}] Failed to initialize AWS clients in worker: {e}")
                self._s3_client = None
                self._sqs_client = None

    async def get_redis(self) -> Optional[aioredis.Redis]:
        if self.redis_client is None:
            try:
                self.redis_client = aioredis.from_url(
                    self.redis_url,
                    decode_responses=False,  # Byte-safe for raw image cache
                    socket_connect_timeout=1.0,
                    socket_timeout=1.0
                )
            except Exception:
                self.redis_client = None
        return self.redis_client

    async def fetch_raw_image(self, s3_image_key: str) -> Optional[bytes]:
        """
        Retrieves raw image binary from AWS S3, Redis buffer cache, or memory fallback.
        """
        # 1. Try AWS S3 if enabled
        if self._s3_client is not None and self.bucket_name and s3_image_key:
            try:
                resp = await asyncio.to_thread(
                    self._s3_client.get_object,
                    Bucket=self.bucket_name,
                    Key=s3_image_key
                )
                body = resp.get("Body")
                if body:
                    data = body.read()
                    if data:
                        return data
            except Exception as exc:
                logger.debug(f"[{self.node_id}] S3 fetch error for '{s3_image_key}': {exc}")

        # 2. Try Redis raw buffer cache
        client = await self.get_redis()
        if client is not None:
            try:
                cached_bytes = await client.get(f"luminapv:raw:{s3_image_key}")
                if cached_bytes:
                    return cached_bytes
            except Exception:
                pass

        # 3. Fallback to in-memory shared store (unit tests / local fallback)
        async with _SHARED_LOCK:
            mem_bytes = _SHARED_LOCAL_RAW_STORAGE.get(s3_image_key)
            if mem_bytes:
                return mem_bytes

        return None

    async def update_batch_tracker(
        self,
        batch_id: str,
        module_result: Optional[Dict[str, Any]],
        is_failure: bool = False,
        error_msg: Optional[str] = None
    ) -> None:
        """
        Atomically updates the batch status in Redis (luminapv:batch:{batch_id})
        and synchronizes with in-memory fallback.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        client = await self.get_redis()

        if client is not None:
            try:
                batch_key = f"luminapv:batch:{batch_id}"
                raw = await client.get(batch_key)
                if raw:
                    batch_dict = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
                else:
                    batch_dict = {
                        "batch_id": batch_id,
                        "total_modules": 1,
                        "completed_modules": 0,
                        "failed_modules": 0,
                        "status": "PROCESSING",
                        "created_at": now_iso,
                        "results": []
                    }

                if is_failure:
                    batch_dict["failed_modules"] = batch_dict.get("failed_modules", 0) + 1
                    if error_msg:
                        batch_dict.setdefault("errors", []).append(error_msg)
                else:
                    batch_dict["completed_modules"] = batch_dict.get("completed_modules", 0) + 1
                    if module_result:
                        batch_dict.setdefault("results", []).append(module_result)

                batch_dict["updated_at"] = now_iso
                total = batch_dict.get("total_modules", 1)
                done = batch_dict.get("completed_modules", 0) + batch_dict.get("failed_modules", 0)
                if done >= total:
                    batch_dict["status"] = "COMPLETED"

                await client.set(batch_key, json.dumps(batch_dict), ex=86400)
            except Exception as e:
                logger.debug(f"[{self.node_id}] Redis batch update fallback for '{batch_id}': {e}")

        # Update in-memory fallback store
        async with _SHARED_LOCK:
            b = _SHARED_LOCAL_BATCH_STORE.setdefault(batch_id, {
                "batch_id": batch_id,
                "total_modules": 1,
                "completed_modules": 0,
                "failed_modules": 0,
                "status": "PROCESSING",
                "created_at": now_iso,
                "results": []
            })
            if is_failure:
                b["failed_modules"] = b.get("failed_modules", 0) + 1
                if error_msg:
                    b.setdefault("errors", []).append(error_msg)
            else:
                b["completed_modules"] = b.get("completed_modules", 0) + 1
                if module_result:
                    b.setdefault("results", []).append(module_result)

            b["updated_at"] = now_iso
            total = b.get("total_modules", 1)
            done = b.get("completed_modules", 0) + b.get("failed_modules", 0)
            if done >= total:
                b["status"] = "COMPLETED"

    async def process_task_payload(
        self,
        task: Dict[str, Any],
        receipt_handle: Optional[str] = None
    ) -> bool:
        """
        Executes a single inspection task: cache lookup, raw image download,
        YOLOv8-cls inference, replication, batch update, and message deletion.
        """
        batch_id = str(task.get("batch_id") or "batch-unknown")
        scan_id = str(task.get("scan_id") or "scan-unknown")
        filename = str(task.get("filename") or f"{scan_id}.png")
        s3_image_key = str(task.get("s3_image_key") or f"raw_uploads/{batch_id}/{scan_id}.png")
        content_sha256 = task.get("content_sha256")

        logger.info(
            f"[{self.node_id}] Consuming task: batch={batch_id}, scan={scan_id}, file='{filename}'"
        )

        image_bytes: Optional[bytes] = None
        result_payload: Optional[Dict[str, Any]] = None

        # 1. Check distributed replica cache by content SHA-256 if available
        if self.replicator is not None and content_sha256:
            try:
                cached_replica = await self.replicator.get_replica_by_sha256(content_sha256)
                if cached_replica:
                    logger.info(
                        f"[{self.node_id}] Cache HIT for task '{scan_id}' (sha256={content_sha256[:12]}...)"
                    )
                    self.cache_hits += 1
                    result_payload = dict(cached_replica)
                    result_payload["filename"] = filename
                    result_payload["cache_hit"] = True
            except Exception:
                pass

        # 2. If Cache Miss, fetch raw image bytes and execute CV inspection
        if result_payload is None:
            image_bytes = await self.fetch_raw_image(s3_image_key)
            if not image_bytes or len(image_bytes) == 0:
                err = f"Raw image stream not found for key: {s3_image_key}"
                logger.warning(f"[{self.node_id}] Task failure: {err}")
                await self.update_batch_tracker(batch_id, None, is_failure=True, error_msg=err)
                self.tasks_failed += 1
                await self._delete_sqs_message(receipt_handle)
                return False

            try:
                # Check cache with raw bytes (validating model SHA-256 consistency)
                if self.replicator is not None:
                    cached = await self.replicator.get_cached_result(
                        image_bytes,
                        filename=filename,
                        use_yolo=True
                    )
                    if cached is not None:
                        result_payload = json.loads(cached.model_dump_json())
                        self.cache_hits += 1

                if result_payload is None:
                    # Run computer vision evaluation pipeline
                    res = evaluate_module(
                        image_bytes,
                        filename=filename,
                        use_yolo=True,
                        node_id=self.node_id
                    )
                    if self.replicator is not None:
                        res = await self.replicator.replicate_result(res, use_yolo=True)
                    result_payload = json.loads(res.model_dump_json())

                    # Persist secondary index in Redis: luminapv:replica:result:<content_sha256>
                    client = await self.get_redis()
                    if client is not None and res.content_sha256:
                        try:
                            await client.set(
                                f"luminapv:replica:result:{res.content_sha256}",
                                json.dumps(result_payload),
                                ex=86400
                            )
                        except Exception:
                            pass

                # Optionally persist to history service if available in runtime environment
                try:
                    from node1_gateway.history import history_service
                    from node1_gateway.auth import UserRecord
                    user_rec = UserRecord(
                        user_id=task.get("user_id", "anonymous"),
                        email=f"{task.get('user_id', 'anonymous')}@luminapv.local",
                        name="Operator",
                        password_hash="",
                        created_at=datetime.now(timezone.utc).isoformat(),
                        updated_at=datetime.now(timezone.utc).isoformat()
                    ) if task.get("user_id") and task.get("user_id") != "anonymous" else None
                    if history_service is not None:
                        await history_service.persist_module_inspection(
                            module_data=result_payload,
                            image_bytes=image_bytes or b"",
                            filename=filename,
                            content_type="image/png",
                            user=user_rec,
                            scan_id=scan_id
                        )
                except Exception as hist_err:
                    logger.debug(f"History persistence hook fallback: {hist_err}")

                # Optionally persist annotated image to S3 if available
                if self._s3_client is not None and self.bucket_name and result_payload.get("rectified_image"):
                    try:
                        import base64
                        b64_data = result_payload["rectified_image"].split(",")[-1]
                        annotated_bytes = base64.b64decode(b64_data)
                        annotated_key = f"inspections/{batch_id}/{scan_id}_annotated.png"
                        await asyncio.to_thread(
                            self._s3_client.put_object,
                            Bucket=self.bucket_name,
                            Key=annotated_key,
                            Body=annotated_bytes,
                            ContentType="image/png"
                        )
                    except Exception as s3_annot_err:
                        logger.debug(f"Failed to persist annotated image to S3: {s3_annot_err}")

            except Exception as cv_err:
                logger.error(f"[{self.node_id}] CV processing error for '{filename}': {cv_err}")
                await self.update_batch_tracker(batch_id, None, is_failure=True, error_msg=str(cv_err))
                self.tasks_failed += 1
                await self._delete_sqs_message(receipt_handle)
                return False

        # 3. Update batch tracker with inspection result
        await self.update_batch_tracker(batch_id, result_payload, is_failure=False)
        self.tasks_processed += 1

        # 4. Acknowledge and delete message from queue
        await self._delete_sqs_message(receipt_handle)
        logger.info(
            f"[{self.node_id}] Successfully inspected & acknowledged '{filename}' for batch '{batch_id}'"
        )
        return True

    async def _delete_sqs_message(self, receipt_handle: Optional[str]) -> None:
        if self._sqs_client is not None and receipt_handle and self.queue_url:
            try:
                await asyncio.to_thread(
                    self._sqs_client.delete_message,
                    QueueUrl=self.queue_url,
                    ReceiptHandle=receipt_handle
                )
            except Exception as e:
                logger.warning(f"[{self.node_id}] SQS delete_message error: {e}")

    async def poll_and_process_one(self) -> bool:
        """
        Attempts to dequeue and process a single task from AWS SQS,
        the Redis task queue, or the shared memory queue.
        Returns True if a task was processed, False if queue was empty.
        """
        # 1. Try polling from AWS SQS (Long Polling if enabled)
        if self._sqs_client is not None and self.queue_url:
            try:
                resp = await asyncio.to_thread(
                    self._sqs_client.receive_message,
                    QueueUrl=self.queue_url,
                    MaxNumberOfMessages=1,
                    WaitTimeSeconds=1,  # Short polling loop in poll_and_process_one
                    VisibilityTimeout=60,
                    AttributeNames=["All"]
                )
                messages = resp.get("Messages", []) if isinstance(resp, dict) else []
                if messages:
                    msg = messages[0]
                    body_raw = msg.get("Body", "{}")
                    task_data = json.loads(body_raw) if isinstance(body_raw, str) else body_raw
                    return await self.process_task_payload(task_data, receipt_handle=msg.get("ReceiptHandle"))
            except Exception as sqs_err:
                logger.debug(f"[{self.node_id}] SQS receive_message error: {sqs_err}")

        # 2. Try polling from Redis task list (luminapv:queue:tasks)
        client = await self.get_redis()
        if client is not None:
            try:
                raw_task = await client.lpop("luminapv:queue:tasks")
                if raw_task:
                    task_str = raw_task.decode("utf-8") if isinstance(raw_task, bytes) else raw_task
                    task_data = json.loads(task_str)
                    return await self.process_task_payload(task_data)
            except Exception:
                pass

        # 3. Try polling from shared in-memory task queue
        task_data: Optional[Dict[str, Any]] = None
        async with _SHARED_LOCK:
            if _SHARED_LOCAL_TASK_QUEUE:
                task_data = _SHARED_LOCAL_TASK_QUEUE.pop(0)

        if task_data is not None:
            return await self.process_task_payload(task_data)

        return False

    async def _consumer_loop(self) -> None:
        logger.info(f"[{self.node_id}] Work Queue Task Consumer started.")
        while self._running:
            try:
                did_work = await self.poll_and_process_one()
                if not did_work:
                    await asyncio.sleep(self.poll_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"[{self.node_id}] Error in worker task consumer loop: {e}")
                await asyncio.sleep(1.0)

        logger.info(f"[{self.node_id}] Work Queue Task Consumer stopped.")

    async def start(self) -> None:
        if self._running or (self._task is not None and not self._task.done()):
            return
        self._running = True
        self._task = asyncio.create_task(self._consumer_loop())

    async def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    def get_stats(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "running": self._running,
            "aws_ingestion_enabled": self.aws_ingestion_enabled,
            "tasks_processed": self.tasks_processed,
            "tasks_failed": self.tasks_failed,
            "cache_hits": self.cache_hits
        }
