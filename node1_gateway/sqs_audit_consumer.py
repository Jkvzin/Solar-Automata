"""
LuminaPV AWS SQS Audit Consumer & Event Processor (Node 1 Gateway)
==================================================================
Implements:
- Long Polling consumer for AWS SQS queue ('fila-imagens-paineis-solares')
- Canonical Event Schema parsing for 'MODULE_INSPECTED' events
- Structured audit event logging ('AUDIT_EVENT: Módulo {module_id} processado com status {status}')
- Message deletion via ReceiptHandle to prevent queue accumulation
- Non-blocking asyncio background loop integrated into FastAPI lifespan
- Resilient conditional activation (only active when AWS_INGESTION_ENABLED=true)
"""

import asyncio
from datetime import datetime, timezone
import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("node1_gateway.sqs_audit")

DEFAULT_QUEUE_URL = os.getenv(
    "AWS_SQS_QUEUE_URL",
    "https://sqs.us-east-2.amazonaws.com/058835009582/fila-imagens-paineis-solares"
)
DEFAULT_AWS_REGION = os.getenv("AWS_DEFAULT_REGION", os.getenv("AWS_REGION", "us-east-2"))
SQS_WAIT_TIME_SECONDS = int(os.getenv("SQS_WAIT_TIME_SECONDS", "20"))
SQS_MAX_MESSAGES = int(os.getenv("SQS_MAX_MESSAGES", "10"))
SQS_VISIBILITY_TIMEOUT = int(os.getenv("SQS_VISIBILITY_TIMEOUT", "30"))


class SQSAuditConsumer:
    """
    Resilient background consumer for AWS SQS optical inspection audit events.
    Consumes canonical JSON events, emits structured audit logs, and deletes processed
    messages to maintain clean queue health.
    """

    def __init__(
        self,
        queue_url: Optional[str] = None,
        sqs_client_provider: Optional[Callable[[], Any]] = None,
        enabled: bool = False,
        poll_wait_time: int = SQS_WAIT_TIME_SECONDS,
        max_messages: int = SQS_MAX_MESSAGES,
        visibility_timeout: int = SQS_VISIBILITY_TIMEOUT
    ) -> None:
        self.queue_url = queue_url or DEFAULT_QUEUE_URL
        self._sqs_client_provider = sqs_client_provider
        self.enabled = enabled
        self.poll_wait_time = poll_wait_time
        self.max_messages = max_messages
        self.visibility_timeout = visibility_timeout
        self._running: bool = False
        self._task: Optional[asyncio.Task] = None
        self.messages_processed: int = 0
        self.messages_deleted: int = 0
        self.last_event: Optional[Dict[str, Any]] = None

    def get_sqs_client(self) -> Any:
        if self._sqs_client_provider is not None:
            return self._sqs_client_provider()
        return None

    def parse_event_body(self, raw_body: str) -> Dict[str, Any]:
        """
        Parses raw message body string into normalized audit event dictionary.
        Supports both canonical JSON event schema and legacy plain image_key strings.
        """
        if not raw_body:
            return {
                "event_type": "UNKNOWN",
                "module_id": "UNKNOWN",
                "status": "UNKNOWN",
                "damage_percentage": 0.0,
                "s3_image_key": "",
                "processed_by": "unknown",
                "scan_id": "",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model_sha256": "N/A"
            }

        try:
            parsed = json.loads(raw_body)
            if isinstance(parsed, dict):
                return {
                    "event_type": parsed.get("event_type", "MODULE_INSPECTED"),
                    "timestamp": parsed.get("timestamp", datetime.now(timezone.utc).isoformat()),
                    "module_id": str(parsed.get("module_id") or "UNKNOWN"),
                    "scan_id": str(parsed.get("scan_id") or ""),
                    "status": str(parsed.get("status") or "PASS").upper(),
                    "damage_percentage": float(parsed.get("damage_percentage", 0.0)),
                    "s3_image_key": str(parsed.get("s3_image_key") or ""),
                    "processed_by": str(parsed.get("processed_by") or "worker-1"),
                    "model_sha256": str(parsed.get("model_sha256") or "N/A"),
                }
        except (json.JSONDecodeError, TypeError):
            pass

        # Fallback for legacy plain image_key payload
        return {
            "event_type": "MODULE_INSPECTED_LEGACY",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "module_id": raw_body,
            "scan_id": "legacy",
            "status": "UNSPECIFIED",
            "damage_percentage": 0.0,
            "s3_image_key": raw_body,
            "processed_by": "unknown",
            "model_sha256": "N/A"
        }

    async def process_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Processes a single SQS message: parses event, emits structured audit log,
        and deletes message via ReceiptHandle.
        """
        receipt_handle = message.get("ReceiptHandle")
        raw_body = message.get("Body", "")

        event = self.parse_event_body(raw_body)
        module_id = event["module_id"]
        status = event["status"]
        damage = event["damage_percentage"]
        processed_by = event["processed_by"]
        scan_id = event["scan_id"]
        image_key = event["s3_image_key"]

        # Structured audit log format
        logger.info(
            f"AUDIT_EVENT: Módulo {module_id} processado com status {status} "
            f"(scan_id={scan_id}, damage={damage}%, processed_by={processed_by}, key={image_key})"
        )

        sqs = self.get_sqs_client()
        if sqs is not None and receipt_handle and self.queue_url:
            try:
                await asyncio.to_thread(
                    sqs.delete_message,
                    QueueUrl=self.queue_url,
                    ReceiptHandle=receipt_handle
                )
                self.messages_deleted += 1
            except Exception as del_err:
                logger.warning(
                    f"Failed to delete SQS message {message.get('MessageId')} from queue '{self.queue_url}': {del_err}"
                )

        self.messages_processed += 1
        self.last_event = event
        return event

    async def consume_batch(
        self,
        max_messages: Optional[int] = None,
        wait_time: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Polls a batch of messages from the SQS queue, processes them, and deletes them.
        Returns list of processed audit event dictionaries.
        """
        sqs = self.get_sqs_client()
        if sqs is None or not self.queue_url:
            return []

        num_messages = max_messages or self.max_messages
        poll_wait = wait_time if wait_time is not None else self.poll_wait_time

        try:
            resp = await asyncio.to_thread(
                sqs.receive_message,
                QueueUrl=self.queue_url,
                MaxNumberOfMessages=num_messages,
                WaitTimeSeconds=poll_wait,
                VisibilityTimeout=self.visibility_timeout,
                AttributeNames=["All"]
            )
        except Exception as poll_err:
            logger.warning(f"SQS receive_message error on queue '{self.queue_url}': {poll_err}")
            return []

        messages = resp.get("Messages", []) if isinstance(resp, dict) else []
        if not messages:
            return []

        processed_events: List[Dict[str, Any]] = []
        for msg in messages:
            ev = await self.process_message(msg)
            if ev:
                processed_events.append(ev)

        return processed_events

    async def _consume_loop(self) -> None:
        """
        Non-blocking continuous long-polling loop.
        """
        logger.info(
            f"SQS Audit Consumer started. Polling queue '{self.queue_url}' "
            f"(WaitTimeSeconds={self.poll_wait_time}, MaxMessages={self.max_messages})"
        )
        while self._running:
            try:
                events = await self.consume_batch()
                if not events:
                    # Brief rest before next poll when queue is quiet
                    await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning(f"Unexpected error in SQS audit consumer loop: {exc}. Retrying in 5 seconds...")
                await asyncio.sleep(5.0)

        logger.info("SQS Audit Consumer loop terminated gracefully.")

    async def start(self) -> None:
        """
        Starts the background consumption loop if enabled and queue is configured.
        """
        if not self.enabled:
            logger.debug("SQS Audit Consumer is disabled (AWS_INGESTION_ENABLED=false).")
            return

        if not self.queue_url:
            logger.warning("SQS Audit Consumer cannot start: no queue URL provided.")
            return

        if self._running or (self._task is not None and not self._task.done()):
            return

        self._running = True
        self._task = asyncio.create_task(self._consume_loop())

    async def stop(self) -> None:
        """
        Stops the background consumption loop cleanly without leaking tasks.
        """
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    def get_stats(self) -> Dict[str, Any]:
        """
        Returns telemetry dictionary regarding the consumer status and message count.
        """
        return {
            "enabled": self.enabled,
            "running": self._running,
            "queue_url": self.queue_url,
            "messages_processed": self.messages_processed,
            "messages_deleted": self.messages_deleted,
            "last_event": self.last_event
        }


# Global singleton instance (configured dynamically by node1_gateway/main.py)
sqs_consumer = SQSAuditConsumer(
    queue_url=DEFAULT_QUEUE_URL,
    enabled=False
)
