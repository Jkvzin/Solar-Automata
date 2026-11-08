"""
LuminaPV User Analysis History & S3 Image Gallery Module (Node 1 Gateway)
=========================================================================
Implements:
- User-partitioned S3 object key builder:
  - Original Image: users/{user_id}/scans/{scan_id}_{filename} (or anonymous/scans/...)
  - Analysis Report: users/{user_id}/reports/{scan_id}_{module_id}.json (or anonymous/reports/...)
- Hexagonal Architecture (Ports & Adapters) for History & Media Persistence:
  - InMemoryHistoryRepository (Local/Test fallback with Read-Your-Writes consistency)
  - S3HistoryStorageService (Amazon S3 persistence + Pre-signed URL generation with automatic fallback)
- FastAPI Router exposing:
  - GET /api/v1/user/history (Paginated user scan history sorted by timestamp descending with S3 pre-signed URLs)
  - GET /api/v1/user/history/{module_id} (Full ModuleResult detail for reloading into the optical inspector canvas)
  - DELETE /api/v1/user/history/{module_id} (Delete an inspection from user history)
  - GET /api/v1/user/history/{module_id}/image (Raw image fallback stream for local/offline environments)
"""

import asyncio
import base64
from datetime import datetime, timezone
import json
import logging
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from pydantic import BaseModel, Field

try:
    from node1_gateway.auth import UserRecord, get_current_active_user
except ImportError:
    from auth import UserRecord, get_current_active_user

logger = logging.getLogger("node1_gateway.history")

DEFAULT_S3_BUCKET = os.getenv(
    "AWS_S3_BUCKET",
    "dataset-paineis-solares-joaoguilherme-joaovictor-gustavo-2026"
)
DEFAULT_AWS_REGION = os.getenv("AWS_DEFAULT_REGION", os.getenv("AWS_REGION", "us-east-2"))
PRESIGNED_URL_EXPIRES_IN = int(os.getenv("S3_PRESIGNED_URL_EXPIRES_IN", "3600"))

_SAFE_FILENAME_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def sanitize_filename(filename: Optional[str]) -> str:
    """Normalizes a filename for safe S3 object key construction."""
    if not filename:
        return "module.png"
    base = os.path.basename(filename.strip().replace("\\", "/"))
    cleaned = _SAFE_FILENAME_RE.sub("_", base).strip("._")
    return cleaned or "module.png"


def build_user_partition_keys(
    user_id: Optional[str],
    scan_id: str,
    filename: str,
    module_id: str
) -> Tuple[str, str]:
    """
    Constructs user-partitioned S3 object keys for an uploaded scan and its JSON analysis report.
    - Authenticated user:
        Image:  users/{user_id}/scans/{scan_id}_{filename}
        Report: users/{user_id}/reports/{scan_id}_{module_id}.json
    - Anonymous fallback:
        Image:  anonymous/scans/{scan_id}_{filename}
        Report: anonymous/reports/{scan_id}_{module_id}.json
    """
    clean_uid = (user_id or "").strip()
    prefix = f"users/{clean_uid}" if clean_uid and clean_uid != "anonymous" else "anonymous"
    safe_name = sanitize_filename(filename)
    safe_mod_id = _SAFE_FILENAME_RE.sub("_", str(module_id).strip()) or "MOD"
    image_key = f"{prefix}/scans/{scan_id}_{safe_name}"
    report_key = f"{prefix}/reports/{scan_id}_{safe_mod_id}.json"
    return image_key, report_key


# -----------------------------------------------------------------------------
# Pydantic DTOs for History Endpoints
# -----------------------------------------------------------------------------
class HistorySummaryItem(BaseModel):
    module_id: str
    scan_id: str
    user_id: str
    filename: str
    status: str
    rows: int = 6
    cols: int = 10
    total_cells: int = 60
    defective_cells_count: int = 0
    module_damage_percentage: float = 0.0
    processed_by_node: str = "worker-1"
    timestamp: str
    s3_image_key: str
    s3_report_key: str
    image_url: str = Field(..., description="S3 pre-signed URL for rendering the module scan.")
    presigned_url: str = Field(..., description="Alias for S3 pre-signed URL.")
    fallback_image_url: Optional[str] = Field(default=None, description="Local fallback image URL when S3 is unreachable.")


class UserHistoryListResponse(BaseModel):
    user_id: str
    total: int
    limit: int
    offset: int
    items: List[Dict[str, Any]]
    history: List[Dict[str, Any]]


class SyncHistoryRequest(BaseModel):
    modules: List[Dict[str, Any]] = Field(default_factory=list, description="Locally processed module inspections to sync to S3")


# -----------------------------------------------------------------------------
# In-Memory & S3 Hybrid History Repository (Ports & Adapters)
# -----------------------------------------------------------------------------
class InMemoryHistoryRepository:
    """
    Thread/Async-safe in-memory store for user scan reports and raw images.
    Guarantees Read-Your-Writes consistency and zero-crash local/test execution
    when AWS S3 is offline or mocked.
    """

    def __init__(self) -> None:
        # Maps user_id -> { module_id -> full_report_dict }
        self._reports_by_user: Dict[str, Dict[str, Dict[str, Any]]] = {}
        # Maps (user_id, module_id) -> (image_bytes, content_type)
        self._images_by_user_mod: Dict[Tuple[str, str], Tuple[bytes, str]] = {}
        self._lock = asyncio.Lock()

    async def save_record(
        self,
        user_id: str,
        module_id: str,
        report: Dict[str, Any],
        image_bytes: Optional[bytes] = None,
        content_type: str = "image/png"
    ) -> Dict[str, Any]:
        uid = user_id or "anonymous"
        record_copy = json.loads(json.dumps(report))
        async with self._lock:
            user_bucket = self._reports_by_user.setdefault(uid, {})
            user_bucket[module_id] = record_copy
            if image_bytes is not None:
                self._images_by_user_mod[(uid, module_id)] = (image_bytes, content_type or "image/png")
        return record_copy

    async def list_records(self, user_id: str) -> List[Dict[str, Any]]:
        uid = user_id or "anonymous"
        async with self._lock:
            user_bucket = self._reports_by_user.get(uid, {})
            items = [json.loads(json.dumps(v)) for v in user_bucket.values()]
        items.sort(key=lambda x: str(x.get("timestamp", "")), reverse=True)
        return items

    async def get_record(self, user_id: str, module_id: str) -> Optional[Dict[str, Any]]:
        uid = user_id or "anonymous"
        async with self._lock:
            user_bucket = self._reports_by_user.get(uid, {})
            rec = user_bucket.get(module_id)
            if rec is not None:
                return json.loads(json.dumps(rec))
            # Also allow lookup by scan_id if module_id passed matches scan_id
            for candidate in user_bucket.values():
                if candidate.get("scan_id") == module_id:
                    return json.loads(json.dumps(candidate))
        return None

    async def delete_record(self, user_id: str, module_id: str) -> Optional[Dict[str, Any]]:
        uid = user_id or "anonymous"
        async with self._lock:
            user_bucket = self._reports_by_user.get(uid, {})
            removed = user_bucket.pop(module_id, None)
            if removed is None:
                for k, v in list(user_bucket.items()):
                    if v.get("scan_id") == module_id:
                        removed = user_bucket.pop(k, None)
                        module_id = k
                        break
            self._images_by_user_mod.pop((uid, module_id), None)
            return removed

    async def get_image(self, user_id: str, module_id: str) -> Optional[Tuple[bytes, str]]:
        uid = user_id or "anonymous"
        async with self._lock:
            direct = self._images_by_user_mod.get((uid, module_id))
            if direct is not None:
                return direct
            user_bucket = self._reports_by_user.get(uid, {})
            for k, v in user_bucket.items():
                if v.get("scan_id") == module_id:
                    return self._images_by_user_mod.get((uid, k))
        return None

    async def clear(self) -> None:
        async with self._lock:
            self._reports_by_user.clear()
            self._images_by_user_mod.clear()


class UserHistoryService:
    """
    Coordinates user-partitioned S3 persistence (put_object, get_object, list_objects_v2,
    delete_object, generate_presigned_url) with automatic fallback to InMemoryHistoryRepository.
    """

    def __init__(
        self,
        memory_repo: Optional[InMemoryHistoryRepository] = None,
        s3_client_provider: Optional[Callable[[], Any]] = None,
        sqs_client_provider: Optional[Callable[[], Any]] = None,
        bucket_name: str = DEFAULT_S3_BUCKET,
        aws_region: str = DEFAULT_AWS_REGION,
        queue_url: Optional[str] = None
    ) -> None:
        self.memory_repo = memory_repo or InMemoryHistoryRepository()
        self._s3_client_provider = s3_client_provider
        self._sqs_client_provider = sqs_client_provider
        self.bucket_name = bucket_name
        self.aws_region = aws_region
        self.queue_url = queue_url

    def get_s3_client(self) -> Any:
        if self._s3_client_provider is not None:
            return self._s3_client_provider()
        return None

    def get_sqs_client(self) -> Any:
        if self._sqs_client_provider is not None:
            return self._sqs_client_provider()
        return None

    def generate_presigned_image_url(
        self,
        s3_image_key: str,
        expires_in: int = PRESIGNED_URL_EXPIRES_IN
    ) -> str:
        """
        Generates a secure AWS S3 pre-signed URL for GET access to the original module image.
        Falls back to a canonical signed URL format if S3 client is unavailable or unconfigured in tests.
        """
        s3 = self.get_s3_client()
        if s3 is not None:
            try:
                url = s3.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": self.bucket_name, "Key": s3_image_key},
                    ExpiresIn=expires_in
                )
                if isinstance(url, str) and url.strip():
                    return url
            except Exception as exc:
                logger.debug(f"S3 generate_presigned_url fallback for '{s3_image_key}': {exc}")

        return (
            f"https://{self.bucket_name}.s3.{self.aws_region}.amazonaws.com/{s3_image_key}"
            f"?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Expires={expires_in}&X-Amz-SignedHeaders=host"
        )

    async def persist_module_inspection(
        self,
        module_data: Dict[str, Any],
        image_bytes: bytes,
        filename: str,
        content_type: str,
        user: Optional[UserRecord] = None,
        scan_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Persists both the raw uploaded module scan image and the complete ModuleResult JSON
        under the user-partitioned S3 keys ONLY when the user is authenticated:
          - users/{user_id}/scans/{scan_id}_{filename}
          - users/{user_id}/reports/{scan_id}_{module_id}.json
        When unauthenticated (user is None), does NOT upload to S3 or SQS; keeps record
        as local-only so it can be persisted locally on the client and synced after login.
        """
        is_authenticated = (
            user is not None
            and bool(getattr(user, "user_id", None))
            and user.user_id != "anonymous"
        )
        user_id = user.user_id if is_authenticated else "anonymous"
        scan_uid = scan_id or str(module_data.get("scan_id") or "").strip() or uuid.uuid4().hex[:12]
        base_module_id = str(module_data.get("module_id") or f"MOD-2026-{scan_uid[:6].upper()}")
        now_iso = str(module_data.get("timestamp") or datetime.now(timezone.utc).isoformat())

        # Prevent silent overwrite if another scan with the same module_id already exists for this user
        existing = await self.memory_repo.get_record(user_id, base_module_id)
        if existing is not None and existing.get("scan_id") and existing.get("scan_id") != scan_uid:
            module_id = f"{base_module_id}-{scan_uid[:4].upper()}"
        else:
            module_id = base_module_id

        safe_filename = filename or str(module_data.get("filename") or "module.png")

        if not is_authenticated:
            # Unauthenticated mode: DO NOT save to S3 or SQS. Keep local-only metadata.
            module_data["module_id"] = module_id
            module_data["scan_id"] = scan_uid
            module_data["user_id"] = "anonymous"
            module_data["filename"] = safe_filename
            module_data["timestamp"] = now_iso
            module_data["s3_image_key"] = None
            module_data["s3_report_key"] = None
            module_data["persisted_in_s3"] = False
            module_data["storage_scope"] = "local"

            # Cache in ephemeral memory so if the user logs in during this session, raw bytes can be synced
            await self.memory_repo.save_record(
                user_id="anonymous",
                module_id=module_id,
                report=module_data,
                image_bytes=image_bytes,
                content_type=content_type or "image/png"
            )
            return module_data

        image_key, report_key = build_user_partition_keys(
            user_id=user_id,
            scan_id=scan_uid,
            filename=safe_filename,
            module_id=module_id
        )

        presigned = self.generate_presigned_image_url(image_key)

        # Enrich module payload with canonical metadata & S3 keys
        module_data["module_id"] = module_id
        module_data["scan_id"] = scan_uid
        module_data["user_id"] = user_id
        module_data["filename"] = safe_filename
        module_data["timestamp"] = now_iso
        module_data["s3_image_key"] = image_key
        module_data["s3_report_key"] = report_key
        module_data["persisted_in_s3"] = True
        module_data["storage_scope"] = "s3"
        module_data["image_url"] = presigned
        module_data["presigned_url"] = presigned
        module_data["fallback_image_url"] = f"/api/v1/user/history/{module_id}/image"

        # Always store in local fallback repository for Read-Your-Writes and offline resilience
        await self.memory_repo.save_record(
            user_id=user_id,
            module_id=module_id,
            report=module_data,
            image_bytes=image_bytes,
            content_type=content_type or "image/png"
        )

        # Persist to Amazon S3 & notify SQS if client is available
        s3 = self.get_s3_client()
        sqs = self.get_sqs_client()

        if s3 is not None:
            try:
                report_bytes = json.dumps(module_data, indent=2).encode("utf-8")
                if image_bytes:
                    await asyncio.to_thread(
                        s3.put_object,
                        Bucket=self.bucket_name,
                        Key=image_key,
                        Body=image_bytes,
                        ContentType=content_type or "image/png"
                    )
                await asyncio.to_thread(
                    s3.put_object,
                    Bucket=self.bucket_name,
                    Key=report_key,
                    Body=report_bytes,
                    ContentType="application/json"
                )
            except Exception as aws_err:
                logger.warning(
                    f"AWS S3 persistence fallback for module '{module_id}' (key='{image_key}'): {aws_err}. "
                    f"Inspection preserved in local fallback repository."
                )

        if sqs is not None and self.queue_url:
            try:
                await asyncio.to_thread(
                    sqs.send_message,
                    QueueUrl=self.queue_url,
                    MessageBody=image_key
                )
            except Exception as sqs_err:
                logger.warning(
                    f"AWS SQS notification fallback for '{image_key}': {sqs_err}."
                )

        return module_data

    async def sync_local_modules_to_user(
        self,
        user: UserRecord,
        modules: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Synchronizes locally stored unauthenticated module inspections (including their
        already-processed JSON and image data) to the authenticated user's S3 partition
        without re-running computer vision inference.
        """
        existing_items, _ = await self.list_user_history(user_id=user.user_id, limit=500, offset=0)
        existing_scan_ids = {item.get("scan_id") for item in existing_items if item.get("scan_id")}

        async def _sync_single_module(mod: Dict[str, Any], scan_uid: str) -> None:
            mod_id = str(mod.get("module_id") or f"MOD-2026-{scan_uid[:6].upper()}")
            filename = str(mod.get("filename") or f"{mod_id}.png")

            # 1. Check if raw image bytes are cached in anonymous memory repository
            cached_img = (
                await self.memory_repo.get_image("anonymous", mod_id)
                or await self.memory_repo.get_image("anonymous", scan_uid)
            )
            img_bytes = b""
            ctype = "image/png"
            if cached_img is not None:
                img_bytes, ctype = cached_img
            else:
                # 2. Fallback: decode base64 rectified_image from client payload
                rect_b64 = str(mod.get("rectified_image") or "")
                if rect_b64:
                    try:
                        raw_b64 = rect_b64.split(",", 1)[1] if "," in rect_b64 else rect_b64
                        img_bytes = base64.b64decode(raw_b64)
                    except Exception:
                        img_bytes = b""

            await self.persist_module_inspection(
                module_data=dict(mod),
                image_bytes=img_bytes,
                filename=filename,
                content_type=ctype,
                user=user,
                scan_id=scan_uid
            )
            await self.memory_repo.delete_record("anonymous", mod_id)

        sync_tasks = []
        for mod in modules or []:
            if not isinstance(mod, dict):
                continue
            scan_uid = str(mod.get("scan_id") or "").strip() or uuid.uuid4().hex[:12]
            if scan_uid in existing_scan_ids:
                continue
            existing_scan_ids.add(scan_uid)
            sync_tasks.append(_sync_single_module(mod, scan_uid))

        if sync_tasks:
            await asyncio.gather(*sync_tasks)

        return await self.list_user_history(user_id=user.user_id, limit=100, offset=0)

    def _enrich_with_urls(self, record: Dict[str, Any]) -> Dict[str, Any]:
        enriched = dict(record)
        s3_image_key = enriched.get("s3_image_key") or ""
        module_id = enriched.get("module_id") or ""
        presigned = self.generate_presigned_image_url(s3_image_key) if s3_image_key else ""
        enriched["image_url"] = presigned
        enriched["presigned_url"] = presigned
        enriched["fallback_image_url"] = f"/api/v1/user/history/{module_id}/image" if module_id else None
        return enriched

    async def list_user_history(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Lists past module inspections for a specific user_id, sorted by timestamp descending.
        Combines any records discovered in S3 under users/{user_id}/reports/ with the
        Read-Your-Writes local repository and generates fresh pre-signed S3 URLs.
        """
        records_by_key: Dict[str, Dict[str, Any]] = {}
        seen_scan_ids: set = set()

        # 1. Load from local Read-Your-Writes store
        local_records = await self.memory_repo.list_records(user_id)
        for rec in local_records:
            mid = rec.get("module_id")
            sid = rec.get("scan_id") or mid
            if mid:
                records_by_key[mid] = rec
                if sid:
                    seen_scan_ids.add(sid)

        # 2. Attempt to list from S3 under users/{user_id}/reports/ if S3 client is configured
        s3 = self.get_s3_client()
        if s3 is not None:
            prefix = f"users/{user_id}/reports/"
            try:
                list_resp = await asyncio.to_thread(
                    s3.list_objects_v2,
                    Bucket=self.bucket_name,
                    Prefix=prefix
                )
                if isinstance(list_resp, dict) and isinstance(list_resp.get("Contents"), list):
                    report_keys = [
                        obj["Key"] for obj in list_resp["Contents"]
                        if isinstance(obj.get("Key"), str) and obj["Key"].endswith(".json")
                    ]
                    # Sort keys to prioritize most recent files before fetching
                    report_keys.sort(reverse=True)
                    target_keys = report_keys[:limit] if limit > 0 else report_keys

                    async def fetch_single_s3_report(key: str) -> Optional[Dict[str, Any]]:
                        try:
                            get_resp = await asyncio.to_thread(
                                s3.get_object,
                                Bucket=self.bucket_name,
                                Key=key
                            )
                            body_stream = get_resp.get("Body") if isinstance(get_resp, dict) else None
                            if body_stream and hasattr(body_stream, "read"):
                                raw_bytes = body_stream.read()
                                if isinstance(raw_bytes, (bytes, bytearray)):
                                    s3_rec = json.loads(raw_bytes.decode("utf-8"))
                                elif isinstance(raw_bytes, str):
                                    s3_rec = json.loads(raw_bytes)
                                else:
                                    return None
                                if isinstance(s3_rec, dict) and s3_rec.get("user_id", user_id) == user_id:
                                    return s3_rec
                        except Exception as item_exc:
                            logger.debug(f"Skipping unreadable S3 report '{key}': {item_exc}")
                        return None

                    fetched_reports = await asyncio.gather(*[fetch_single_s3_report(k) for k in target_keys])
                    for s3_rec in fetched_reports:
                        if s3_rec:
                            mid = str(s3_rec.get("module_id") or "")
                            sid = str(s3_rec.get("scan_id") or "")
                            if not mid:
                                continue
                            if sid and sid in seen_scan_ids:
                                continue
                            if mid in records_by_key and sid:
                                existing_sid = records_by_key[mid].get("scan_id")
                                if existing_sid and existing_sid != sid:
                                    mid = f"{mid}-{sid[:4].upper()}"
                                    s3_rec["module_id"] = mid
                            if mid not in records_by_key:
                                records_by_key[mid] = s3_rec
                                if sid:
                                    seen_scan_ids.add(sid)
                                await self.memory_repo.save_record(
                                    user_id=user_id,
                                    module_id=mid,
                                    report=s3_rec
                                )
            except Exception as list_exc:
                logger.debug(f"S3 list_objects_v2 fallback for user '{user_id}': {list_exc}")

        all_records = list(records_by_key.values())
        all_records.sort(key=lambda r: str(r.get("timestamp", "")), reverse=True)
        total = len(all_records)

        sliced = all_records[offset: offset + limit] if limit > 0 else all_records[offset:]
        enriched_items = [self._enrich_with_urls(item) for item in sliced]
        return enriched_items, total

    async def get_user_module_detail(self, user_id: str, module_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetches the full ModuleResult inspection payload (including cell bounding boxes and defect classes)
        for the authenticated user.
        """
        rec = await self.memory_repo.get_record(user_id, module_id)
        if rec is not None:
            return self._enrich_with_urls(rec)

        # Fallback: scan S3 reports for this user if not cached in memory
        items, _ = await self.list_user_history(user_id=user_id, limit=500, offset=0)
        for item in items:
            if item.get("module_id") == module_id or item.get("scan_id") == module_id:
                return item
        return None

    async def delete_user_module(self, user_id: str, module_id: str) -> Optional[Dict[str, Any]]:
        """
        Deletes a module inspection record from both the local repository and S3.
        """
        removed = await self.memory_repo.delete_record(user_id, module_id)
        if removed is None:
            detail = await self.get_user_module_detail(user_id, module_id)
            if detail is not None:
                removed = detail
                await self.memory_repo.delete_record(user_id, detail.get("module_id", module_id))
        s3 = self.get_s3_client()
        if removed and s3 is not None:
            for key_field in ("s3_report_key", "s3_image_key"):
                s3_key = removed.get(key_field)
                if s3_key:
                    try:
                        await asyncio.to_thread(
                            s3.delete_object,
                            Bucket=self.bucket_name,
                            Key=s3_key
                        )
                    except Exception as del_exc:
                        logger.debug(f"S3 delete_object fallback for '{s3_key}': {del_exc}")
        return removed


# Singleton History Repository & Service (wired to main gateway S3/SQS clients)
history_memory_repository = InMemoryHistoryRepository()
history_service = UserHistoryService(
    memory_repo=history_memory_repository,
    bucket_name=DEFAULT_S3_BUCKET,
    aws_region=DEFAULT_AWS_REGION
)


def get_history_service() -> UserHistoryService:
    return history_service


# -----------------------------------------------------------------------------
# FastAPI Router: /api/v1/user/history
# -----------------------------------------------------------------------------
history_router = APIRouter(tags=["User History & Gallery"])


@history_router.get(
    "/api/v1/user/history",
    response_model=UserHistoryListResponse,
    status_code=status.HTTP_200_OK,
    summary="List authenticated operator's past module inspections with S3 pre-signed image URLs"
)
async def get_user_history(
    limit: int = Query(default=50, ge=1, le=500, description="Max number of history items to return"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    current_user: UserRecord = Depends(get_current_active_user),
    service: UserHistoryService = Depends(get_history_service)
):
    items, total = await service.list_user_history(
        user_id=current_user.user_id,
        limit=limit,
        offset=offset
    )
    return UserHistoryListResponse(
        user_id=current_user.user_id,
        total=total,
        limit=limit,
        offset=offset,
        items=items,
        history=items
    )


@history_router.post(
    "/api/v1/user/history/sync",
    response_model=UserHistoryListResponse,
    status_code=status.HTTP_200_OK,
    summary="Sync locally processed unauthenticated inspections to the authenticated user's S3 bucket"
)
async def sync_user_history(
    payload: SyncHistoryRequest,
    current_user: UserRecord = Depends(get_current_active_user),
    service: UserHistoryService = Depends(get_history_service)
):
    items, total = await service.sync_local_modules_to_user(
        user=current_user,
        modules=payload.modules
    )
    return UserHistoryListResponse(
        user_id=current_user.user_id,
        total=total,
        limit=100,
        offset=0,
        items=items,
        history=items
    )


@history_router.get(
    "/api/v1/user/history/{module_id}",
    status_code=status.HTTP_200_OK,
    summary="Fetch full detailed inspection report for a historical module scan"
)
async def get_user_history_detail(
    module_id: str,
    current_user: UserRecord = Depends(get_current_active_user),
    service: UserHistoryService = Depends(get_history_service)
):
    detail = await service.get_user_module_detail(
        user_id=current_user.user_id,
        module_id=module_id
    )
    if detail is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inspeção '{module_id}' não encontrada no histórico do operador."
        )
    return detail


@history_router.delete(
    "/api/v1/user/history/{module_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete a module inspection from the authenticated operator's history"
)
async def delete_user_history_item(
    module_id: str,
    current_user: UserRecord = Depends(get_current_active_user),
    service: UserHistoryService = Depends(get_history_service)
):
    removed = await service.delete_user_module(
        user_id=current_user.user_id,
        module_id=module_id
    )
    if removed is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inspeção '{module_id}' não encontrada no histórico do operador."
        )
    return {
        "status": "deleted",
        "module_id": removed.get("module_id", module_id),
        "user_id": current_user.user_id,
        "s3_image_key": removed.get("s3_image_key"),
        "s3_report_key": removed.get("s3_report_key")
    }


@history_router.get(
    "/api/v1/user/history/{module_id}/image",
    status_code=status.HTTP_200_OK,
    summary="Serve raw module scan image from local fallback or S3"
)
async def get_user_history_image(
    module_id: str,
    current_user: UserRecord = Depends(get_current_active_user),
    service: UserHistoryService = Depends(get_history_service)
):
    cached = await service.memory_repo.get_image(current_user.user_id, module_id)
    if cached is not None:
        img_bytes, ctype = cached
        return Response(content=img_bytes, media_type=ctype or "image/png")

    detail = await service.get_user_module_detail(current_user.user_id, module_id)
    if detail and detail.get("s3_image_key"):
        s3 = service.get_s3_client()
        if s3 is not None:
            try:
                resp = await asyncio.to_thread(
                    s3.get_object,
                    Bucket=service.bucket_name,
                    Key=detail["s3_image_key"]
                )
                body = resp.get("Body") if isinstance(resp, dict) else None
                if body and hasattr(body, "read"):
                    data = body.read()
                    if isinstance(data, (bytes, bytearray)):
                        return Response(content=bytes(data), media_type="image/png")
            except Exception:
                pass

    if detail and detail.get("rectified_image"):
        try:
            rect_b64 = str(detail["rectified_image"])
            raw_b64 = rect_b64.split(",", 1)[1] if "," in rect_b64 else rect_b64
            img_bytes = base64.b64decode(raw_b64)
            return Response(content=img_bytes, media_type="image/png")
        except Exception:
            pass

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Imagem da inspeção '{module_id}' não disponível."
    )

