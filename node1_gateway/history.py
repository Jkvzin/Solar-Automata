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
            return self._images_by_user_mod.get((uid, module_id))

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
        under the user-partitioned S3 keys:
          - users/{user_id}/scans/{scan_id}_{filename}
          - users/{user_id}/reports/{scan_id}_{module_id}.json
        (or anonymous/scans/... and anonymous/reports/... when unauthenticated).
        """
        user_id = user.user_id if user is not None else "anonymous"
        scan_uid = scan_id or uuid.uuid4().hex[:12]
        module_id = str(module_data.get("module_id") or f"MOD-2026-{scan_uid[:6].upper()}")
        now_iso = str(module_data.get("timestamp") or datetime.now(timezone.utc).isoformat())

        image_key, report_key = build_user_partition_keys(
            user_id=user_id,
            scan_id=scan_uid,
            filename=filename,
            module_id=module_id
        )

        # Enrich module payload with canonical metadata & S3 keys
        module_data["module_id"] = module_id
        module_data["scan_id"] = scan_uid
        module_data["user_id"] = user_id
        module_data["filename"] = filename or module_data.get("filename") or "module.png"
        module_data["timestamp"] = now_iso
        module_data["s3_image_key"] = image_key
        module_data["s3_report_key"] = report_key

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
        records_by_mod_id: Dict[str, Dict[str, Any]] = {}

        # 1. Load from local Read-Your-Writes store
        local_records = await self.memory_repo.list_records(user_id)
        for rec in local_records:
            mid = rec.get("module_id")
            if mid:
                records_by_mod_id[mid] = rec

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
                    for obj in list_resp["Contents"]:
                        key = obj.get("Key", "")
                        if not key.endswith(".json"):
                            continue
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
                                    continue
                                if isinstance(s3_rec, dict) and s3_rec.get("user_id", user_id) == user_id:
                                    mid = s3_rec.get("module_id")
                                    if mid and mid not in records_by_mod_id:
                                        records_by_mod_id[mid] = s3_rec
                        except Exception as item_exc:
                            logger.debug(f"Skipping unreadable S3 report '{key}': {item_exc}")
            except Exception as list_exc:
                logger.debug(f"S3 list_objects_v2 fallback for user '{user_id}': {list_exc}")

        all_records = list(records_by_mod_id.values())
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

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Imagem da inspeção '{module_id}' não disponível."
    )

