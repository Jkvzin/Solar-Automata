"""
Node 1: API Gateway & Orchestrator Service (FastAPI)
===================================================
Author: Principal Distributed Systems Engineer & Senior Computer Vision Architect
Role:
- Serves the LuminaPV operator diagnostic console ('luminapv-el-console.html') at GET /
- Ingests bulk module image uploads via POST /api/v1/analyze/bulk
- Dispatches parallel HTTP multipart requests to Node 2 (Worker) using httpx.AsyncClient + asyncio.gather
- Aggregates individual module telemetry into the consolidated batch summary
- Proxies/serves static fixtures for sample loading
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import socket
from typing import Any, Dict, List, Optional, Annotated
import uuid

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import httpx
from pydantic import BaseModel, Field
import redis.asyncio as aioredis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [node1-gateway] %(message)s"
)
logger = logging.getLogger("node1_gateway")

WORKER_SERVICE_URL = os.getenv("WORKER_SERVICE_URL", "http://localhost:8001").rstrip("/")
REDIS_URL = os.getenv("REDIS_URL", "redis://state.luminapv.local:6379/0")
GATEWAY_TIMEOUT_SECONDS = float(os.getenv("GATEWAY_TIMEOUT_SECONDS", "60.0"))
DEFAULT_LOCK_TTL_SECONDS = int(os.getenv("DEFAULT_LOCK_TTL_SECONDS", "30"))

# AWS Ingestion Configuration (resilient and protected)
AWS_INGESTION_ENABLED = os.getenv("AWS_INGESTION_ENABLED", "false").lower() in ("true", "1", "yes")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", os.getenv("AWS_REGION", "us-east-2"))
BUCKET_NAME = os.getenv("AWS_S3_BUCKET", "dataset-paineis-solares-joaoguilherme-joaovictor-gustavo-2026")
QUEUE_URL = os.getenv("AWS_SQS_QUEUE_URL", "https://sqs.us-east-2.amazonaws.com/058835009582/fila-imagens-paineis-solares")

try:
    import boto3
    s3_client = boto3.client('s3', region_name=AWS_REGION) if AWS_INGESTION_ENABLED else None
    sqs_client = boto3.client('sqs', region_name=AWS_REGION) if AWS_INGESTION_ENABLED else None
except Exception as _boto_err:
    logger.debug(f"AWS Boto3 initialization bypassed: {_boto_err}")
    s3_client = None
    sqs_client = None

# -----------------------------------------------------------------------------
# Distributed Mutual Exclusion (DistributedLock) - Issue #3
# -----------------------------------------------------------------------------
ACTIVE_LOCKS_KEY = "luminapv:active_locks"
_LOCAL_ACTIVE_LOCKS: Dict[str, Dict[str, Any]] = {}
_LOCAL_LOCK_SYNC = asyncio.Lock()

_LUA_RELEASE_LOCK = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    redis.call("del", KEYS[1])
    redis.call("hdel", KEYS[2], KEYS[1])
    return 1
else
    return 0
end
"""


class DistributedLock:
    """
    Gerenciador de lock distribuído (Issue #3) com garantia de aquisição atômica no Redis
    (SET resource token NX EX ttl), timeout contra deadlocks, liberação garantida via Lua
    e fallback thread/async-safe em memória para testes unitários ou execução standalone.
    """

    def __init__(
        self,
        resource: str,
        ttl_seconds: int = DEFAULT_LOCK_TTL_SECONDS,
        redis_client: Optional[aioredis.Redis] = None,
        owner: Optional[str] = None
    ) -> None:
        self.resource = resource
        self.ttl_seconds = ttl_seconds
        self.redis_client = redis_client
        self.owner = owner or f"gateway-{socket.gethostname()}"
        self.token = str(uuid.uuid4())
        self.acquired = False
        self.acquired_at: Optional[str] = None

    async def acquire(self, timeout: float = 5.0, retry_interval: float = 0.05) -> bool:
        start_time = asyncio.get_event_loop().time()
        now_iso = datetime.now(timezone.utc).isoformat()
        lock_info = {
            "resource": self.resource,
            "token": self.token,
            "owner": self.owner,
            "acquired_at": now_iso,
            "ttl_seconds": self.ttl_seconds
        }

        while True:
            if self.redis_client is not None:
                try:
                    res = await self.redis_client.set(
                        self.resource,
                        self.token,
                        nx=True,
                        ex=self.ttl_seconds
                    )
                    if res:
                        self.acquired = True
                        self.acquired_at = now_iso
                        await self.redis_client.hset(
                            ACTIVE_LOCKS_KEY,
                            self.resource,
                            json.dumps(lock_info)
                        )
                        return True
                except Exception as exc:
                    logger.debug(f"Redis indisponível para lock '{self.resource}' ({exc}); usando fallback local.")
                    break
            else:
                break

            if (asyncio.get_event_loop().time() - start_time) >= timeout:
                return False
            await asyncio.sleep(retry_interval)

        # Fallback local em memória
        async with _LOCAL_LOCK_SYNC:
            now_ts = asyncio.get_event_loop().time()
            entry = _LOCAL_ACTIVE_LOCKS.get(self.resource)
            if entry and entry.get("expires_ts", 0) > now_ts:
                return False
            _LOCAL_ACTIVE_LOCKS[self.resource] = {
                **lock_info,
                "expires_ts": now_ts + self.ttl_seconds
            }
            self.acquired = True
            self.acquired_at = now_iso
            return True

    async def release(self) -> bool:
        if not self.acquired:
            return False

        released = False
        if self.redis_client is not None:
            try:
                res = await self.redis_client.eval(
                    _LUA_RELEASE_LOCK,
                    2,
                    self.resource,
                    ACTIVE_LOCKS_KEY,
                    self.token
                )
                released = (res == 1)
            except Exception as exc:
                logger.debug(f"Redis unlock failed for '{self.resource}': {exc}")

        async with _LOCAL_LOCK_SYNC:
            local_entry = _LOCAL_ACTIVE_LOCKS.get(self.resource)
            if local_entry and local_entry.get("token") == self.token:
                del _LOCAL_ACTIVE_LOCKS[self.resource]
                released = True

        self.acquired = False
        return released

    async def __aenter__(self):
        ok = await self.acquire()
        if not ok:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Resource '{self.resource}' is currently locked by another worker/process."
            )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.release()


class DistributedLockManager:
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.redis_client: Optional[aioredis.Redis] = None

    async def get_client(self) -> Optional[aioredis.Redis]:
        if self.redis_client is None:
            try:
                self.redis_client = aioredis.from_url(
                    self.redis_url,
                    decode_responses=True,
                    socket_connect_timeout=1.0,
                    socket_timeout=1.0
                )
            except Exception:
                self.redis_client = None
        return self.redis_client

    def create_lock(self, resource: str, ttl_seconds: int = DEFAULT_LOCK_TTL_SECONDS) -> DistributedLock:
        return DistributedLock(
            resource=resource,
            ttl_seconds=ttl_seconds,
            redis_client=self.redis_client
        )

    async def get_active_locks(self) -> List[Dict[str, Any]]:
        client = await self.get_client()
        active: List[Dict[str, Any]] = []
        if client is not None:
            try:
                raw_hash = await client.hgetall(ACTIVE_LOCKS_KEY)
                for res_key, val in raw_hash.items():
                    try:
                        ttl_rem = await client.ttl(res_key)
                        if ttl_rem > 0:
                            item = json.loads(val)
                            item["ttl_remaining_seconds"] = ttl_rem
                            active.append(item)
                        else:
                            await client.hdel(ACTIVE_LOCKS_KEY, res_key)
                    except Exception:
                        pass
                if active:
                    return active
            except Exception:
                pass

        now_ts = asyncio.get_event_loop().time()
        async with _LOCAL_LOCK_SYNC:
            for k, v in list(_LOCAL_ACTIVE_LOCKS.items()):
                rem = v.get("expires_ts", 0) - now_ts
                if rem > 0:
                    active.append({
                        "resource": v["resource"],
                        "token": v["token"],
                        "owner": v["owner"],
                        "acquired_at": v["acquired_at"],
                        "ttl_remaining_seconds": round(rem, 1)
                    })
                else:
                    del _LOCAL_ACTIVE_LOCKS[k]
        return active


lock_manager = DistributedLockManager(REDIS_URL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await lock_manager.get_client()
    yield
    if lock_manager.redis_client is not None:
        try:
            await lock_manager.redis_client.close()
        except Exception:
            pass


app = FastAPI(
    title="LuminaPV API Gateway & Orchestrator (Node 1)",
    description="Distributed gateway orchestrator dispatching parallel CV tasks to Node 2 worker.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Determine paths for frontend console and fixtures
BASE_DIR = Path(__file__).resolve().parent

POSSIBLE_FRONTEND_DIRS = [
    BASE_DIR / "frontend",
    BASE_DIR.parent / "frontend",
    Path("/app/frontend")
]

POSSIBLE_CONSOLE_PATHS = [
    BASE_DIR / "luminapv-el-console.html",
    BASE_DIR.parent / "luminapv-el-console.html",
    Path("/app/luminapv-el-console.html")
]

POSSIBLE_FIXTURE_DIRS = [
    BASE_DIR / "fixtures",
    BASE_DIR.parent / "fixtures",
    Path("/app/fixtures")
]

frontend_dir: Optional[Path] = None
for fd in POSSIBLE_FRONTEND_DIRS:
    if (fd / "index.html").is_file():
        frontend_dir = fd
        break

console_file_path: Optional[Path] = None
for p in POSSIBLE_CONSOLE_PATHS:
    if p.is_file():
        console_file_path = p
        break

fixtures_dir_path: Optional[Path] = None
for f in POSSIBLE_FIXTURE_DIRS:
    if f.is_dir():
        fixtures_dir_path = f
        break

# Mount static frontend assets if available
if frontend_dir:
    css_dir = frontend_dir / "css"
    js_dir = frontend_dir / "js"
    if css_dir.is_dir():
        app.mount("/css", StaticFiles(directory=str(css_dir)), name="css")
    if js_dir.is_dir():
        app.mount("/js", StaticFiles(directory=str(js_dir)), name="js")


# -----------------------------------------------------------------------------
# Pydantic Contract Models
# -----------------------------------------------------------------------------
class BatchSummary(BaseModel):
    total_modules: int
    defective_modules_count: int
    batch_failure_percentage: float
    average_cell_degradation_percentage: float


class BulkAnalysisResponse(BaseModel):
    batch_summary: BatchSummary
    modules: List[Dict[str, Any]]


# -----------------------------------------------------------------------------
# Web & Static Asset Endpoints
# -----------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse, tags=["Console Frontend"])
async def serve_console():
    """
    Serves the LuminaPV Optical Diagnostic Console interface.
    Prioritizes the modular frontend (frontend/index.html), falling back to legacy single-file HTML.
    """
    if frontend_dir and (frontend_dir / "index.html").is_file():
        return FileResponse(frontend_dir / "index.html", media_type="text/html")
    if console_file_path and console_file_path.is_file():
        return FileResponse(console_file_path, media_type="text/html")
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="LuminaPV console template not found on server."
    )


@app.get("/legacy", response_class=HTMLResponse, tags=["Console Frontend"])
async def serve_legacy_console():
    """
    Serves the legacy monolithic console ('luminapv-el-console.html') for backwards compatibility.
    """
    if console_file_path and console_file_path.is_file():
        return FileResponse(console_file_path, media_type="text/html")
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Legacy LuminaPV console 'luminapv-el-console.html' not found on server."
    )


@app.get("/fixtures/{filename:path}", tags=["Console Frontend"])
async def serve_fixture(filename: str):
    """
    Serves sample fixtures (e.g. fixtures/luminapv-batch-sample.json).
    """
    if fixtures_dir_path:
        file_path = (fixtures_dir_path / filename).resolve()
        if file_path.is_file() and file_path.is_relative_to(fixtures_dir_path.resolve()):
            return FileResponse(file_path, media_type="application/json")
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Fixture file '{filename}' not found."
    )


@app.get("/health", tags=["Telemetry"])
@app.get("/api/v1/health", tags=["Telemetry"])
async def health_check():
    """
    Gateway health probe and worker connectivity status.
    """
    worker_status = "unknown"
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{WORKER_SERVICE_URL}/health")
            worker_status = "connected" if resp.status_code == 200 else f"status_{resp.status_code}"
    except Exception as e:
        worker_status = f"unreachable ({type(e).__name__})"

    return {
        "status": "healthy",
        "service": "node1_gateway",
        "worker_service_url": WORKER_SERVICE_URL,
        "worker_connectivity": worker_status
    }


# -----------------------------------------------------------------------------
# Core Bulk Analysis Orchestrator
# -----------------------------------------------------------------------------
async def dispatch_module_to_worker(
    client: httpx.AsyncClient,
    file_bytes: bytes,
    filename: str,
    content_type: str
) -> Dict[str, Any]:
    """
    Transfers a single raw module image stream to Node 2 worker over internal REST.
    """
    url = f"{WORKER_SERVICE_URL}/api/v1/process-module"
    files = {
        "file": (filename, file_bytes, content_type or "image/png")
    }

    try:
        response = await client.post(url, files=files)
    except httpx.RequestError as exc:
        logger.error(f"Network failure calling Node 2 worker at {url} for '{filename}': {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Node 2 worker at {WORKER_SERVICE_URL} is unreachable: {str(exc)}"
        )

    if response.status_code == 422:
        # Worker rejected corrupted/unparseable image
        err_body = response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text
        logger.warning(f"Node 2 rejected module '{filename}' with HTTP 422: {err_body}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Node 2 unprocessable module for '{filename}': {err_body}"
        )

    if response.status_code != 200:
        logger.error(f"Node 2 returned error HTTP {response.status_code} for '{filename}': {response.text}")
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Worker error processing '{filename}': {response.text}"
        )

    return response.json()


@app.post(
    "/api/v1/analyze/bulk",
    response_model=BulkAnalysisResponse,
    status_code=status.HTTP_200_OK,
    tags=["Analysis"],
    summary="Bulk ingestion and parallel dispatch of photovoltaic module scans"
)
async def analyze_bulk(files: list[UploadFile]):
    """
    Primary ingestion endpoint for the LuminaPV Console.
    Receives N module images in a single multipart request, protects critical section
    using DistributedLock ('lock:module:<sha256>') per image (Issue #3),
    and safely integrates AWS S3/SQS with fallback for local runs.
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files received in multipart request key 'files'."
        )

    # 1. AWS S3/SQS Ingestion (protegido e resiliente contra falha HTTP 500)
    if AWS_INGESTION_ENABLED or (s3_client is not None and not isinstance(s3_client, type(None))):
        uploaded_keys = []
        for file in files:
            try:
                file_key = f"{uuid.uuid4()}-{file.filename}"
                conteudo_imagem = await file.read()
                arquivo_virtual = io.BytesIO(conteudo_imagem)

                if s3_client is not None:
                    s3_client.upload_fileobj(
                        arquivo_virtual,
                        BUCKET_NAME,
                        file_key
                    )
                    uploaded_keys.append(file_key)

                if sqs_client is not None:
                    sqs_client.send_message(
                        QueueUrl=QUEUE_URL,
                        MessageBody=file_key
                    )
            except Exception as aws_err:
                logger.warning(
                    f"AWS ingestion fallback: failed to upload '{file.filename}' to S3/SQS ({aws_err}). "
                    f"Continuing local/cluster processing."
                )
    else:
        logger.debug("AWS ingestion disabled via configuration; processing entirely in-cluster.")

    logger.info(f"Initiating parallel bulk analysis for {len(files)} module scans...")

    # Read uploaded files into memory buffers concurrently
    file_payloads = []
    for f in files:
        await f.seek(0)
        content = await f.read()
        file_payloads.append((content, f.filename or "module.png", f.content_type or "image/png"))

    # Dispatch to Node 2 worker concurrently using httpx.AsyncClient protected by DistributedLock
    limits = httpx.Limits(max_keepalive_connections=20, max_connections=50)
    timeout = httpx.Timeout(GATEWAY_TIMEOUT_SECONDS, connect=10.0)

    async def process_module_with_lock(client: httpx.AsyncClient, content: bytes, name: str, ctype: str):
        content_sha256 = hashlib.sha256(content).hexdigest()
        lock_resource = f"lock:module:{content_sha256}"
        lock = lock_manager.create_lock(lock_resource, ttl_seconds=DEFAULT_LOCK_TTL_SECONDS)

        # Exclusão Mútua Distribuída na seção crítica de processamento da imagem
        async with lock:
            return await dispatch_module_to_worker(client, content, name, ctype)

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        tasks = [
            process_module_with_lock(client, content, name, ctype)
            for content, name, ctype in file_payloads
        ]

        # Use asyncio.gather to execute all worker requests concurrently
        modules = await asyncio.gather(*tasks)

    # Ensure every module has a strictly unique module_id across the batch
    seen_ids = set()
    for idx, m in enumerate(modules, start=1):
        mid = m.get("module_id")
        if not mid or mid in seen_ids:
            mid = f"MOD-2026-{idx:03d}"
            m["module_id"] = mid
        seen_ids.add(mid)

    # Compute batch-wide aggregated telemetry
    total_modules = len(modules)
    defective_modules_count = sum(1 for m in modules if m.get("status") == "FAIL")

    batch_failure_percentage = (
        round((defective_modules_count / float(total_modules)) * 100.0, 1)
        if total_modules > 0 else 0.0
    )

    total_damage_sum = sum(float(m.get("module_damage_percentage", 0.0)) for m in modules)
    average_cell_degradation_percentage = (
        round(total_damage_sum / float(total_modules), 1)
        if total_modules > 0 else 0.0
    )

    batch_summary = BatchSummary(
        total_modules=total_modules,
        defective_modules_count=defective_modules_count,
        batch_failure_percentage=batch_failure_percentage,
        average_cell_degradation_percentage=average_cell_degradation_percentage
    )

    logger.info(
        f"Bulk analysis complete: {total_modules} modules processed, "
        f"{defective_modules_count} failed ({batch_failure_percentage}%), "
        f"avg cell degradation: {average_cell_degradation_percentage}%"
    )

    return BulkAnalysisResponse(
        batch_summary=batch_summary,
        modules=modules
    )


@app.post("/api/v1/cluster/election", tags=["Coordination"])
async def trigger_cluster_election(action: Optional[str] = "step_down"):
    """
    Permite simular reeleição de líder no cluster LuminaPV a partir do Gateway (Issue #3).
    Ações suportadas: 'step_down' ou 'reelect'.
    """
    client = await lock_manager.get_client()
    previous_leader = None
    if client is not None:
        try:
            raw_lead = await client.get("luminapv:leader")
            if raw_lead:
                try:
                    previous_leader = json.loads(raw_lead).get("leader_id")
                except Exception:
                    previous_leader = raw_lead
            await client.delete("luminapv:leader")
        except Exception as e:
            logger.warning(f"Erro ao remover líder no Redis via gateway: {e}")

    try:
        async with httpx.AsyncClient(timeout=2.0) as http_c:
            await http_c.post(f"{WORKER_SERVICE_URL}/api/v1/cluster/election?action={action}")
    except Exception:
        pass

    return {
        "status": "election_triggered",
        "action": action,
        "previous_leader": previous_leader,
        "message": "Leader lease relinquished, failover election triggered across active workers."
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
