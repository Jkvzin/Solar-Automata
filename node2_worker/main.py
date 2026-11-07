"""
Node 2: Computer Vision Worker Service (FastAPI + OpenCV + SciPy + YOLOv8-cls)
==============================================================================
Exposes internal REST endpoint POST /api/v1/process-module for automated,
in-memory optical diagnostic inspection of photovoltaic electroluminescence scans,
and performs periodic service discovery self-registration (heartbeat) in Redis.
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import logging
import os
import socket
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
import redis.asyncio as aioredis

from cv_engine import evaluate_module, get_model_sha256, ModuleResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [node2-worker] %(message)s"
)
logger = logging.getLogger("node2_worker")

# -----------------------------------------------------------------------------
# Naming Scheme & Service Discovery Configuration
# -----------------------------------------------------------------------------
NODE_ID = os.getenv("NODE_ID", "worker-1")
SERVICE_NAME = os.getenv("SERVICE_NAME", "service://luminapv-vision-worker")
WORKER_PORT = int(os.getenv("PORT", "8001"))
WORKER_DNS_NAME = os.getenv("WORKER_DNS_NAME", f"{NODE_ID}.luminapv.local")
REDIS_URL = os.getenv("REDIS_URL", "redis://state.luminapv.local:6379/0")

HEARTBEAT_INTERVAL_SECONDS = float(os.getenv("HEARTBEAT_INTERVAL_SECONDS", "3.0"))
HEARTBEAT_TTL_SECONDS = int(os.getenv("HEARTBEAT_TTL_SECONDS", "10"))
REGISTRY_KEY_PREFIX = "luminapv:services:vision-worker"

# Fallback in-memory registry for standalone or unit-test environments without Redis
_LOCAL_REGISTRY_FALLBACK: Dict[str, Dict[str, Any]] = {}


class WorkerServiceRegistry:
    """
    Gerencia a identidade de nomeação DNS local e o auto-registro periódico
    (heartbeat com TTL) da instância de worker no catálogo distribuído Redis,
    mantendo fallback transparente em memória para testes unitários.
    """

    def __init__(self) -> None:
        self.node_id: str = NODE_ID
        self.service_name: str = SERVICE_NAME
        self.dns_name: str = WORKER_DNS_NAME
        self.port: int = WORKER_PORT
        self.redis_url: str = REDIS_URL
        self.redis_client: Optional[aioredis.Redis] = None
        self.registry_backend: str = "memory-fallback"
        self.heartbeat_count: int = 0
        self.last_seen: Optional[str] = None
        self._task: Optional[asyncio.Task] = None

    def build_node_metadata(self) -> Dict[str, Any]:
        now_iso = datetime.now(timezone.utc).isoformat()
        self.last_seen = now_iso
        hostname = socket.gethostname()
        return {
            "node_id": self.node_id,
            "service_name": self.service_name,
            "dns_name": self.dns_name,
            "hostname": hostname,
            "host": self.dns_name,
            "port": self.port,
            "endpoint": f"http://{self.dns_name}:{self.port}",
            "status": "ONLINE",
            "model_sha256": get_model_sha256(),
            "last_seen": now_iso,
            "ttl_seconds": HEARTBEAT_TTL_SECONDS,
            "heartbeat_count": self.heartbeat_count,
            "registry_backend": self.registry_backend,
        }

    async def register_heartbeat(self) -> Dict[str, Any]:
        """
        Registra ou renova o heartbeat deste worker no Redis (com TTL)
        ou no dicionário de fallback em memória caso o Redis não esteja acessível.
        """
        self.heartbeat_count += 1
        metadata = self.build_node_metadata()
        node_key = f"{REGISTRY_KEY_PREFIX}:{self.node_id}"
        index_key = f"{REGISTRY_KEY_PREFIX}:nodes"

        try:
            if self.redis_client is None:
                self.redis_client = aioredis.from_url(
                    self.redis_url,
                    decode_responses=True,
                    socket_connect_timeout=1.0,
                    socket_timeout=1.0,
                )
            await self.redis_client.set(
                node_key,
                json.dumps(metadata),
                ex=HEARTBEAT_TTL_SECONDS,
            )
            await self.redis_client.sadd(index_key, self.node_id)
            self.registry_backend = "redis"
            metadata["registry_backend"] = "redis"
        except Exception as exc:
            self.registry_backend = "memory-fallback"
            metadata["registry_backend"] = "memory-fallback"
            logger.debug(
                f"Redis indisponível em '{self.redis_url}' ({exc}); "
                f"usando registro local em memória para '{self.node_id}'."
            )

        _LOCAL_REGISTRY_FALLBACK[self.node_id] = metadata
        return metadata

    async def _heartbeat_loop(self) -> None:
        logger.info(
            f"Iniciando loop de heartbeat para '{self.node_id}' ({self.dns_name}:{self.port}) -> {self.service_name}"
        )
        while True:
            try:
                await self.register_heartbeat()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning(f"Falha no ciclo de heartbeat de '{self.node_id}': {exc}")
            await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)

    async def start(self) -> None:
        await self.register_heartbeat()
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._heartbeat_loop())

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self.redis_client is not None:
            try:
                await self.redis_client.delete(f"{REGISTRY_KEY_PREFIX}:{self.node_id}")
                await self.redis_client.close()
            except Exception:
                pass
            self.redis_client = None


registry = WorkerServiceRegistry()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await registry.start()
    yield
    await registry.stop()


app = FastAPI(
    title="LuminaPV CV Worker (Node 2)",
    description="Microservice for perspective homography, dynamic grid inference, and EL defect inspection.",
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


@app.get("/", tags=["Telemetry"])
async def root_info():
    """
    Root discovery endpoint for Node 2 Worker.
    """
    return {
        "service": "node2_worker",
        "node_id": registry.node_id,
        "service_name": registry.service_name,
        "dns_name": registry.dns_name,
        "name": "LuminaPV Computer Vision Worker",
        "status": "online",
        "endpoints": {
            "health": "/health",
            "registry_node": "/api/v1/registry/node",
            "process_module": "POST /api/v1/process-module",
            "docs": "/docs"
        }
    }


@app.get("/health", tags=["Telemetry"])
@app.get("/api/v1/health", tags=["Telemetry"])
async def health_check():
    """
    Cloud Run, Docker, and Service Discovery health/readiness probe.
    """
    meta = await registry.register_heartbeat()
    return {
        "status": "healthy",
        "service": "node2_worker",
        "version": "1.0.0",
        "node_id": meta["node_id"],
        "service_name": meta["service_name"],
        "dns_name": meta["dns_name"],
        "endpoint": meta["endpoint"],
        "model_sha256": meta["model_sha256"],
        "last_seen": meta["last_seen"],
        "registry_backend": meta["registry_backend"],
    }


@app.get("/api/v1/registry/node", tags=["Service Discovery"])
async def get_node_registry_info():
    """
    Retorna os metadados completos de auto-registro e nomeação deste nó worker.
    """
    return await registry.register_heartbeat()


@app.post(
    "/api/v1/process-module",
    response_model=ModuleResult,
    status_code=status.HTTP_200_OK,
    tags=["Computer Vision"],
    summary="Process a single full-size solar module EL image"
)
async def process_module(
    file: UploadFile = File(...),
    use_yolo: bool = True
):
    """
    Accepts raw module image binary via multipart/form-data.
    Performs perspective rectification, dynamic grid topology estimation,
    in-memory cell slicing, and luminescence defect classification.
    """
    filename = file.filename or "module_scan.png"
    logger.info(
        f"[{registry.node_id}] Received module inspection request for: "
        f"'{filename}' (type: {file.content_type}, use_yolo={use_yolo})"
    )

    try:
        image_bytes = await file.read()
    except Exception as e:
        logger.error(f"Failed to read upload stream for '{filename}': {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read image stream: {str(e)}"
        )

    if not image_bytes or len(image_bytes) == 0:
        logger.warning(f"Empty upload received for '{filename}'")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Image file '{filename}' is empty."
        )

    try:
        # Run computer vision evaluation pipeline tagged with current worker node_id
        result = evaluate_module(
            image_bytes,
            filename=filename,
            use_yolo=use_yolo,
            node_id=registry.node_id
        )
        logger.info(
            f"[{registry.node_id}] Successfully processed '{filename}': Grid={result.rows}x{result.cols}, "
            f"TotalCells={result.total_cells}, Defective={result.defective_cells_count}, "
            f"Damage={result.module_damage_percentage}%, Status={result.status}"
        )
        return result
    except ValueError as ve:
        logger.error(f"Image decode or vision processing failed for '{filename}': {ve}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unprocessable image stream for '{filename}': {str(ve)}"
        )
    except Exception as e:
        logger.exception(f"Unexpected error processing module '{filename}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal computer vision engine failure: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=WORKER_PORT)
