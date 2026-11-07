"""
Node 2: Computer Vision Worker Service (FastAPI + OpenCV + SciPy + YOLOv8-cls)
==============================================================================
Exposes internal REST endpoint POST /api/v1/process-module for automated,
in-memory optical diagnostic inspection of photovoltaic electroluminescence scans,
performs periodic service discovery self-registration (heartbeat) in Redis,
and manages distributed Primary-Backup result replication & inference caching.
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
import socket
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
import redis.asyncio as aioredis

from cv_engine import evaluate_module, generate_module_id, get_model_sha256, ModuleResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [node2-worker] %(message)s"
)
logger = logging.getLogger("node2_worker")

# -----------------------------------------------------------------------------
# Naming Scheme, Service Discovery & Replication Configuration
# -----------------------------------------------------------------------------
NODE_ID = os.getenv("NODE_ID", "worker-1")
SERVICE_NAME = os.getenv("SERVICE_NAME", "service://luminapv-vision-worker")
WORKER_PORT = int(os.getenv("PORT", "8001"))
WORKER_DNS_NAME = os.getenv("WORKER_DNS_NAME", f"{NODE_ID}.luminapv.local")
REDIS_URL = os.getenv("REDIS_URL", "redis://state.luminapv.local:6379/0")

HEARTBEAT_INTERVAL_SECONDS = float(os.getenv("HEARTBEAT_INTERVAL_SECONDS", "3.0"))
HEARTBEAT_TTL_SECONDS = int(os.getenv("HEARTBEAT_TTL_SECONDS", "10"))
REPLICA_TTL_SECONDS = int(os.getenv("REPLICA_TTL_SECONDS", "3600"))

REGISTRY_KEY_PREFIX = "luminapv:services:vision-worker"
REPLICA_KEY_PREFIX = "luminapv:replica:module"
REPLICA_INDEX_KEY = "luminapv:replica:modules:index"
MODEL_HASH_KEY = "luminapv:replica:model:sha256"

# Fallback in-memory stores for standalone or unit-test environments without Redis
_LOCAL_REGISTRY_FALLBACK: Dict[str, Dict[str, Any]] = {}
_LOCAL_REPLICA_STORE: Dict[str, Dict[str, Any]] = {}


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


class DistributedResultReplicator:
    """
    Gerencia a Replicação Primário-Backup de resultados de inspeção (ModuleResult),
    o cache distribuído de inferência indexado por hash SHA-256 da imagem e a
    verificação de consistência do modelo YOLOv8-cls ('best.pt') entre nós.

    Modelo de Consistência:
      - 'read-your-writes': Leituras subsequentes de um cliente/gateway após a escrita
        de um lote enxergam imediatamente o resultado consolidado no store primário.
      - 'eventual': As réplicas locais de cada worker sincronizam assincronamente
        os artefatos replicados sob demanda e em background.
    """

    def __init__(self, service_registry: WorkerServiceRegistry) -> None:
        self.registry = service_registry
        self.consistency_model: str = "read-your-writes"
        self.replication_strategy: str = "primary-backup"
        self.cache_hits: int = 0
        self.cache_misses: int = 0

    async def replicate_result(self, result: ModuleResult, use_yolo: bool = True) -> ModuleResult:
        """
        Persiste e replica o ModuleResult no store distribuído (Redis) e na réplica local
        deste worker, marcando 'replicated=True' e registrando o hash do modelo.
        """
        content_key = f"{result.content_sha256}:yolo={int(use_yolo)}"
        redis_key = f"{REPLICA_KEY_PREFIX}:{content_key}"
        result.replicated = True
        result.consistency_model = self.consistency_model
        payload_dict = json.loads(result.model_dump_json())

        # 1. Atualiza réplica local em memória (garantindo Read-Your-Writes local imediato)
        _LOCAL_REPLICA_STORE[content_key] = payload_dict
        if result.content_sha256:
            _LOCAL_REPLICA_STORE[result.content_sha256] = payload_dict

        # 2. Propaga para o store distribuído Redis (Primary-Backup store compartilhado)
        client = self.registry.redis_client
        if client is not None:
            try:
                await client.set(
                    redis_key,
                    json.dumps(payload_dict),
                    ex=REPLICA_TTL_SECONDS
                )
                if result.content_sha256:
                    await client.set(
                        f"{REPLICA_KEY_PREFIX}:{result.content_sha256}",
                        json.dumps(payload_dict),
                        ex=REPLICA_TTL_SECONDS
                    )
                    await client.sadd(REPLICA_INDEX_KEY, result.content_sha256)
                if result.model_sha256:
                    await client.set(MODEL_HASH_KEY, result.model_sha256)
            except Exception as exc:
                logger.debug(f"Replicação Redis em fallback local para '{content_key}': {exc}")

        return result

    async def get_cached_result(
        self,
        image_bytes: bytes,
        filename: str,
        use_yolo: bool = True
    ) -> Optional[ModuleResult]:
        """
        Consulta o cache distribuído de inferência (Redis -> Réplica Local).
        Valida se o 'model_sha256' do registro corresponde exatamente ao 'best.pt'
        ativo neste worker para evitar inferências obsoletas (model drift).
        """
        content_sha256 = hashlib.sha256(image_bytes).hexdigest()
        content_key = f"{content_sha256}:yolo={int(use_yolo)}"
        redis_key = f"{REPLICA_KEY_PREFIX}:{content_key}"
        current_model_hash = get_model_sha256()

        raw_data: Optional[Dict[str, Any]] = None

        # 1. Tenta buscar do store compartilhado Redis
        client = self.registry.redis_client
        if client is not None:
            try:
                cached_str = await client.get(redis_key)
                if cached_str:
                    raw_data = json.loads(cached_str)
                    # Sincroniza réplica local (Consistência Eventual)
                    _LOCAL_REPLICA_STORE[content_key] = raw_data
                    _LOCAL_REPLICA_STORE[content_sha256] = raw_data
            except Exception:
                raw_data = None

        # 2. Fallback para a réplica local em memória
        if raw_data is None and content_key in _LOCAL_REPLICA_STORE:
            raw_data = dict(_LOCAL_REPLICA_STORE[content_key])

        if raw_data is None:
            self.cache_misses += 1
            return None

        # Validação de consistência de versão de pesos do modelo IA
        cached_model_hash = raw_data.get("model_sha256")
        if cached_model_hash and cached_model_hash != current_model_hash:
            logger.info(
                f"Invalidando cache para '{filename}': hash do modelo mudou "
                f"({cached_model_hash[:8]} != {current_model_hash[:8]})"
            )
            self.cache_misses += 1
            return None

        self.cache_hits += 1
        cloned = dict(raw_data)
        cloned["filename"] = filename
        cloned["module_id"] = generate_module_id(filename)
        cloned["replicated"] = True
        cloned["cache_hit"] = True
        cloned["consistency_model"] = self.consistency_model
        return ModuleResult(**cloned)

    async def get_replica_by_sha256(self, content_sha256: str) -> Optional[Dict[str, Any]]:
        """
        Recupera uma réplica específica pelo seu hash SHA-256 de conteúdo.
        """
        client = self.registry.redis_client
        if client is not None:
            try:
                raw = await client.get(f"{REPLICA_KEY_PREFIX}:{content_sha256}")
                if raw:
                    data = json.loads(raw)
                    _LOCAL_REPLICA_STORE[content_sha256] = data
                    return data
            except Exception:
                pass
        return _LOCAL_REPLICA_STORE.get(content_sha256)

    async def get_replication_stats(self) -> Dict[str, Any]:
        """
        Retorna telemetria consolidada sobre replicação de dados, cache distribuído
        e consistência dos pesos do modelo IA entre workers.
        """
        local_hash = get_model_sha256()
        cluster_model_hash = local_hash
        replicated_keys: List[str] = [
            k for k in _LOCAL_REPLICA_STORE.keys() if ":yolo=" not in k
        ]

        client = self.registry.redis_client
        if client is not None:
            try:
                remote_hash = await client.get(MODEL_HASH_KEY)
                if remote_hash:
                    cluster_model_hash = remote_hash
                members = await client.smembers(REPLICA_INDEX_KEY)
                if members:
                    replicated_keys = sorted(list(set(replicated_keys) | set(members)))
            except Exception:
                pass

        return {
            "node_id": self.registry.node_id,
            "dns_name": self.registry.dns_name,
            "consistency_model": self.consistency_model,
            "secondary_replica_sync": "eventual",
            "replication_strategy": self.replication_strategy,
            "registry_backend": self.registry.registry_backend,
            "model_sha256": local_hash,
            "cluster_model_sha256": cluster_model_hash,
            "model_consistent": (local_hash == cluster_model_hash),
            "total_replicated_modules": len(replicated_keys),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "replicated_keys": replicated_keys,
        }


registry = WorkerServiceRegistry()
replicator = DistributedResultReplicator(registry)


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
        "consistency_model": replicator.consistency_model,
        "name": "LuminaPV Computer Vision Worker",
        "status": "online",
        "endpoints": {
            "health": "/health",
            "registry_node": "/api/v1/registry/node",
            "replicas_stats": "/api/v1/replicas/stats",
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
    stats = await replicator.get_replication_stats()
    return {
        "status": "healthy",
        "service": "node2_worker",
        "version": "1.0.0",
        "node_id": meta["node_id"],
        "service_name": meta["service_name"],
        "dns_name": meta["dns_name"],
        "endpoint": meta["endpoint"],
        "model_sha256": meta["model_sha256"],
        "model_consistent": stats["model_consistent"],
        "consistency_model": stats["consistency_model"],
        "replicated_modules_count": stats["total_replicated_modules"],
        "last_seen": meta["last_seen"],
        "registry_backend": meta["registry_backend"],
    }


@app.get("/api/v1/registry/node", tags=["Service Discovery"])
async def get_node_registry_info():
    """
    Retorna os metadados completos de auto-registro e nomeação deste nó worker.
    """
    return await registry.register_heartbeat()


@app.get("/api/v1/replicas/stats", tags=["Replication & Consistency"])
async def get_replicas_stats():
    """
    Retorna estatísticas de replicação de dados, cache distribuído de inferência
    e verificação de consistência do modelo YOLOv8-cls ('best.pt').
    """
    return await replicator.get_replication_stats()


@app.get("/api/v1/replicas/{content_sha256}", tags=["Replication & Consistency"])
async def get_replicated_module(content_sha256: str):
    """
    Recupera um resultado de inspeção replicado a partir do hash SHA-256 da imagem.
    """
    replica = await replicator.get_replica_by_sha256(content_sha256)
    if replica is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Replica with content_sha256 '{content_sha256}' not found."
        )
    return replica


@app.post(
    "/api/v1/process-module",
    response_model=ModuleResult,
    status_code=status.HTTP_200_OK,
    tags=["Computer Vision"],
    summary="Process a single full-size solar module EL image"
)
async def process_module(
    file: UploadFile = File(...),
    use_yolo: bool = True,
    use_cache: bool = True
):
    """
    Accepts raw module image binary via multipart/form-data.
    Checks distributed inference cache first (Read-Your-Writes consistency);
    if not cached, performs perspective rectification, dynamic grid topology estimation,
    in-memory cell slicing, batch YOLOv8-cls classification, and replicates the result.
    """
    filename = file.filename or "module_scan.png"
    logger.info(
        f"[{registry.node_id}] Received module inspection request for: "
        f"'{filename}' (type: {file.content_type}, use_yolo={use_yolo}, use_cache={use_cache})"
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
        if use_cache:
            cached = await replicator.get_cached_result(
                image_bytes,
                filename=filename,
                use_yolo=use_yolo
            )
            if cached is not None:
                logger.info(
                    f"[{registry.node_id}] Distributed cache HIT for '{filename}' "
                    f"(sha256={cached.content_sha256[:12]}..., originally processed by {cached.processed_by_node})"
                )
                return cached

        # Run computer vision evaluation pipeline tagged with current worker node_id
        result = evaluate_module(
            image_bytes,
            filename=filename,
            use_yolo=use_yolo,
            node_id=registry.node_id
        )
        result = await replicator.replicate_result(result, use_yolo=use_yolo)
        logger.info(
            f"[{registry.node_id}] Successfully processed & replicated '{filename}': Grid={result.rows}x{result.cols}, "
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
