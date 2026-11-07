# Solar-Automata ⚡🔬
### Distributed Optical Inspection & Electroluminescence (EL) Diagnostic Platform for Photovoltaic Modules

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8.svg?logo=opencv)](https://opencv.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.11+-8CAAE6.svg?logo=scipy)](https://scipy.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg?logo=docker)](https://www.docker.com/)
[![Google Cloud Run](https://img.shields.io/badge/GCP-Cloud_Run-4285F4.svg?logo=googlecloud)](https://cloud.google.com/run)
[![Tests Passing](https://img.shields.io/badge/tests-19%2F19%20passing-brightgreen.svg)]()
[![Status](https://img.shields.io/badge/status-production--ready-success.svg)]()

**Solar-Automata** is a high-performance, distributed computer vision platform designed for automated optical quality assurance (QA) and electroluminescence (EL) diagnostic inspection of solar photovoltaic (PV) modules. 

Combining an asynchronous API gateway, high-precision computer vision worker nodes, and an industrial-grade diagnostic workstation console (**LuminaPV Console**), the system automatically detects crystalline silicon defects—including micro-cracks, inactive finger grids, hot-spots, cell delamination, and dead cells—without manual human intervention.

---

## 📑 Table of Contents

- [System Architecture & Topology](#1-system-architecture--topology)
- [Key Innovations & Technical Capabilities](#2-key-innovations--technical-capabilities)
- [Workspace & Repository Layout](#3-workspace--repository-layout)
- [Quickstart: Local Deployment](#4-quickstart-local-deployment)
  - [Option A: Docker Compose (Recommended)](#option-a-docker-compose-recommended)
  - [Option B: Native Python Execution](#option-b-native-python-execution)
- [REST API Reference & Data Contracts](#5-rest-api-reference--data-contracts)
- [Automated Testing & Verification](#6-automated-testing--verification)
- [Industrial Frontend Console (LuminaPV)](#7-industrial-frontend-console-luminapv)
- [Google Cloud Run Deployment](#8-google-cloud-run-deployment)
- [Distributed Systems Architecture (Entrega 3)](#9-distributed-systems-architecture-entrega-3)
- [EL Hardware & Inspection Principles](#10-el-hardware--inspection-principles)
- [License & Credits](#11-license--credits)

---

## 1. System Architecture & Topology

Solar-Automata is engineered around a decoupled, multi-node microservices architecture that separates user-facing traffic, bulk orchestration, and compute-intensive signal processing.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                   Browser / LuminaPV Diagnostic Console                  │
│             (Vanilla CSS/JS, Dark Technical Industrial Aesthetic)        │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │
           HTTP POST /api/v1/analyze/bulk (multipart/form-data; key="files")
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                     NODE 1: API Gateway & Orchestrator                   │
│                                (FastAPI)                                 │
│  - Host Port: 8080 (gateway.luminapv.local)                              │
│  - Serves modular LuminaPV frontend at GET /                             │
│  - Legacy single-file console fallback at GET /legacy                    │
│  - Ingests bulk uploads (N full-size module images concurrently)         │
│  - Dispatches parallel async HTTP requests via httpx.AsyncClient         │
│  - Aggregates module telemetry into real-time batch metrics summary      │
│  - Health probe & worker connectivity telemetry at GET /health           │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │
         Internal Async REST: POST /api/v1/process-module (raw binary)
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    NODE 2: Computer Vision Worker Nodes                  │
│               (FastAPI + OpenCV + SciPy + YOLOv8-cls + Redis)            │
│  - Replicas: worker-1.luminapv.local (8001), worker-2.luminapv.local     │
│  - Ingests single module raw image stream (PNG, JPEG, TIFF, WebP)        │
│  - Perspective Homography Normalization (rectification of tilted panels) │
│  - Dynamic Grid Topology Inference (Sobel gradients & valley projection) │
│  - Zero-disk in-memory cell slicing & Batch YOLOv8-cls inference         │
│  - Primary-Backup result replication & Read-Your-Writes cache            │
│  - Industrial Pass/Fail evaluation (FAIL if damage >= 5.0%, else PASS)   │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Key Innovations & Technical Capabilities

### 🔍 Pure Dynamic Grid Topology Inference
- **Zero hardcoded dimensions**: Adapts seamlessly to standard ($6\times10$, $6\times12$), half-cut ($12\times10$, $12\times12$), and shingled cell topologies without manual configuration or fixed aspect ratio constraints.
- **Sobel & Luminescence Valley Projections**: Computes horizontal and vertical gradient projection profiles combined with intensity valley extraction using `scipy.signal.find_peaks` with dynamic prominence thresholds to isolate cell busbars and inter-cell ribbons.

### 📐 Perspective Homography Rectification
- Robust quadrilateral contour estimation detects tilted, skewed, or off-axis solar panels captured by production line cameras.
- Calculates an optimal $3\times3$ homography matrix using `cv2.getPerspectiveTransform` and rectifies the active cell matrix to an orthogonal projection before grid slicing.

### ⚡ Zero-Disk In-Memory Processing & Vectorized Batch YOLOv8-cls
- Entire inspection cycle—from multipart upload decoding to homography rectification, cell segmentation, batch YOLOv8-cls inference (`best.pt`), and JSON serialization—executes exclusively in memory via NumPy/OpenCV buffers.
- Eliminates file system latency, prevents temporary file leaks, and enhances horizontal scaling on containerized environments.

### 🛡️ Strict Defect Classification & Decision Rules
Each sliced cell is evaluated via the trained **YOLOv8-cls** model (`boa` vs. `defeituosa`) with automatic fallback to luminescence heuristics:
- **`defeituosa`**: Cell classified as defective by the YOLOv8-cls neural network (`best.pt`).
- **`micro-crack`**: Hairline crystalline fractures causing localized luminescence loss.
- **`dead`**: Completely inactive cell or disconnected wafer area (near-zero emission).
- **`hot-spot`**: Localized resistance anomalies exhibiting abnormal high-intensity emission.
- **`delamination`**: EVA/encapsulant degradation with diffuse edge decay.
- **`grid-fault`**: Broken busbar or interrupted finger electrodes.
- **`none`**: Structurally sound, nominal cell.

$$\text{Module Damage Percentage} = \left( \frac{\text{Defective Cells Count}}{\text{Total Cells Count}} \right) \times 100$$

$$\text{Inspection Verdict} = \begin{cases} \mathbf{FAIL}, & \text{if } \text{Damage} \ge 5.0\% \\ \mathbf{PASS}, & \text{if } \text{Damage} < 5.0\% \end{cases}$$

### 📊 Consolidated Batch Telemetry
The orchestrator aggregates module results into industrial metrics:
- `total_modules`: Total modules inspected in the batch.
- `defective_modules_count`: Quantity of modules flagged as `FAIL`.
- `batch_failure_percentage`: $\frac{\text{Defective Modules}}{\text{Total Modules}} \times 100$.
- `average_cell_degradation_percentage`: Average cell degradation across the entire batch.

---

## 3. Workspace & Repository Layout

```
Solar-Automata/
├── frontend/                          # Decoupled LuminaPV diagnostic console
│   ├── index.html                     # Semantic HTML5 workstation entry point
│   ├── css/
│   │   ├── variables.css              # Design tokens (VS Code Dark Technical palette)
│   │   ├── layout.css                 # Grid/Flexbox workstation layout & panels
│   │   └── components.css             # Buttons, badges, tables, modals & cards
│   └── js/
│       ├── config.js                  # Global configuration, defect dictionary & helpers
│       ├── state.js                   # Reactive state, formatters, topology & logger
│       ├── api.js                     # REST ingestion, streaming fallback & CSV export
│       ├── charts.js                  # HiDPI canvas charts (sparkline & histogram)
│       ├── render.js                  # DOM rendering pipeline (viewport, plate, inspector)
│       └── app.js                     # Pan/zoom, events, shortcuts & boot sequence
│
├── node1_gateway/                     # API Gateway & Batch Orchestrator (FastAPI)
│   ├── main.py                        # Bulk ingestion, async gather & static host
│   ├── requirements.txt               # Gateway dependencies (fastapi, httpx, uvicorn)
│   └── Dockerfile                     # Production container spec
│
├── node2_worker/                      # Computer Vision Worker (FastAPI + OpenCV + YOLOv8-cls)
│   ├── main.py                        # REST endpoints, heartbeat registry & replication manager
│   ├── cv_engine.py                   # Dynamic Sobel grid detection & batch YOLOv8-cls engine
│   ├── models/best.pt                 # Trained YOLOv8-cls solar cell classification weights
│   ├── requirements.txt               # Worker dependencies (opencv, scipy, ultralytics, redis)
│   └── Dockerfile                     # Production container spec (Debian + GL/X11 libs)
│
├── fixtures/                          # Sample batch payloads for instant testing
│   └── luminapv-batch-sample.json     # Pre-analyzed batch dataset
│
├── tests/                             # Comprehensive automated test suite
│   ├── fixtures/                      # Sample synthetic scans & reference EL imagery
│   │   ├── scan_module_6x10_pass.png
│   │   ├── scan_module_6x12_fail.png
│   │   ├── scan_module_12x10_fail.png
│   │   └── scan_module_tilted_6x10.png
│   ├── generate_test_modules.py       # Deterministic EL solar panel scan generator
│   ├── test_cv_engine.py              # Unit tests for CV algorithms & batch YOLOv8-cls
│   └── test_api_integration.py        # Integration tests for distributed endpoints & replicas
│
├── docker-compose.yml                 # Multi-container orchestration (Registry + 2 Workers + Gateway)
├── DEPLOYMENT_GCP.md                  # Complete Google Cloud Run deployment manual
├── brand-spec.md                      # UI tokens & industrial design system specifications
├── luminapv-el-console.html           # Legacy monolithic workstation (maintained for parity)
└── README.md                          # Project documentation
```

---

## 4. Quickstart: Local Deployment

### Option A: Docker Compose (Recommended)

Requires Docker and Docker Compose installed:

```bash
# Clone repository
git clone https://github.com/Jkvzin/Solar-Automata.git
cd Solar-Automata

# Build images and start services in the background (Registry + 2 Workers + Gateway)
docker compose up --build -d

# Verify container health
docker compose ps

# View live aggregate logs
docker compose logs -f
```

Open your browser at:
👉 **[http://localhost:8080/](http://localhost:8080/)**

To shut down the cluster:
```bash
docker compose down
```

---

### Option B: Native Python Execution

#### Prerequisites
- Python 3.11 or higher
- C/C++ compiler and OpenCV system dependencies (if running on Linux)

#### 1. Setup Virtual Environment & Dependencies
```bash
# Create and activate virtual environment
python -m venv venv

# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Linux/macOS:
source venv/bin/activate

# Install dependencies for both nodes
pip install -r node2_worker/requirements.txt
pip install -r node1_gateway/requirements.txt
```

#### 2. Start Node 2 (CV Worker)
In your first terminal:
```bash
cd node2_worker
uvicorn main:app --host 0.0.0.0 --port 8001
```

#### 3. Start Node 1 (API Gateway)
In your second terminal:
```bash
cd node1_gateway

# Configure worker target URL
# Windows PowerShell:
$env:WORKER_SERVICE_URL="http://localhost:8001"
# Linux/macOS:
export WORKER_SERVICE_URL="http://localhost:8001"

uvicorn main:app --host 0.0.0.0 --port 8080
```

Access the UI at **[http://localhost:8080/](http://localhost:8080/)**.

---

## 5. REST API Reference & Data Contracts

### 🌐 Node 1: API Gateway (Port 8080)

#### Ingest Bulk Modules for Inspection
`POST /api/v1/analyze/bulk`  
*Consumes:* `multipart/form-data` with repeated file entries under key `"files"`.

**Sample cURL Request:**
```bash
curl -X POST "http://localhost:8080/api/v1/analyze/bulk" \
  -F "files=@tests/fixtures/scan_module_6x10_pass.png" \
  -F "files=@tests/fixtures/scan_module_6x12_fail.png"
```

---

### 🔬 Node 2: Vision Worker (Ports 8001 / 8002)

#### Process Individual Module Scan
`POST /api/v1/process-module`  
*Consumes:* `multipart/form-data` with single file upload under key `"file"`.

```bash
curl -X POST "http://localhost:8001/api/v1/process-module" \
  -F "file=@tests/fixtures/scan_module_6x10_pass.png"
```

#### Service Discovery & Replication Telemetry
- `GET /api/v1/registry/node`: Returns active node naming and heartbeat metadata.
- `GET /api/v1/replicas/stats`: Returns replication strategy, consistency model, cache hits/misses, and SHA-256 model consistency across the cluster.
- `GET /api/v1/replicas/{content_sha256}`: Fetches a replicated `ModuleResult` by image content SHA-256 hash.

---

## 6. Automated Testing & Verification

The repository includes a comprehensive, multi-layer automated test suite with deterministic module generators, computer vision unit tests, and end-to-end integration tests.

### 1. Run Computer Vision Unit Tests
```bash
python -m unittest tests/test_cv_engine.py
```

### 2. Run Distributed Integration Tests
```bash
python -m unittest tests/test_api_integration.py
```

### 3. Run the Full Test Suite
```bash
python -m unittest discover -s tests -v
```

---

## 7. Industrial Frontend Console (LuminaPV)

The frontend is built according to the **VS Code Dark Technical** industrial design specification (`brand-spec.md`):

| Characteristic | Specification |
|---|---|
| **Palette** | Flat 1px borders (`#2D2D30`), deep canvas (`#1E1E1E`), panel surfaces (`#252526`), accent blue (`#007ACC`) |
| **Typography** | UI: Segoe UI / Inter (11–13px); Telemetry: Monospace (`JetBrains Mono`, `Consolas`) |
| **Viewport** | Interactive pan, zoom, reset, 1:1 pixel inspection, and adaptive overlay toggle |
| **Telemetry** | Live distribution histograms and degradation sparklines rendered on HiDPI `<canvas>` |
| **Reporting** | Instant CSV report generation with complete cell-level coordinate, defect, and confidence metadata |
| **Architecture** | Zero framework overhead (Vanilla CSS variables + ES6 modules), responsive grid layout |

---

## 8. Google Cloud Run Deployment

Solar-Automata is fully containerized and production-ready for deployment on **Google Cloud Run**. Detailed step-by-step CLI commands, IAM permission configurations, and cost-optimization recommendations are available in:
📄 **[DEPLOYMENT_GCP.md](DEPLOYMENT_GCP.md)**

---

## 9. Distributed Systems Architecture (Entrega 3)

### 9.1. Esquema de Nomeação e Descoberta de Serviços (Naming & Service Discovery)
O cluster utiliza um esquema híbrido composto por **DNS estruturado na rede interna Docker** (`*.luminapv.local`) e **Catálogo Dinâmico com Auto-Registro via Heartbeat**:
- **Domínio Interno Padronizado**:
  - `gateway.luminapv.local` (`node1_gateway` — Porta `8080`)
  - `state.luminapv.local` / `redis.luminapv.local` (`luminapv-registry` — Porta `6379`)
  - `worker-1.luminapv.local` (`node2-worker-1` — Porta `8001`)
  - `worker-2.luminapv.local` (`node2-worker-2` — Porta `8002` externa / `8001` interna)
- **Identificador Lógico de Serviço**: `service://luminapv-vision-worker`.
- **Protocolo de Heartbeat**: Cada réplica de `node2_worker` executa um ciclo assíncrono em background (`HEARTBEAT_INTERVAL_SECONDS = 3.0s`) gravando seus metadados (`node_id`, `dns_name`, `endpoint`, `role`, `term`, `leader_id`, `model_sha256`, `last_seen`) na chave `luminapv:services:vision-worker:<node_id>` com expiração automática (`TTL = 10s`), permitindo detecção rápida de falhas e remoção imediata de nós inativos.
- **Roteamento, Balanceamento e Failover Dinâmico (`node1_gateway`)**: O resolvedor `ServiceCatalogResolver` consulta as instâncias saudáveis no catálogo e distribui os módulos do lote via **Round-Robin** concorrente (`asyncio.gather`). Caso uma instância falhe durante a transmissão, o gateway redireciona a requisição automaticamente para o próximo worker saudável registrado (*failover* transparente).
- **Telemetria Consolidada do Cluster**: O endpoint `GET /api/v1/cluster/status` expõe o líder eleito, termo, nós ativos, esquema de nomeação, locks distribuídos em vigor e estatísticas de replicação.

### 9.2. Mecanismos de Coordenação (Eleição de Líder e Exclusão Mútua)
- **Eleição de Líder entre Workers (`node2_worker`)**: Os nós ativos disputam um *lease* atômico distribuído no Redis (`SET luminapv:leader <payload> NX EX 8`) com renovação periódica. O nó eleito assume papel `LEADER`, incrementa o termo e atualiza o lease; as demais réplicas operam como `FOLLOWER`. Se o líder cair (`docker stop`), o lease expira em 8 segundos e os seguidores detectam a vacância, elegendo um novo líder em até 5–10 segundos. O estado é exposto em `GET /health` (`role`, `term`, `leader_id`) e pode ser testado via `POST /api/v1/cluster/election`.
- **Exclusão Mútua Distribuída (`DistributedLock` no `node1_gateway`)**: O processamento concorrente de módulos no `POST /api/v1/analyze/bulk` adquire locks atômicos no Redis baseados no hash SHA-256 do arquivo (`lock:module:<sha256>`) com TTL contra *deadlocks* e liberação atômica em bloco `finally` via script Lua (`_LUA_RELEASE_LOCK`). Locks ativos são rastreados em `luminapv:active_locks`.
- **Resiliência na Ingestão AWS**: As chamadas AWS S3/SQS em `analyze_bulk` são configuráveis via variável de ambiente (`AWS_INGESTION_ENABLED`) e protegidas por `try/except`, garantindo continuidade operacional sem erro HTTP 500 em execuções locais ou Docker sem credenciais na nuvem.

### 9.3. Replicação de Dados/Modelos e Especificação Formal do Modelo de Consistência
- **Estratégia de Replicação (*Primary-Backup* + Cache Distribuído de Inferência)**:
  - Todo resultado de inspeção (`ModuleResult`) processado por um worker é indexado pelo hash SHA-256 do conteúdo da imagem (`luminapv:replica:module:<content_sha256>`) e replicado simultaneamente no store primário compartilhado (`state.luminapv.local`) e nos stores em memória locais das réplicas (`_LOCAL_REPLICA_STORE`).
  - **Consistência de Pesos de IA (`model_sha256`)**: Cada registro replicado armazena o hash SHA-256 do arquivo de pesos `best.pt` (YOLOv8-cls). Ao consultar uma réplica em cache, o worker valida se `cached.model_sha256 == local.model_sha256`, invalidando automaticamente caches antigos caso o modelo seja atualizado.
- **Definição Formal do Modelo de Consistência**:
  1. **Consistência *Read-Your-Writes* (Leitura das Próprias Escritas — Centrada no Cliente)**:
     - Toda operação `POST /api/v1/process-module` ou `POST /api/v1/analyze/bulk` confirma a gravação síncrona no store primário antes de responder ao cliente. Assim, qualquer leitura subsequente realizada pelo mesmo operador ou pelo console web (`GET /api/v1/replicas/{sha256}` ou reenvio de lote) enxerga garantidamente a versão mais recente escrita.
  2. **Consistência Eventual (*Eventual Consistency* — Centrada em Dados entre Réplicas)**:
     - Os caches internos em memória (`_LOCAL_REPLICA_STORE`) das demais réplicas de `node2_worker` convergem assincronamente à medida que novas leituras ou ciclos de sincronização ocorrem.
  3. **Justificativa pelo Teorema CAP / PACELC**:
     - Em inspeção óptica industrial em linha de produção, o sistema prioriza **Disponibilidade e Baixa Latência (PA/EL)**: caso ocorra partição ou queda de um worker (`node2-worker-1`), o cluster continua operando através das réplicas sobreviventes (`node2-worker-2`) e do fallback local, garantindo continuidade operacional sem perda dos diagnósticos já replicados.

---

## 10. EL Hardware & Inspection Principles

### Physics of Electroluminescence in Photovoltaics
Electroluminescence inspection functions as the reverse of the photovoltaic effect:
1. When forward-biased with a DC power supply (operating near module short-circuit current $I_{sc}$), silicon solar cells emit near-infrared (NIR) radiative recombination photons with a peak around **$1150\text{ nm}$**.
2. **Defect Mechanism**: Micro-cracks and crystalline dislocations increase series resistance or induce shunt pathways, starving localized areas of current. In the resulting EL image, damaged zones appear as dark lines, inactive black patches ("dead cells"), or intense current concentrations ("hot-spots").
3. **Recommended Imaging Equipment**:
   - InGaAs cameras or cooled scientific Si-CCD cameras with enhanced NIR sensitivity (900–1700 nm).
   - Light-tight darkroom inspection tunnel or optical light shielding enclosure.

---

## 11. License & Credits

- **Repository**: [Jkvzin/Solar-Automata](https://github.com/Jkvzin/Solar-Automata)
- **Engine**: LuminaPV Distributed Optical Diagnostic Engine
- **License**: MIT License. Open for academic, industrial, and commercial solar quality control applications.
