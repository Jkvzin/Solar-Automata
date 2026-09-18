# Solar-Automata ⚡🔬
### Distributed Optical Inspection & Electroluminescence (EL) Diagnostic Platform for Photovoltaic Modules

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8.svg?logo=opencv)](https://opencv.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.11+-8CAAE6.svg?logo=scipy)](https://scipy.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg?logo=docker)](https://www.docker.com/)
[![Google Cloud Run](https://img.shields.io/badge/GCP-Cloud_Run-4285F4.svg?logo=googlecloud)](https://cloud.google.com/run)
[![Tests Passing](https://img.shields.io/badge/tests-14%2F14%20passing-brightgreen.svg)]()
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
- [EL Hardware & Inspection Principles](#9-el-hardware--inspection-principles)
- [License & Credits](#10-license--credits)

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
│  - Host Port: 8080                                                       │
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
│                    NODE 2: Computer Vision Worker Node                   │
│                        (FastAPI + OpenCV + SciPy)                        │
│  - Host Port: 8001                                                       │
│  - Ingests single module raw image stream (PNG, JPEG, TIFF, WebP)        │
│  - Perspective Homography Normalization (rectification of tilted panels) │
│  - Dynamic Grid Topology Inference (Sobel gradients & valley projection) │
│  - Zero-disk in-memory cell slicing (R x C discrete NumPy sub-arrays)    │
│  - Luminescence defect classification (strict enum & calibrated score)   │
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

### ⚡ Zero-Disk In-Memory Processing Pipeline
- Entire inspection cycle—from multipart upload decoding to homography rectification, cell segmentation, luminescence analysis, and JSON serialization—executes exclusively in memory via NumPy/OpenCV buffers.
- Eliminates file system latency, prevents temporary file leaks, and enhances horizontal scaling on containerized environments like Google Cloud Run.

### 🛡️ Strict Defect Classification & Decision Rules
Each sliced cell is evaluated across luminescence histogram distributions, standard deviations, and local gradient anomalies:
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
├── node2_worker/                      # Computer Vision Worker (FastAPI + OpenCV + SciPy)
│   ├── main.py                        # Single module REST endpoint & health probes
│   ├── cv_engine.py                   # Dynamic Sobel grid detection & defect classification
│   ├── requirements.txt               # Worker dependencies (opencv, scipy, numpy, etc.)
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
│   ├── test_cv_engine.py              # Unit tests for CV algorithms & edge cases
│   └── test_api_integration.py        # Integration tests for distributed endpoints
│
├── docker-compose.yml                 # Multi-container orchestration with live mounts
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

# Build images and start services in the background
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

**Response (`200 OK`):**
```json
{
  "batch_summary": {
    "total_modules": 2,
    "defective_modules_count": 1,
    "batch_failure_percentage": 50.0,
    "average_cell_degradation_percentage": 5.4
  },
  "modules": [
    {
      "module_id": "MOD-2026-001",
      "filename": "scan_module_6x10_pass.png",
      "status": "PASS",
      "rows": 6,
      "cols": 10,
      "total_cells": 60,
      "defective_cells_count": 1,
      "module_damage_percentage": 1.7,
      "image_width": 1000,
      "image_height": 600,
      "cells": [
        {
          "index": 0,
          "row": 0,
          "col": 0,
          "is_defective": false,
          "defect_type": "none",
          "confidence": 0.99,
          "bbox": {
            "x": 20,
            "y": 18,
            "width": 94,
            "height": 92,
            "x_pct": 2.0,
            "y_pct": 3.0,
            "width_pct": 9.4,
            "height_pct": 15.3
          }
        }
      ]
    }
  ]
}
```

#### Gateway Health Check
`GET /health` or `GET /api/v1/health`  
Returns gateway status and upstream worker connectivity:
```json
{
  "status": "healthy",
  "service": "node1_gateway",
  "worker_service_url": "http://node2_worker:8001",
  "worker_connectivity": "connected"
}
```

---

### 🔬 Node 2: Vision Worker (Port 8001)

#### Process Individual Module Scan
`POST /api/v1/process-module`  
*Consumes:* `multipart/form-data` with single file upload under key `"file"`.

**Sample cURL Request:**
```bash
curl -X POST "http://localhost:8001/api/v1/process-module" \
  -F "file=@tests/fixtures/scan_module_6x10_pass.png"
```

---

## 6. Automated Testing & Verification

The repository includes a comprehensive, multi-layer automated test suite with deterministic module generators, computer vision unit tests, and end-to-end integration tests.

### 1. Generate Synthetic Photovoltaic Test Scans
Produces standardized test scans in `tests/fixtures/` ($6\times10$ PASS, $6\times12$ FAIL, $12\times10$ FAIL, and perspective-tilted $6\times10$):
```bash
python tests/generate_test_modules.py
```

### 2. Run Computer Vision Unit Tests
Validates dynamic grid peak detection, perspective homography rectification, in-memory cell slicing, defect classification enums, and corrupted file handling:
```bash
python -m unittest tests/test_cv_engine.py
```

### 3. Run Distributed Integration Tests
Tests parallel dispatch, multipart decoding, batch telemetry calculation, 422 validation, and worker routing:
```bash
python -m unittest tests/test_api_integration.py
```

### 4. Run the Full Test Suite
```bash
python -m unittest discover tests
```
*Expected Result:* `Ran 14 tests in ~0.35s - OK`

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

Solar-Automata is fully containerized and production-ready for deployment on **Google Cloud Run**:
1. **Node 2 Worker** is deployed as an internal/private or authenticated Cloud Run service.
2. The deployed worker HTTPS URL is injected into **Node 1 Gateway** as the `WORKER_SERVICE_URL` environment variable.
3. Node 1 serves the frontend and proxies bulk inspection traffic with automatic connection pooling and async dispatch.

Detailed step-by-step CLI commands, IAM permission configurations, and cost-optimization recommendations are available in:
📄 **[DEPLOYMENT_GCP.md](DEPLOYMENT_GCP.md)**

---

## 9. EL Hardware & Inspection Principles

### Physics of Electroluminescence in Photovoltaics
Electroluminescence inspection functions as the reverse of the photovoltaic effect:
1. When forward-biased with a DC power supply (operating near module short-circuit current $I_{sc}$), silicon solar cells emit near-infrared (NIR) radiative recombination photons with a peak around **$1150\text{ nm}$**.
2. **Defect Mechanism**: Micro-cracks and crystalline dislocations increase series resistance or induce shunt pathways, starving localized areas of current. In the resulting EL image, damaged zones appear as dark lines, inactive black patches ("dead cells"), or intense current concentrations ("hot-spots").
3. **Recommended Imaging Equipment**:
   - InGaAs cameras or cooled scientific Si-CCD cameras with enhanced NIR sensitivity (900–1700 nm).
   - Light-tight darkroom inspection tunnel or optical light shielding enclosure.

---

## 10. License & Credits

- **Repository**: [Jkvzin/Solar-Automata](https://github.com/Jkvzin/Solar-Automata)
- **Engine**: LuminaPV Distributed Optical Diagnostic Engine
- **License**: MIT License. Open for academic, industrial, and commercial solar quality control applications.
