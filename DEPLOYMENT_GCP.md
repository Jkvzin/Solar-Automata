# Google Cloud Run Distributed Deployment Guide
## LuminaPV Two-Node Optical Inspection Backend

This document details the complete, step-by-step production deployment workflow for the LuminaPV distributed architecture on **Google Cloud Run**, satisfying rigorous academic and industrial criteria for containerized microservice orchestration, service discovery, and secure inter-service REST communication.

---

## 1. Architectural Topology on Google Cloud

```
[ Web Browser Client ]
          │
          │ HTTPS (Public Port 443)
          ▼
┌──────────────────────────────────────────────────────────────┐
│ Cloud Run Service 1: `luminapv-gateway` (Node 1)             │
│ - Ingress: All (Publicly accessible)                         │
│ - Serves: LuminaPV HTML Diagnostic Console & Static Assets    │
│ - Ingests: POST /api/v1/analyze/bulk (multipart/form-data)   │
│ - Dispatches: Parallel REST requests to Node 2               │
│ - Aggregates: Batch metrics & damage statistics              │
└──────────────────────────────────────────────────────────────┘
          │
          │ HTTPS (Internal Service-to-Service REST Call)
          │ Destination: $WORKER_SERVICE_URL/api/v1/process-module
          ▼
┌──────────────────────────────────────────────────────────────┐
│ Cloud Run Service 2: `luminapv-worker` (Node 2)              │
│ - Ingress: Internal / IAM-protected                          │
│ - Compute: 2 vCPU, 2 GiB RAM (OpenCV / SciPy / NumPy)        │
│ - Perspective Normalization & Dynamic Grid Detection         │
│ - In-Memory Cell Slicing & Defect Classification             │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. Prerequisites & Cloud Setup

### 2.1 Install Google Cloud CLI & Authenticate
Ensure `gcloud` CLI is installed and configured:

```bash
# Login to Google Cloud
gcloud auth login

# Set your active Google Cloud Project ID
PROJECT_ID="your-gcp-project-id"
REGION="us-central1"

gcloud config set project $PROJECT_ID
gcloud config set run/region $REGION
```

### 2.2 Enable Required Google Cloud APIs
Enable Cloud Run, Cloud Build, and Artifact Registry:

```bash
gcloud services enable \
    run.googleapis.com \
    artifactregistry.googleapis.com \
    cloudbuild.googleapis.com
```

### 2.3 Create Artifact Registry Docker Repository
```bash
gcloud artifacts repositories create luminapv-repo \
    --repository-format=docker \
    --location=$REGION \
    --description="Docker repository for LuminaPV microservices"
```

---

## 3. Step 1: Build & Deploy Node 2 (CV Worker) First

Node 2 must be deployed first so its generated Cloud Run HTTPS URL can be captured and injected into Node 1.

### 3.1 Build and Push Node 2 Worker Image
From the repository root:

```bash
gcloud builds submit ./node2_worker \
    --tag $REGION-docker.pkg.dev/$PROJECT_ID/luminapv-repo/node2-worker:latest
```

### 3.2 Deploy Node 2 Worker on Cloud Run
Deploy with adequate memory and CPU to handle high-resolution image processing and matrix transformations:

```bash
gcloud run deploy luminapv-worker \
    --image $REGION-docker.pkg.dev/$PROJECT_ID/luminapv-repo/node2-worker:latest \
    --platform managed \
    --region $REGION \
    --cpu 2 \
    --memory 2Gi \
    --concurrency 10 \
    --timeout 120s \
    --allow-unauthenticated
```

> **Note on Service-to-Service Security**: For enterprise zero-trust deployments where the worker is private (without `--allow-unauthenticated`), see Section 5 below.

### 3.3 Capture and Export the Generated Worker HTTPS URL
Capture the dynamic service URL assigned by Google Cloud Run:

```bash
# Linux / macOS / Cloud Shell:
export WORKER_SERVICE_URL=$(gcloud run services describe luminapv-worker \
    --region $REGION \
    --format='value(status.url)')

echo "Worker deployed at: $WORKER_SERVICE_URL"
```

In Windows PowerShell:
```powershell
$WORKER_SERVICE_URL = (gcloud run services describe luminapv-worker `
    --region $REGION `
    --format='value(status.url)')

Write-Host "Worker deployed at: $WORKER_SERVICE_URL"
```

---

## 4. Step 2: Build & Deploy Node 1 (API Gateway)

Now build and deploy Node 1, injecting `$WORKER_SERVICE_URL` as an environment variable.

### 4.1 Build and Push Node 1 Gateway Image
From the repository root (including `luminapv-el-console.html` and `fixtures/`):

```bash
gcloud builds submit . \
    --config=cloudbuild-gateway.yaml \
    --substitutions=_REGION=$REGION,_PROJECT_ID=$PROJECT_ID
```

*(Alternatively, build directly using Docker and push to Artifact Registry:)*
```bash
docker build -f node1_gateway/Dockerfile -t $REGION-docker.pkg.dev/$PROJECT_ID/luminapv-repo/node1-gateway:latest .
docker push $REGION-docker.pkg.dev/$PROJECT_ID/luminapv-repo/node1-gateway:latest
```

### 4.2 Deploy Node 1 Gateway with Injected Worker URL
```bash
gcloud run deploy luminapv-gateway \
    --image $REGION-docker.pkg.dev/$PROJECT_ID/luminapv-repo/node1-gateway:latest \
    --platform managed \
    --region $REGION \
    --cpu 1 \
    --memory 1Gi \
    --concurrency 80 \
    --timeout 90s \
    --allow-unauthenticated \
    --set-env-vars WORKER_SERVICE_URL=$WORKER_SERVICE_URL
```

### 4.3 Obtain the Public LuminaPV Console URL
```bash
CONSOLE_URL=$(gcloud run services describe luminapv-gateway \
    --region $REGION \
    --format='value(status.url)')

echo "LuminaPV Optical Inspection Console is live at: $CONSOLE_URL"
```

Open `$CONSOLE_URL` in any modern web browser to access the live console.

---

## 5. Enterprise Service-to-Service IAM Authentication (Optional)

In academic or enterprise zero-trust configurations, Node 2 should reject all public traffic and only accept authenticated requests signed by Node 1's Service Account.

### 5.1 Create a Dedicated Gateway Service Account
```bash
gcloud iam service-accounts create luminapv-gateway-sa \
    --display-name="LuminaPV API Gateway Service Account"
```

### 5.2 Grant Cloud Run Invoker Role on Node 2 Worker
```bash
gcloud run services add-iam-policy-binding luminapv-worker \
    --region $REGION \
    --member="serviceAccount:luminapv-gateway-sa@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/run.invoker"
```

### 5.3 Redeploy Node 2 with Ingress Restricted & No Public Access
```bash
gcloud run services update luminapv-worker \
    --region $REGION \
    --no-allow-unauthenticated \
    --ingress internal-and-cloud-load-balancing
```

### 5.4 Attach the Service Account to Node 1 Gateway
```bash
gcloud run services update luminapv-gateway \
    --region $REGION \
    --service-account "luminapv-gateway-sa@$PROJECT_ID.iam.gserviceaccount.com"
```

---

## 6. Verification and Smoke Testing

### 6.1 Test Gateway Health Endpoint
```bash
curl -i $CONSOLE_URL/health
```
Expected output:
```json
{
  "status": "healthy",
  "service": "node1_gateway",
  "worker_service_url": "https://luminapv-worker-...",
  "worker_connectivity": "connected"
}
```

### 6.2 Test Bulk Ingestion via CLI
```bash
curl -X POST "$CONSOLE_URL/api/v1/analyze/bulk" \
    -F "files=@tests/fixtures/synthetic_module_6x10_pass.png" \
    -F "files=@tests/fixtures/synthetic_module_6x12_fail.png"
```

---

## 7. Teardown and Resource Cleanup
To prevent ongoing charges after validation:

```bash
# Delete Cloud Run services
gcloud run services delete luminapv-gateway --region $REGION --quiet
gcloud run services delete luminapv-worker --region $REGION --quiet

# Delete Artifact Registry repository
gcloud artifacts repositories delete luminapv-repo --location $REGION --quiet
```
