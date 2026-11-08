FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency definition and install packages
COPY node1_gateway/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy gateway source code and static assets
COPY node1_gateway/main.py .
COPY node1_gateway/auth.py .
COPY node1_gateway/history.py .
COPY luminapv-el-console.html ./luminapv-el-console.html
COPY frontend/ ./frontend/
COPY fixtures/ ./fixtures/

ENV PORT=8080
EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
