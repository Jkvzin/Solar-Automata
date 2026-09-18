"""
Node 2: Computer Vision Worker Service (FastAPI + OpenCV + SciPy)
================================================================
Exposes internal REST endpoint POST /api/v1/process-module for automated,
in-memory optical diagnostic inspection of photovoltaic electroluminescence scans.
"""

import logging
from typing import Optional
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from cv_engine import evaluate_module, ModuleResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [node2-worker] %(message)s"
)
logger = logging.getLogger("node2_worker")

app = FastAPI(
    title="LuminaPV CV Worker (Node 2)",
    description="Microservice for perspective homography, dynamic grid inference, and EL defect inspection.",
    version="1.0.0"
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
        "name": "LuminaPV Computer Vision Worker",
        "status": "online",
        "endpoints": {
            "health": "/health",
            "process_module": "POST /api/v1/process-module",
            "docs": "/docs"
        }
    }


@app.get("/health", tags=["Telemetry"])
@app.get("/api/v1/health", tags=["Telemetry"])
async def health_check():
    """
    Cloud Run and Docker health/readiness probe.
    """
    return {
        "status": "healthy",
        "service": "node2_worker",
        "version": "1.0.0"
    }


@app.post(
    "/api/v1/process-module",
    response_model=ModuleResult,
    status_code=status.HTTP_200_OK,
    tags=["Computer Vision"],
    summary="Process a single full-size solar module EL image"
)
async def process_module(file: UploadFile = File(...)):
    """
    Accepts raw module image binary via multipart/form-data.
    Performs perspective rectification, dynamic grid topology estimation,
    in-memory cell slicing, and luminescence defect classification.
    """
    filename = file.filename or "module_scan.png"
    logger.info(f"Received module inspection request for: '{filename}' (type: {file.content_type})")

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
        # Run computer vision evaluation pipeline
        result = evaluate_module(image_bytes, filename=filename)
        logger.info(
            f"Successfully processed '{filename}': Grid={result.rows}x{result.cols}, "
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
    uvicorn.run(app, host="0.0.0.0", port=8001)
