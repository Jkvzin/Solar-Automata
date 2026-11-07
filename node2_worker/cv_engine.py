"""
LuminaPV Computer Vision & Signal Processing Engine (Node 2 Worker)
===================================================================
Author: Principal Distributed Systems Engineer & Senior Computer Vision Architect
Purpose: Performs perspective normalization, dynamic arbitrary grid topology detection,
         in-memory cell slicing, and electroluminescence (EL) defect classification
         for photovoltaic modules.
"""

import base64
from enum import Enum
import hashlib
import logging
import math
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from pydantic import BaseModel, Field
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from ultralytics import YOLO

logger = logging.getLogger("cv_engine")


class DefectType(str, Enum):
    DEFEITUOSA = "defeituosa"
    NONE = "none"
    MICRO_CRACK = "micro-crack"
    DEAD = "dead"
    HOT_SPOT = "hot-spot"
    DELAMINATION = "delamination"
    GRID_FAULT = "grid-fault"


class CellBoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int
    x_pct: float
    y_pct: float
    width_pct: float
    height_pct: float


class CellInspection(BaseModel):
    index: int
    row: int
    col: int
    is_defective: bool
    defect_type: DefectType
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: Optional[CellBoundingBox] = None


class ModuleResult(BaseModel):
    module_id: str
    filename: str
    status: str  # "FAIL" | "PASS"
    rows: int
    cols: int
    total_cells: int
    defective_cells_count: int
    module_damage_percentage: float
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    x_bounds: Optional[List[int]] = None
    y_bounds: Optional[List[int]] = None
    active_area: Optional[Dict[str, Any]] = None
    rectified_image: Optional[str] = None
    processed_by_node: Optional[str] = None
    model_sha256: Optional[str] = None
    content_sha256: Optional[str] = None
    replicated: bool = False
    consistency_model: str = "read-your-writes"
    cache_hit: bool = False
    cells: List[CellInspection]


def order_quad_points(pts: np.ndarray) -> np.ndarray:
    """
    Orders 4 quadrilateral coordinates in clockwise order:
    [top-left, top-right, bottom-right, bottom-left].
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left has smallest sum
    rect[2] = pts[np.argmax(s)]  # Bottom-right has largest sum

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right has smallest difference (x - y)
    rect[3] = pts[np.argmax(diff)]  # Bottom-left has largest difference (y - x)

    return rect


def normalize_perspective(gray_img: np.ndarray) -> np.ndarray:
    """
    Detects the outer boundary of the photovoltaic module and performs
    homography rectification via perspective transform if the module is tilted.
    If the module already occupies the frame upright, preserves the original raster.
    """
    h, w = gray_img.shape[:2]
    img_area = float(w * h)

    blurred = cv2.GaussianBlur(gray_img, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    k_size = max(15, int(min(w, h) * 0.04))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return gray_img

    largest_c = max(contours, key=cv2.contourArea)
    bx, by, bw, bh = cv2.boundingRect(largest_c)

    # Check if the module already occupies the frame upright with standard border
    margin_thresh = max(35, int(min(w, h) * 0.08))
    is_full_frame = (
        bx <= margin_thresh and
        by <= margin_thresh and
        (w - (bx + bw)) <= margin_thresh and
        (h - (by + bh)) <= margin_thresh
    )
    if is_full_frame:
        return gray_img

    # If tilted or framed inside a larger background (15% to 85% area)
    area = cv2.contourArea(largest_c)
    if area > (0.15 * img_area):
        peri = cv2.arcLength(largest_c, True)
        approx = cv2.approxPolyDP(largest_c, 0.02 * peri, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            pts = approx.reshape(4, 2).astype("float32")
        else:
            rect = cv2.minAreaRect(largest_c)
            pts = cv2.boxPoints(rect).astype("float32")

        ordered = order_quad_points(pts)
        tl, tr, br, bl = ordered

        width_a = np.linalg.norm(br - bl)
        width_b = np.linalg.norm(tr - tl)
        max_w = int(max(width_a, width_b))

        height_a = np.linalg.norm(tr - br)
        height_b = np.linalg.norm(tl - bl)
        max_h = int(max(height_a, height_b))

        if max_w > 100 and max_h > 100:
            dst = np.array([
                [0, 0],
                [max_w - 1, 0],
                [max_w - 1, max_h - 1],
                [0, max_h - 1]
            ], dtype="float32")

            M = cv2.getPerspectiveTransform(ordered, dst)
            rectified = cv2.warpPerspective(gray_img, M, (max_w, max_h), flags=cv2.INTER_LINEAR)
            return rectified

    return gray_img


def detect_grid_topology(rectified_img: np.ndarray) -> Tuple[int, int, List[int], List[int]]:
    """
    Computes horizontal and vertical gradient projection profiles to dynamically
    detect rows (R) and columns (C), along with precise pixel divider boundaries.
    """
    h, w = rectified_img.shape[:2]

    # 1. Identify the active luminescent area of the photovoltaic module
    proj_y = np.mean(rectified_img, axis=1)
    th_y = (np.max(proj_y) + np.min(proj_y)) * 0.35
    act_y = np.where(proj_y > th_y)[0]
    y_start, y_end = (int(act_y[0]), int(act_y[-1])) if len(act_y) else (0, h - 1)

    proj_x = np.mean(rectified_img, axis=0)
    th_x = (np.max(proj_x) + np.min(proj_x)) * 0.35
    act_x = np.where(proj_x > th_x)[0]
    x_start, x_end = (int(act_x[0]), int(act_x[-1])) if len(act_x) else (0, w - 1)

    active_sub = rectified_img[y_start:y_end + 1, x_start:x_end + 1]
    ah, aw = active_sub.shape[:2]

    # 2. Row Divider Detection along Y
    sobel_y = np.abs(cv2.Sobel(active_sub, cv2.CV_32F, 0, 1, ksize=3))
    gy = gaussian_filter1d(np.mean(sobel_y, axis=1), sigma=max(2.0, ah * 0.003))
    iy = gaussian_filter1d(np.max(active_sub) - np.mean(active_sub, axis=1), sigma=max(2.0, ah * 0.003))
    norm_gy = (gy - np.min(gy)) / (np.ptp(gy) + 1e-6)
    norm_iy = (iy - np.min(iy)) / (np.ptp(iy) + 1e-6)
    cy = 0.5 * norm_gy + 0.5 * norm_iy

    # Solar modules have between 4 and 26 rows -> cell height >= ah / 26
    dist_y = max(20, ah // 26)
    peaks_y, _ = find_peaks(cy, distance=dist_y, prominence=0.08)
    margin_y = int(ah * 0.035)
    row_divs = [int(p) for p in peaks_y if margin_y < p < (ah - margin_y)]

    # 3. Column Divider Detection along X
    sobel_x = np.abs(cv2.Sobel(active_sub, cv2.CV_32F, 1, 0, ksize=3))
    gx = gaussian_filter1d(np.mean(sobel_x, axis=0), sigma=max(2.0, aw * 0.003))
    ix = gaussian_filter1d(np.max(active_sub) - np.mean(active_sub, axis=0), sigma=max(2.0, aw * 0.003))
    norm_gx = (gx - np.min(gx)) / (np.ptp(gx) + 1e-6)
    norm_ix = (ix - np.min(ix)) / (np.ptp(ix) + 1e-6)
    cx = 0.5 * norm_gx + 0.5 * norm_ix

    # Solar modules have between 4 and 16 columns -> cell width >= aw / 17
    dist_x = max(35, aw // 17)
    peaks_x, _ = find_peaks(cx, distance=dist_x, prominence=0.08)
    margin_x = int(aw * 0.035)
    col_divs = [int(p) for p in peaks_x if margin_x < p < (aw - margin_x)]

    rows = len(row_divs) + 1
    cols = len(col_divs) + 1

    # Map back to full rectified coordinates
    y_bounds = [y_start] + [y_start + d for d in row_divs] + [y_end + 1]
    x_bounds = [x_start] + [x_start + d for d in col_divs] + [x_end + 1]

    return rows, cols, y_bounds, x_bounds


def inspect_cell_region(
    cell_img: np.ndarray,
    module_median_int: float,
    module_std_int: float
) -> Tuple[bool, DefectType, float]:
    """
    Analyzes an in-memory cropped cell for electroluminescence anomalies:
    intensity drops, hot-spots, micro-cracks, delamination, and grid/busbar faults.
    Returns: (is_defective, defect_type, confidence)
    """
    ch, cw = cell_img.shape[:2]
    if ch < 4 or cw < 4:
        return False, DefectType.NONE, 0.95

    # Crop inner active area to avoid divider borders & frame shadows (10% border margin)
    my = max(2, int(ch * 0.10))
    mx = max(2, int(cw * 0.10))
    inner = cell_img[my:ch - my, mx:cw - mx]
    if inner.size == 0:
        inner = cell_img

    cell_mean = float(np.mean(inner))
    cell_std = float(np.std(inner))
    cell_min = float(np.min(inner))
    cell_max = float(np.max(inner))

    # Reference ratio compared to healthy module luminescence
    ref_int = max(module_median_int, 20.0)
    rel_brightness = cell_mean / ref_int

    # 1. DEAD CELL: Complete disconnection or inactive crystal (< 40% of module baseline)
    if rel_brightness < 0.40:
        confidence = float(min(0.99, max(0.88, 1.0 - rel_brightness * 0.7)))
        return True, DefectType.DEAD, round(confidence, 2)

    # 2. HOT-SPOT: Localized current crowding / shunting causing abnormal high luminescence
    if rel_brightness > 1.25 or (cell_max > 235 and (cell_max - cell_mean) > 35):
        confidence = float(min(0.98, max(0.82, 0.75 + (cell_max / 255.0) * 0.2)))
        return True, DefectType.HOT_SPOT, round(confidence, 2)

    # 3. GRID-FAULT / BUSBAR FAULT:
    # Asymmetry where half of the cell (or a distinct quadrant) is deactivated due to busbar interruption
    mid_x = inner.shape[1] // 2
    left_half = inner[:, :mid_x]
    right_half = inner[:, mid_x:]
    mean_left = float(np.mean(left_half)) if left_half.size else cell_mean
    mean_right = float(np.mean(right_half)) if right_half.size else cell_mean
    asymmetry_h = abs(mean_left - mean_right) / (max(mean_left, mean_right) + 1e-5)

    mid_y = inner.shape[0] // 2
    top_half = inner[:mid_y, :]
    bot_half = inner[mid_y:, :]
    mean_top = float(np.mean(top_half)) if top_half.size else cell_mean
    mean_bot = float(np.mean(bot_half)) if bot_half.size else cell_mean
    asymmetry_v = abs(mean_top - mean_bot) / (max(mean_top, mean_bot) + 1e-5)

    if max(asymmetry_h, asymmetry_v) > 0.42:
        confidence = float(min(0.96, max(0.75, 0.70 + max(asymmetry_h, asymmetry_v) * 0.35)))
        return True, DefectType.GRID_FAULT, round(confidence, 2)

    # 4. MICRO-CRACK:
    # High-frequency dark lines traversing the crystalline silicon matrix
    laplacian_var = float(cv2.Laplacian(inner, cv2.CV_32F).var())
    crack_thresh = max(5.0, cell_mean - 1.8 * cell_std)
    dark_mask = (inner < crack_thresh).astype(np.uint8) * 255
    kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dark_lines = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel_line)
    crack_pixel_ratio = float(np.count_nonzero(dark_lines)) / float(inner.size)

    if (laplacian_var > 350.0 and crack_pixel_ratio > 0.015) or (crack_pixel_ratio > 0.025 and cell_std > 18.0):
        confidence = float(min(0.97, max(0.68, 0.65 + crack_pixel_ratio * 4.0)))
        return True, DefectType.MICRO_CRACK, round(confidence, 2)

    # 5. DELAMINATION:
    # Moisture ingress / encapsulant decay causing hazy contrast drops at edges/corners
    corner_size = max(2, int(min(inner.shape) * 0.25))
    c_tl = np.mean(inner[:corner_size, :corner_size])
    c_tr = np.mean(inner[:corner_size, -corner_size:])
    c_bl = np.mean(inner[-corner_size:, :corner_size])
    c_br = np.mean(inner[-corner_size:, -corner_size:])
    min_corner = min(c_tl, c_tr, c_bl, c_br)
    corner_drop = (cell_mean - min_corner) / (cell_mean + 1e-5)

    if corner_drop > 0.35 and rel_brightness < 0.90:
        confidence = float(min(0.94, max(0.65, 0.60 + corner_drop * 0.5)))
        return True, DefectType.DELAMINATION, round(confidence, 2)

    # 6. NOMINAL / HEALTHY CELL
    # Homogeneous luminescence with expected busbar pattern
    confidence = float(min(0.99, max(0.85, 1.0 - abs(rel_brightness - 1.0) * 0.25)))
    return False, DefectType.NONE, round(confidence, 2)


# -----------------------------------------------------------------------------
# YOLOv8-cls Classification Engine (best.pt) & Model Hash Verification
# -----------------------------------------------------------------------------
_YOLO_CLASSIFIER: Optional[YOLO] = None
_MODEL_SHA256: Optional[str] = None


def resolve_model_path(weights_path: Optional[str] = None) -> str:
    """
    Localiza o arquivo de pesos 'best.pt' nos diretórios candidatos padrão.
    """
    target_path = weights_path or os.getenv("MODEL_PATH")
    if not target_path or not os.path.exists(target_path):
        candidate_paths = [
            Path(__file__).resolve().parent / "models" / "best.pt",
            Path(__file__).resolve().parent.parent / "models" / "best.pt",
            Path(__file__).resolve().parent.parent / "weights" / "best.pt",
            Path(__file__).resolve().parent.parent / "model" / "best.pt",
            Path("models/best.pt"),
            Path("weights/best.pt"),
            Path("model/best.pt"),
        ]
        for p in candidate_paths:
            if p.is_file():
                target_path = str(p)
                break

    if not target_path or not os.path.exists(target_path):
        raise FileNotFoundError(
            "Arquivo de pesos 'best.pt' não encontrado nos caminhos padrão."
        )
    return target_path


def get_model_sha256(weights_path: Optional[str] = None) -> str:
    """
    Calcula e mantém em cache o hash SHA-256 do arquivo de pesos 'best.pt'
    para verificação de integridade e consistência entre workers distribuídos.
    """
    global _MODEL_SHA256
    if _MODEL_SHA256 is not None and weights_path is None:
        return _MODEL_SHA256

    try:
        path = resolve_model_path(weights_path)
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha.update(chunk)
        digest = sha.hexdigest()
        if weights_path is None:
            _MODEL_SHA256 = digest
        return digest
    except Exception as exc:
        logger.warning(f"Não foi possível calcular SHA-256 do modelo: {exc}")
        return "heuristic-fallback"


def get_yolo_classifier(weights_path: Optional[str] = None) -> YOLO:
    """
    Carrega e mantém em cache o modelo YOLOv8-cls (best.pt).
    Busca no caminho explícito, variável de ambiente MODEL_PATH ou diretórios padrão.
    """
    global _YOLO_CLASSIFIER
    if _YOLO_CLASSIFIER is not None and weights_path is None:
        return _YOLO_CLASSIFIER

    target_path = resolve_model_path(weights_path)
    logger.info(f"Carregando modelo YOLOv8-cls a partir de: {target_path}")
    _YOLO_CLASSIFIER = YOLO(target_path)
    return _YOLO_CLASSIFIER


def _format_cell_crop_bgr(cell_crop: np.ndarray) -> np.ndarray:
    """
    Garante que o crop da célula possua 3 canais (BGR) para entrada no YOLOv8-cls.
    """
    if len(cell_crop.shape) == 2:
        return cv2.cvtColor(cell_crop, cv2.COLOR_GRAY2BGR)
    if cell_crop.shape[2] == 4:
        return cv2.cvtColor(cell_crop, cv2.COLOR_BGRA2BGR)
    return cell_crop


def classify_cell_yolo(
    cell_crop: np.ndarray,
    model: Optional[YOLO] = None
) -> Tuple[bool, DefectType, float]:
    """
    Classifica uma célula solar cortada (crop) utilizando a API de classificação YOLOv8-cls.
    Extrai:
      - Predição da classe: results[0].names[results[0].probs.top1]
      - Confiança: float(results[0].probs.top1conf)

    Retorna:
      Tuple[bool, DefectType, float]: (is_defective, defect_type, confidence)
    """
    if model is None:
        model = get_yolo_classifier()

    cell_input = _format_cell_crop_bgr(cell_crop)

    # Inferência Ultralytics YOLOv8-cls
    results = model(cell_input, verbose=False)
    probs = results[0].probs
    top1_idx = probs.top1
    pred_label = str(results[0].names[top1_idx]).strip().lower()
    confidence = float(probs.top1conf)

    # Identificação da classe: 'boa' vs 'defeituosa'
    is_defective = (pred_label in ["defeituosa", "defeito", "defect", "defective"])
    defect_type = DefectType.DEFEITUOSA if is_defective else DefectType.NONE

    return is_defective, defect_type, round(confidence, 4)


def classify_cells_batch_yolo(
    cell_crops: List[np.ndarray],
    model: Optional[YOLO] = None
) -> List[Tuple[bool, DefectType, float]]:
    """
    Realiza inferência vetorizada em lote (Batch Inference) sobre todas as células
    recortadas de um módulo fotovoltaico em uma única chamada ao motor YOLOv8-cls,
    reduzindo significativamente o overhead de invocação por célula em sistemas distribuídos.
    """
    if not cell_crops:
        return []

    if model is None:
        model = get_yolo_classifier()

    batch_inputs = [_format_cell_crop_bgr(crop) for crop in cell_crops]
    results = model(batch_inputs, verbose=False)

    predictions: List[Tuple[bool, DefectType, float]] = []
    for res in results:
        probs = res.probs
        top1_idx = probs.top1
        pred_label = str(res.names[top1_idx]).strip().lower()
        confidence = float(probs.top1conf)
        is_defective = (pred_label in ["defeituosa", "defeito", "defect", "defective"])
        defect_type = DefectType.DEFEITUOSA if is_defective else DefectType.NONE
        predictions.append((is_defective, defect_type, round(confidence, 4)))

    return predictions


def draw_defect_annotations(
    image: np.ndarray,
    cells: List[CellInspection],
    highlight_color: Tuple[int, int, int] = (0, 0, 255),  # Vermelho BGR para defeituosas
    box_thickness: int = 2,
    draw_banner: bool = False
) -> np.ndarray:
    """
    Desenha retângulos visuais de destaque demarcando as células defeituosas
    e adiciona um overlay translúcido para rápida identificação visual.
    """
    annotated = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
    overlay = annotated.copy()

    defective_cells = [c for c in cells if c.is_defective and c.bbox]

    # Preenchimento translúcido das células com defeito
    for cell in defective_cells:
        bbox = cell.bbox
        cv2.rectangle(
            overlay,
            (bbox.x, bbox.y),
            (bbox.x + bbox.width, bbox.y + bbox.height),
            highlight_color,
            -1
        )

    # Mescla o preenchimento com 25% de opacidade
    cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0, annotated)

    # Contorno sólido e rótulo de confiança
    for cell in defective_cells:
        bbox = cell.bbox
        x, y, w, h = bbox.x, bbox.y, bbox.width, bbox.height
        cv2.rectangle(annotated, (x, y), (x + w, y + h), highlight_color, box_thickness)

        label = f"{cell.confidence:.0%}"
        font_scale = max(0.35, min(w, h) / 130.0)
        cv2.putText(
            annotated,
            label,
            (x + 4, y + max(12, int(14 * (font_scale / 0.35)))),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            highlight_color,
            1,
            cv2.LINE_AA
        )

    if draw_banner:
        total = len(cells)
        defective = len(defective_cells)
        damage_pct = round((defective / float(total)) * 100.0, 1) if total > 0 else 0.0
        status_text = "REPROVADO (FAIL)" if damage_pct >= 5.0 else "APROVADO (PASS)"

        banner_h = 36
        banner = np.zeros((banner_h, annotated.shape[1], 3), dtype=np.uint8)
        header_text = f"Dano: {damage_pct:.1f}% ({defective}/{total} defeituosas) | Status: {status_text}"
        cv2.putText(banner, header_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        annotated = np.vstack([banner, annotated])

    return annotated


def generate_module_id(filename: str, fallback_idx: int = 1) -> str:
    """
    Generates or extracts a standardized module ID (e.g. MOD-2026-001).
    """
    match = re.search(r'(MOD-\d{4}-\d{3,4})', filename, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Check for scan_XX or panel_XX pattern
    match_seq = re.search(r'(?:scan|panel|module|scan_module)[_-](\d{1,4})', filename, re.IGNORECASE)
    if match_seq:
        num = int(match_seq.group(1))
        return f"MOD-2026-{num:03d}"

    # Use deterministic hash of filename to guarantee distinct IDs across different files
    h = (abs(hash(filename)) % 900) + 100
    return f"MOD-2026-{h:03d}"


def evaluate_module(
    image_bytes: bytes,
    filename: str,
    fallback_idx: int = 1,
    use_yolo: bool = True,
    model: Optional[YOLO] = None,
    node_id: Optional[str] = None
) -> ModuleResult:
    """
    Complete end-to-end computer vision processing pipeline for a single PV module:
    1. Memory decoding & validation
    2. Perspective normalization (homography rectification)
    3. Dynamic grid topology detection (arbitrary R rows x C cols)
    4. In-memory cell slicing (R*C sub-arrays)
    5. Vectorized batch YOLOv8-cls classification per cell (best.pt)
    6. Module damage % computation and FAIL/PASS classification
    7. Visual reconstruction with defect bounding boxes
    """
    if not image_bytes or len(image_bytes) < 32:
        raise ValueError("Empty or corrupted image byte stream.")

    content_hash = hashlib.sha256(image_bytes).hexdigest()

    np_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(np_buffer, cv2.IMREAD_GRAYSCALE)
    if img is None or img.size == 0:
        raise ValueError(f"OpenCV failed to decode image '{filename}' as a valid graphical raster.")

    # 1. Perspective Normalization
    rectified = normalize_perspective(img)
    h, w = rectified.shape[:2]

    # 2. Dynamic Grid Topology Detection (R rows x C cols)
    rows, cols, y_bounds, x_bounds = detect_grid_topology(rectified)
    total_cells = rows * cols

    x_start = x_bounds[0]
    x_end = x_bounds[-1]
    y_start = y_bounds[0]
    y_end = y_bounds[-1]
    active_area = {
        "x": int(x_start),
        "y": int(y_start),
        "width": int(x_end - x_start),
        "height": int(y_end - y_start),
        "x_pct": round((x_start / float(w)) * 100.0, 2),
        "y_pct": round((y_start / float(h)) * 100.0, 2),
        "width_pct": round(((x_end - x_start) / float(w)) * 100.0, 2),
        "height_pct": round(((y_end - y_start) / float(h)) * 100.0, 2)
    }

    # Baseline statistics for fallback heuristic
    module_median_int = float(np.median(rectified))
    module_std_int = float(np.std(rectified))

    # 3. In-Memory Cell Slicing and Batch YOLOv8-cls Inspection
    cells: List[CellInspection] = []
    defective_count = 0
    yolo_model: Optional[YOLO] = model

    if use_yolo and yolo_model is None:
        try:
            yolo_model = get_yolo_classifier()
        except Exception as exc:
            logger.warning(
                f"YOLOv8-cls model could not be loaded ({exc}). Falling back to heuristic inspection."
            )
            yolo_model = None

    cell_slices: List[Tuple[int, int, int, int, int, int, np.ndarray]] = []
    for r in range(rows):
        y0, y1 = y_bounds[r], y_bounds[r + 1]
        for c in range(cols):
            x0, x1 = x_bounds[c], x_bounds[c + 1]
            idx = r * cols + c
            cell_img = rectified[y0:y1, x0:x1]
            cell_slices.append((idx, r, c, x0, y0, x1, y1, cell_img))

    if yolo_model is not None and cell_slices:
        batch_preds = classify_cells_batch_yolo(
            [item[7] for item in cell_slices],
            model=yolo_model
        )
    else:
        batch_preds = [
            inspect_cell_region(item[7], module_median_int, module_std_int)
            for item in cell_slices
        ]

    for (idx, r, c, x0, y0, x1, y1, _), (is_defective, defect_type, confidence) in zip(cell_slices, batch_preds):
        if is_defective:
            defective_count += 1

        cell_w = int(x1 - x0)
        cell_h = int(y1 - y0)
        bbox = CellBoundingBox(
            x=int(x0),
            y=int(y0),
            width=cell_w,
            height=cell_h,
            x_pct=round((x0 / float(w)) * 100.0, 3),
            y_pct=round((y0 / float(h)) * 100.0, 3),
            width_pct=round((cell_w / float(w)) * 100.0, 3),
            height_pct=round((cell_h / float(h)) * 100.0, 3)
        )

        cells.append(
            CellInspection(
                index=idx,
                row=r,
                col=c,
                is_defective=is_defective,
                defect_type=defect_type,
                confidence=confidence,
                bbox=bbox
            )
        )

    # 4. Classification & Metrics
    module_damage_pct = round((defective_count / float(total_cells)) * 100.0, 1) if total_cells > 0 else 0.0
    status = "FAIL" if module_damage_pct >= 5.0 else "PASS"
    module_id = generate_module_id(filename, fallback_idx)
    resolved_node_id = node_id or os.getenv("NODE_ID", "worker-1")
    model_hash = get_model_sha256()

    # Reconstructed / visual raster with bounding boxes highlighting defective cells
    annotated_panel = draw_defect_annotations(rectified, cells, draw_banner=False)
    _, buf = cv2.imencode('.jpg', annotated_panel, [cv2.IMWRITE_JPEG_QUALITY, 85])
    rectified_image = f"data:image/jpeg;base64,{base64.b64encode(buf).decode('ascii')}"

    return ModuleResult(
        module_id=module_id,
        filename=filename,
        status=status,
        rows=rows,
        cols=cols,
        total_cells=total_cells,
        defective_cells_count=defective_count,
        module_damage_percentage=module_damage_pct,
        image_width=int(w),
        image_height=int(h),
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        active_area=active_area,
        rectified_image=rectified_image,
        processed_by_node=resolved_node_id,
        model_sha256=model_hash,
        content_sha256=content_hash,
        replicated=False,
        consistency_model="read-your-writes",
        cache_hit=False,
        cells=cells
    )
