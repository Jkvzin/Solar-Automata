#!/usr/bin/env python3
"""
LuminaPV - Script de Demonstração e Validação Ponta a Ponta
===========================================================
Pipeline:
1. Ingestão da imagem do painel/módulo solar completo.
2. Normalização de perspectiva e detecção dinâmica da grade (R linhas x C colunas).
3. Recorte em memória das células individuais (crops).
4. Inferência de classificação com YOLOv8-cls (best.pt) para cada célula.
   - Extrai: results[0].names[results[0].probs.top1]
   - Confiança: float(results[0].probs.top1conf)
5. Desenho dos retângulos visuais demarcando as células defeituosas na imagem reconstruída.
6. Cálculo e exibição da porcentagem total de dano: (células defeituosas / total de células) * 100.
7. Salvamento da imagem anotada resultante.

Uso:
    python test_panel.py
    python test_panel.py --image caminho/painel.png --model node2_worker/models/best.pt --output resultado.png
"""

import argparse
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from ultralytics import YOLO

# Adiciona o diretório do worker para reutilizar os métodos do motor CV
WORKER_DIR = Path(__file__).resolve().parent / "node2_worker"
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

try:
    from cv_engine import normalize_perspective, detect_grid_topology
except ImportError:
    # Fallback se executado fora da raiz do repositório
    def normalize_perspective(img):
        return img
    def detect_grid_topology(img):
        h, w = img.shape[:2]
        return 6, 10, [int(i * h / 6) for i in range(7)], [int(j * w / 10) for j in range(11)]


def find_default_weights() -> str:
    """Busca o arquivo best.pt nas localizações padrão."""
    candidates = [
        Path(__file__).resolve().parent / "node2_worker" / "models" / "best.pt",
        Path(__file__).resolve().parent / "models" / "best.pt",
        Path(__file__).resolve().parent / "weights" / "best.pt",
        Path(__file__).resolve().parent / "model" / "best.pt",
        Path(__file__).resolve().parent.parent / "model" / "best.pt",
        Path("node2_worker/models/best.pt"),
        Path("models/best.pt"),
        Path("weights/best.pt"),
        Path("model/best.pt")
    ]
    for c in candidates:
        if c.is_file():
            return str(c.resolve())
    return "models/best.pt"


def find_default_image() -> str:
    """Busca uma imagem padrão de teste no repositório."""
    candidates = [
        Path(__file__).resolve().parent / "tests" / "fixtures" / "scan_module_6x12_fail.png",
        Path(__file__).resolve().parent / "tests" / "fixtures" / "scan_module_6x10_pass.png",
        Path(__file__).resolve().parent / "tests" / "fixtures" / "scan_module_12x10_fail.png",
        Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "scan_module_6x12_fail.png"
    ]
    for c in candidates:
        if c.is_file():
            return str(c.resolve())
    return ""


def run_pipeline(
    image_path: str,
    model_path: str,
    output_path: str = "resultado_painel.png",
    damage_threshold: float = 5.0
):
    print("=" * 70)
    print("  LuminaPV - Inspeção de Módulos Solares via YOLOv8-cls")
    print("=" * 70)

    # 1. Carregamento do Modelo
    if not os.path.isfile(model_path):
        print(f"[ERRO] Arquivo de pesos '{model_path}' não encontrado!")
        print("Certifique-se de posicionar o arquivo 'best.pt' no diretório correto.")
        sys.exit(1)

    print(f"[*] Carregando modelo YOLOv8-cls: {model_path}")
    t0_model = time.perf_counter()
    model = YOLO(model_path)
    t_model = (time.perf_counter() - t0_model) * 1000
    print(f"    Classes mapeadas no modelo: {model.names}")
    print(f"    Modelo carregado em {t_model:.1f}ms\n")

    # 2. Leitura da Imagem
    if not os.path.isfile(image_path):
        print(f"[ERRO] Imagem de teste '{image_path}' não encontrada!")
        sys.exit(1)

    print(f"[*] Carregando imagem do painel: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERRO] Falha ao decodificar imagem '{image_path}'.")
        sys.exit(1)

    h_orig, w_orig = img.shape[:2]
    print(f"    Resolução original: {w_orig}x{h_orig} px")

    # Converter para escala de cinza para etapas de segmentação de grade
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    # 3. Retificação de Perspectiva e Segmentação da Grade
    print("[*] Normalizando perspectiva e detectando topologia da grade...")
    rectified_gray = normalize_perspective(gray)
    rows, cols, y_bounds, x_bounds = detect_grid_topology(rectified_gray)
    total_cells = rows * cols
    print(f"    Grade identificada: {rows} linhas x {cols} colunas (Total: {total_cells} células)")

    # Obter versão BGR da imagem retificada para desenho
    if len(img.shape) == 3 and (rectified_gray.shape == gray.shape):
        rectified_bgr = img.copy()
    else:
        rectified_bgr = cv2.cvtColor(rectified_gray, cv2.COLOR_GRAY2BGR)

    # 4. Recorte e Classificação Célula por Célula
    print(f"[*] Iniciando classificação de {total_cells} células com YOLOv8-cls...")
    cells_data = []
    defective_count = 0
    t0_infer = time.perf_counter()

    for r in range(rows):
        y0, y1 = y_bounds[r], y_bounds[r + 1]
        for c in range(cols):
            x0, x1 = x_bounds[c], x_bounds[c + 1]
            idx = r * cols + c

            # Recorte da célula diretamente em memória (crop)
            cell_crop = rectified_bgr[y0:y1, x0:x1]

            # Inferência Ultralytics YOLOv8-cls
            results = model(cell_crop, verbose=False)

            # Extração da predição e confiança requeridas
            top1_idx = results[0].probs.top1
            pred_class = str(results[0].names[top1_idx]).strip().lower()
            confidence = float(results[0].probs.top1conf)

            # Determinar se a célula é defeituosa
            is_defective = (pred_class in ["defeituosa", "defeito", "defect", "defective"])
            if is_defective:
                defective_count += 1

            cells_data.append({
                "index": idx,
                "row": r,
                "col": c,
                "bbox": (x0, y0, x1 - x0, y1 - y0),
                "class": pred_class,
                "is_defective": is_defective,
                "confidence": confidence
            })

    t_infer = (time.perf_counter() - t0_infer) * 1000
    avg_per_cell = t_infer / max(1, total_cells)
    print(f"    Inferência concluída em {t_infer:.1f}ms ({avg_per_cell:.2f}ms/célula)\n")

    # 5. Cálculo da Porcentagem Total de Dano
    module_damage_percentage = (defective_count / float(total_cells)) * 100.0 if total_cells > 0 else 0.0
    status = "REPROVADO (FAIL)" if module_damage_percentage >= damage_threshold else "APROVADO (PASS)"

    # 6. Desenho Visual dos Retângulos Demarcando as Células Defeituosas
    print("[*] Renderizando destaques visuais e reconstruindo painel...")
    annotated = rectified_bgr.copy()
    overlay = annotated.copy()

    # Destaque translúcido vermelho nas células defeituosas
    for cell in cells_data:
        if cell["is_defective"]:
            x, y, w, h = cell["bbox"]
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), -1)

    cv2.addWeighted(overlay, 0.28, annotated, 0.72, 0, annotated)

    # Contorno sólido nos retângulos defeituosos com etiqueta de confiança
    for cell in cells_data:
        x, y, w, h = cell["bbox"]
        if cell["is_defective"]:
            # Bounding box vermelho de destaque
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
            label = f"{cell['confidence']:.0%}"
            cv2.putText(
                annotated,
                label,
                (x + 3, y + max(12, int(h * 0.25))),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )
        else:
            # Contorno sutil verde/cinza para células sadias
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (60, 60, 60), 1)

    # Banner superior com métricas consolidadas
    banner_h = 42
    banner = np.zeros((banner_h, annotated.shape[1], 3), dtype=np.uint8)
    status_bgr = (0, 0, 255) if module_damage_percentage >= damage_threshold else (0, 200, 0)

    cv2.putText(
        banner,
        f"LuminaPV Diagnostic | Grade: {rows}x{cols} ({total_cells} celulas)",
        (10, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA
    )
    cv2.putText(
        banner,
        f"Dano: {module_damage_percentage:.1f}% ({defective_count} defeituosas / {total_cells - defective_count} boas) - STATUS: {status}",
        (10, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        status_bgr,
        2,
        cv2.LINE_AA
    )

    final_image = np.vstack([banner, annotated])

    # 7. Salvar Imagem Anotada
    cv2.imwrite(output_path, final_image)
    print(f"    Imagem anotada salva com sucesso em: {output_path}")

    # 8. Relatório Final no Terminal
    print("\n" + "=" * 70)
    print("  RELATÓRIO DE INSPEÇÃO DO MÓDULO")
    print("=" * 70)
    print(f"  • Imagem de entrada:          {image_path}")
    print(f"  • Topologia da grade:         {rows} linhas x {cols} colunas")
    print(f"  • Total de células:           {total_cells}")
    print(f"  • Células boas:               {total_cells - defective_count} ({(100.0 - module_damage_percentage):.1f}%)")
    print(f"  • Células defeituosas:        {defective_count} ({module_damage_percentage:.1f}%)")
    print(f"  • Porcentagem total de dano:  {module_damage_percentage:.2f}%")
    print(f"  • Limite de reprovação:       {damage_threshold:.1f}%")
    print(f"  • Resultado da Inspeção:      {status}")
    print(f"  • Arquivo de saída gerado:    {output_path}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="LuminaPV - Teste de classificação de células com YOLOv8-cls"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=find_default_image(),
        help="Caminho para a imagem do painel solar completo."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=find_default_weights(),
        help="Caminho para o arquivo best.pt do modelo YOLOv8-cls."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="resultado_painel.png",
        help="Caminho do arquivo de saída para a imagem com destaque visual."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        help="Limiar de porcentagem de dano para reprovação (padrão: 5.0%%)."
    )

    args = parser.parse_args()

    if not args.image:
        print("[ERRO] Nenhuma imagem de teste encontrada. Especifique com --image.")
        sys.exit(1)

    run_pipeline(
        image_path=args.image,
        model_path=args.model,
        output_path=args.output,
        damage_threshold=args.threshold
    )


if __name__ == "__main__":
    main()
