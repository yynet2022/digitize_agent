"""幾何学的要素（座標軸および目盛り線）検出ツールモジュール。

グラフ画像から水平 X 軸・垂直 Y 軸の位置、および目盛り線（Tick marks）の
ピクセル座標を検出します。
"""

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pydantic import BaseModel, Field
from scipy.signal import find_peaks


class DetectAxesAndTicksInput(BaseModel):
    """detect_axes_and_ticks 関数の入力バリデーションモデル。"""

    image_path: str = Field(description="Path to the plot image file.")
    min_line_length_ratio: float = Field(
        default=0.3,
        ge=0.05,
        le=1.0,
        description="Minimum line length ratio relative to image size.",
    )


def _find_axis_line(
    projection: np.ndarray,
    min_length: int,
    prefer_lower_or_left: bool = True,
) -> int:
    """プロジェクションプロファイルから軸線のピクセル位置を特定する。

    Args:
        projection: 1次元プロジェクション配列。
        min_length: 軸線と見なす最小強度（長さ）。
        prefer_lower_or_left: 下端または左端側の軸を優先するか。

    Returns:
        int: 特定された軸のピクセルインデックス。
    """
    peaks, properties = find_peaks(projection, height=min_length, distance=10)
    if len(peaks) == 0:
        return int(np.argmax(projection))

    if prefer_lower_or_left:
        # X軸なら下端側、Y軸なら左端側の強いピークを選択
        return int(peaks[-1] if prefer_lower_or_left else peaks[0])

    return int(peaks[np.argmax(properties["peak_heights"])])


def detect_axes_and_ticks(
    image_path: str,
    min_line_length_ratio: float = 0.3,
) -> dict[str, Any]:
    """グラフ画像から座標軸線および目盛り線のピクセル位置を特定する。

    Args:
        image_path: 対象グラフ画像のファイルパス。
        min_line_length_ratio: 軸と見なす最小長比率。

    Returns:
        dict[str, Any]: X軸・Y軸および目盛り候補の座標辞書、またはエラー。
    """
    try:
        validated = DetectAxesAndTicksInput(
            image_path=image_path,
            min_line_length_ratio=min_line_length_ratio,
        )
    except Exception as exc:
        return {"status": "error", "message": f"Validation error: {exc}"}

    img_path = Path(validated.image_path)
    if not img_path.is_file():
        return {
            "status": "error",
            "message": f"Plot image not found: {validated.image_path}",
        }

    image = cv2.imread(str(img_path))
    if image is None:
        return {
            "status": "error",
            "message": f"Failed to load image: {validated.image_path}",
        }

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 二値化（白が前景、黒が背景）
    _, bin_inv = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # 水平線・垂直線のモルフォロジー分離
    min_w = max(10, int(width * validated.min_line_length_ratio))
    min_h = max(10, int(height * validated.min_line_length_ratio))

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_w // 2, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_h // 2))

    h_lines = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, h_kernel)
    v_lines = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, v_kernel)

    # 水平・垂直プロジェクションの算出 (ピクセル数換算)
    h_proj = np.sum(h_lines > 0, axis=1)
    v_proj = np.sum(v_lines > 0, axis=0)

    # 軸線の位置特定 (X軸は通常下部、Y軸は通常左部)
    y_peaks, _ = find_peaks(h_proj, height=min_w * 0.5, distance=15)
    if len(y_peaks) > 0:
        # 下半分の最も下にあるピーク、なければ全体の最下ピーク
        lower_half = [p for p in y_peaks if p >= height * 0.4]
        x_axis_y = int(lower_half[-1] if lower_half else y_peaks[-1])
    else:
        x_axis_y = int(np.argmax(h_proj)) if np.max(h_proj) > 0 else height - 1

    x_peaks, _ = find_peaks(v_proj, height=min_h * 0.5, distance=15)
    if len(x_peaks) > 0:
        # 左半分の最も左にあるピーク、なければ全体の最左ピーク
        left_half = [p for p in x_peaks if p <= width * 0.6]
        y_axis_x = int(left_half[0] if left_half else x_peaks[0])
    else:
        y_axis_x = int(np.argmax(v_proj)) if np.max(v_proj) > 0 else 0

    # 軸線の有効範囲 (x_range, y_range) の特定
    x_row = bin_inv[max(0, x_axis_y - 2) : min(height, x_axis_y + 3), :]
    x_active = np.where(np.any(x_row > 0, axis=0))[0]
    if len(x_active) > 0:
        x_range = [int(x_active[0]), int(x_active[-1])]
    else:
        x_range = [y_axis_x, width - 1]

    y_col = bin_inv[:, max(0, y_axis_x - 2) : min(width, y_axis_x + 3)]
    y_active = np.where(np.any(y_col > 0, axis=1))[0]
    if len(y_active) > 0:
        y_range = [int(y_active[0]), int(y_active[-1])]
    else:
        y_range = [0, x_axis_y]

    # 目盛り候補の検出
    # X目盛り: X軸近傍における垂直エッジ・線の突出ピーク
    band_y1 = max(0, x_axis_y - 12)
    band_y2 = min(height, x_axis_y + 13)
    x_tick_band = bin_inv[band_y1:band_y2, :]
    x_tick_proj = np.sum(x_tick_band > 0, axis=0)

    # 軸線自身の連続平坦成分を減算するため局所差分を考慮
    x_tick_peaks, _ = find_peaks(
        x_tick_proj, height=3, distance=10, prominence=2
    )
    x_ticks = [int(p) for p in x_tick_peaks if x_range[0] <= p <= x_range[1]]

    # Y目盛り: Y軸近傍における水平エッジ・線の突出ピーク
    band_x1 = max(0, y_axis_x - 12)
    band_x2 = min(width, y_axis_x + 13)
    y_tick_band = bin_inv[:, band_x1:band_x2]
    y_tick_proj = np.sum(y_tick_band > 0, axis=1)

    y_tick_peaks, _ = find_peaks(
        y_tick_proj, height=3, distance=10, prominence=2
    )
    y_ticks = [int(p) for p in y_tick_peaks if y_range[0] <= p <= y_range[1]]

    return {
        "x_axis": {"y_pixel": x_axis_y, "x_range": x_range},
        "y_axis": {"x_pixel": y_axis_x, "y_range": y_range},
        "x_tick_candidates": sorted(x_ticks),
        "y_tick_candidates": sorted(y_ticks),
    }


class DetectLegendRegionInput(BaseModel):
    """detect_legend_region 関数の入力バリデーションモデル。"""

    image_path: str = Field(description="Path to the plot image file.")
    plot_bbox: list[int] | None = Field(
        default=None,
        min_length=4,
        max_length=4,
        description="Optional [x_min, y_min, x_max, y_max] of plot area.",
    )


def detect_legend_region(
    image_path: str,
    plot_bbox: list[int] | None = None,
) -> dict[str, Any]:
    """プロット画像から凡例（Legend）領域の矩形を検出する。

    Args:
        image_path: 対象グラフ画像のファイルパス。
        plot_bbox: プロット枠の矩形 [x_min, y_min, x_max, y_max]。

    Returns:
        dict[str, Any]: 検出された凡例矩形 [x_min, y_min, x_max, y_max]
            および信頼度スコア。
    """
    try:
        validated = DetectLegendRegionInput(
            image_path=image_path,
            plot_bbox=plot_bbox,
        )
    except Exception as exc:
        return {"status": "error", "message": f"Validation error: {exc}"}

    img_path = Path(validated.image_path)
    if not img_path.is_file():
        return {
            "status": "error",
            "message": f"Plot image not found: {validated.image_path}",
        }

    image = cv2.imread(str(img_path))
    if image is None:
        return {
            "status": "error",
            "message": f"Failed to load image: {validated.image_path}",
        }

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # プロット枠の決定
    if validated.plot_bbox is not None:
        px0, py0, px1, py1 = validated.plot_bbox
    else:
        # プロット領域の概算
        axes_info = detect_axes_and_ticks(validated.image_path)
        if axes_info.get("status") != "error":
            px0 = axes_info["y_axis"]["x_pixel"]
            py1 = axes_info["x_axis"]["y_pixel"]
            px1 = axes_info["x_axis"]["x_range"][1]
            py0 = axes_info["y_axis"]["y_range"][0]
        else:
            px0 = int(width * 0.1)
            py0 = int(height * 0.1)
            px1 = int(width * 0.9)
            py1 = int(height * 0.9)

    plot_w = max(1, px1 - px0)
    plot_h = max(1, py1 - py0)
    plot_area = plot_w * plot_h

    # エッジ・輪郭による凡例矩形枠の検出
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(
        edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    candidate_boxes: list[dict[str, Any]] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 800 or area > plot_area * 0.5:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        # 4〜8頂点の凸多角形（矩形枠）を検査
        if 4 <= len(approx) <= 8:
            x, y, w, h = cv2.boundingRect(cnt)
            # プロット枠の内側にあるか確認
            if (
                x >= px0 - 5
                and y >= py0 - 5
                and (x + w) <= px1 + 5
                and (y + h) <= py1 + 5
            ):
                aspect = w / float(h)
                if 1.0 <= aspect <= 6.0 and w >= 50 and h >= 30:
                    score = min(1.0, area / (plot_area * 0.15))
                    candidate_boxes.append(
                        {
                            "bbox": [int(x), int(y), int(x + w), int(y + h)],
                            "area": int(area),
                            "confidence": round(float(score), 3),
                        }
                    )

    if candidate_boxes:
        # 最も確度の高いボックス（面積とアスペクト比でソート）
        candidate_boxes.sort(key=lambda b: b["area"], reverse=True)
        best = candidate_boxes[0]
        return {
            "status": "success",
            "legend_detected": True,
            "legend_bbox": best["bbox"],
            "confidence": best["confidence"],
            "candidates_count": len(candidate_boxes),
        }

    # 枠線がない場合：プロット枠内のテキスト密度・色シンボル分布から探索
    # モルフォロジー演算でテキスト塊を連結
    _, bin_inv = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    t_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    text_blobs = cv2.morphologyEx(bin_inv, cv2.MORPH_CLOSE, t_kernel)

    roi = np.zeros_like(text_blobs)
    roi[py0:py1, px0:px1] = text_blobs[py0:py1, px0:px1]

    sub_cnts, _ = cv2.findContours(
        roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    text_rects = []
    for sc in sub_cnts:
        sx, sy, sw, sh = cv2.boundingRect(sc)
        if 20 <= sw <= plot_w * 0.5 and 8 <= sh <= 40:
            text_rects.append([sx, sy, sx + sw, sy + sh])

    # 複数テキストが縦方向に並んでいるクラスタ（凡例リスト）を探索
    if len(text_rects) >= 2:
        # 近接するテキスト矩形群を統合
        tx0 = min(r[0] for r in text_rects)
        ty0 = min(r[1] for r in text_rects)
        tx1 = max(r[2] for r in text_rects)
        ty1 = max(r[3] for r in text_rects)
        # 余白を持たせる
        pad_x = 10
        pad_y = 5
        leg_box = [
            max(px0, tx0 - pad_x),
            max(py0, ty0 - pad_y),
            min(px1, tx1 + pad_x),
            min(py1, ty1 + pad_y),
        ]
        return {
            "status": "success",
            "legend_detected": True,
            "legend_bbox": leg_box,
            "confidence": 0.65,
            "candidates_count": 1,
        }

    return {
        "status": "success",
        "legend_detected": False,
        "legend_bbox": None,
        "confidence": 0.0,
        "candidates_count": 0,
    }
