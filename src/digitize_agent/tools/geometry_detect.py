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
