"""特定色によるプロットピクセル抽出ツールモジュール。

HSV 色空間の閾値処理により、連続線プロットまたは散布図マーカーの
ピクセル座標を抽出します。
"""

from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from pydantic import BaseModel, Field


class ExtractPlotPixelsInput(BaseModel):
    """extract_plot_pixels_by_color 関数の入力バリデーションモデル。"""

    image_path: str = Field(description="Path to the plot image file.")
    hsv_lower: list[int] = Field(
        min_length=3,
        max_length=3,
        description="HSV lower bound [H (0-179), S (0-255), V (0-255)].",
    )
    hsv_upper: list[int] = Field(
        min_length=3,
        max_length=3,
        description="HSV upper bound [H (0-179), S (0-255), V (0-255)].",
    )
    extract_mode: Literal["continuous_line", "scatter_centroids"] = Field(
        default="continuous_line",
        description="Extraction mode: continuous_line or scatter_centroids.",
    )


def extract_plot_pixels_by_color(
    image_path: str,
    hsv_lower: list[int],
    hsv_upper: list[int],
    extract_mode: str = "continuous_line",
) -> dict[str, Any]:
    """HSV色閾値によりプロットのピクセル座標群を抽出する。

    Args:
        image_path: 対象画像のファイルパス。
        hsv_lower: HSV 下限値リスト [H, S, V]。
        hsv_upper: HSV 上限値リスト [H, S, V]。
        extract_mode: 抽出モード
            ("continuous_line" または "scatter_centroids")。

    Returns:
        dict[str, Any]: 抽出点数およびピクセル座標配列、またはエラー。
    """
    try:
        validated = ExtractPlotPixelsInput(
            image_path=image_path,
            hsv_lower=hsv_lower,
            hsv_upper=hsv_upper,
            extract_mode=extract_mode,  # type: ignore[arg-type]
        )
    except Exception as exc:
        return {"status": "error", "message": f"Validation error: {exc}"}

    img_path = Path(validated.image_path)
    if not img_path.is_file():
        return {
            "status": "error",
            "message": f"Image file not found: {validated.image_path}",
        }

    image = cv2.imread(str(img_path))
    if image is None:
        return {
            "status": "error",
            "message": f"Failed to load image: {validated.image_path}",
        }

    try:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_bound = np.array(validated.hsv_lower, dtype=np.uint8)
        upper_bound = np.array(validated.hsv_upper, dtype=np.uint8)

        # HSV 範囲マスク生成
        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        # モルフォロジー開閉演算で微細ノイズを除去（中心対称な 3x3 カーネル）
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        pixel_points: list[list[int]] = []

        if validated.extract_mode == "continuous_line":
            # X座標ごとにグループ化し、Y座標の中央値を算出
            y_indices, x_indices = np.where(mask > 0)
            if len(x_indices) > 0:
                unique_xs = np.unique(x_indices)
                for x in unique_xs:
                    ys = y_indices[x_indices == x]
                    y_median = int(np.median(ys))
                    pixel_points.append([int(x), y_median])
                pixel_points.sort(key=lambda pt: pt[0])
        else:
            # 輪郭検出による散布図マーカー重心の算出
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for cnt in contours:
                if cv2.contourArea(cnt) < 1.0:
                    continue
                mom = cv2.moments(cnt)
                if mom["m00"] != 0:
                    cx = int(round(mom["m10"] / mom["m00"]))
                    cy = int(round(mom["m01"] / mom["m00"]))
                else:
                    pt = cnt[0][0]
                    cx, cy = int(pt[0]), int(pt[1])
                pixel_points.append([cx, cy])
            # X座標順にソート
            pixel_points.sort(key=lambda pt: (pt[0], pt[1]))

        return {
            "point_count": len(pixel_points),
            "pixel_points": pixel_points,
        }

    except Exception as exc:
        return {
            "status": "error",
            "message": f"Pixel extraction failed: {exc}",
        }
