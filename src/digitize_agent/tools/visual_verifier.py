"""検証用オーバーレイ画像描画ツールモジュール。

デジタイズされた CSV データを元画像の座標系へ逆変換して重ね合わせ描画
（オーバーレイ）を行い、適合度スコアを算出します。
"""

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from digitize_agent.tools.calibration import AxisCalibration


class RenderVerificationOverlayInput(BaseModel):
    """render_verification_overlay 関数の入力バリデーションモデル。"""

    original_image_path: str = Field(
        description="Path to the cropped original plot image."
    )
    csv_path: str = Field(description="Path to the digitized CSV file.")
    x_calibration: AxisCalibration = Field(
        description="X axis calibration parameters."
    )
    y_calibration: AxisCalibration = Field(
        description="Y axis calibration parameters."
    )
    output_overlay_path: str = Field(
        description="Output verification image path."
    )


def _convert_value_to_pixel(
    val: float,
    calib: AxisCalibration,
) -> float:
    """実数値をピクセル値へと逆変換する。

    Args:
        val: 実ドメイン値。
        calib: 軸キャリブレーション設定。

    Returns:
        float: 逆変換されたピクセル位置。

    Raises:
        ValueError: 基準値が不正または一致している場合。
    """
    p1, p2 = calib.pixel_refs
    v1, v2 = calib.val_refs

    if v1 == v2:
        raise ValueError(
            f"Reference values must be distinct: v1={v1}, v2={v2}"
        )

    if calib.scale_type == "log":
        if v1 <= 0 or v2 <= 0 or val <= 0:
            raise ValueError(
                f"Log scale values must be strictly positive: val={val}"
            )
        log_v1 = math.log10(v1)
        log_v2 = math.log10(v2)
        log_val = math.log10(val)
        t = (log_val - log_v1) / (log_v2 - log_v1)
    else:
        t = (val - v1) / (v2 - v1)

    return float(p1 + t * (p2 - p1))


def _calculate_alignment_metric(
    image: np.ndarray,
    pixel_points: list[tuple[int, int]],
) -> float:
    """元画像のエッジと再構築ピクセル点群との重なり度合いを計算する。

    Args:
        image: 元画像 (BGR)。
        pixel_points: 再構築されたピクセル座標群 [(px, py), ...]。

    Returns:
        float: 0.0 から 1.0 までの適合度指標スコア。
    """
    if not pixel_points:
        return 0.0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    # エッジの近傍（±2ピクセル）を許容するため膨張処理
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated_edges = cv2.dilate(edges, kernel)

    height, width = image.shape[:2]
    matched_count = 0
    valid_count = 0

    for px, py in pixel_points:
        if 0 <= px < width and 0 <= py < height:
            valid_count += 1
            if dilated_edges[py, px] > 0:
                matched_count += 1

    if valid_count == 0:
        return 0.0

    return float(round(matched_count / valid_count, 4))


def render_verification_overlay(
    original_image_path: str,
    csv_path: str,
    x_calibration: dict[str, Any],
    y_calibration: dict[str, Any],
    output_overlay_path: str,
) -> dict[str, Any]:
    """元画像とデジタイズデータを半透明重ね合わせして検証画像を生成する。

    Args:
        original_image_path: 元画像のファイルパス。
        csv_path: デジタイズ結果 CSV ファイルパス。
        x_calibration: X軸のキャリブレーション辞書。
        y_calibration: Y軸のキャリブレーション辞書。
        output_overlay_path: 検証画像の保存先ファイルパス。

    Returns:
        dict[str, Any]: 生成画像パスおよび適合度指標スコア、またはエラー。
    """
    try:
        validated = RenderVerificationOverlayInput(
            original_image_path=original_image_path,
            csv_path=csv_path,
            x_calibration=AxisCalibration(**x_calibration),
            y_calibration=AxisCalibration(**y_calibration),
            output_overlay_path=output_overlay_path,
        )
    except Exception as exc:
        return {"status": "error", "message": f"Validation error: {exc}"}

    img_path = Path(validated.original_image_path)
    if not img_path.is_file():
        return {
            "status": "error",
            "message": (
                f"Original image not found: {validated.original_image_path}"
            ),
        }

    c_path = Path(validated.csv_path)
    if not c_path.is_file():
        return {
            "status": "error",
            "message": f"CSV file not found: {validated.csv_path}",
        }

    image = cv2.imread(str(img_path))
    if image is None:
        return {
            "status": "error",
            "message": (
                f"Failed to load image: {validated.original_image_path}"
            ),
        }

    try:
        df = pd.read_csv(c_path)
        if len(df) == 0:
            return {
                "status": "error",
                "message": f"CSV file is empty: {validated.csv_path}",
            }

        # x, y カラムの存在確認（なければ先頭2列を採用）
        col_x = "x" if "x" in df.columns else df.columns[0]
        col_y = "y" if "y" in df.columns else df.columns[1]

        pixel_pts: list[tuple[int, int]] = []
        for _, row in df.iterrows():
            vx = float(row[col_x])
            vy = float(row[col_y])
            px = _convert_value_to_pixel(vx, validated.x_calibration)
            py = _convert_value_to_pixel(vy, validated.y_calibration)
            pixel_pts.append((int(round(px)), int(round(py))))

        # オーバーレイ描画用キャンバス
        overlay = image.copy()
        if len(pixel_pts) >= 2:
            pts_array = np.array(pixel_pts, dtype=np.int32).reshape((-1, 1, 2))
            # 蛍光シアン (BGR: 255, 255, 0) で線描画
            cv2.polylines(
                overlay,
                [pts_array],
                isClosed=False,
                color=(255, 255, 0),
                thickness=2,
            )

        # 蛍光マゼンタ (BGR: 255, 0, 255) でデータ点描画
        for px, py in pixel_pts:
            cv2.circle(
                overlay, (px, py), radius=3, color=(255, 0, 255), thickness=-1
            )

        # 透過合成 (元画像 50% + オーバーレイ 50%)
        blended = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)

        out_path = Path(validated.output_overlay_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ext = out_path.suffix if out_path.suffix else ".png"
        success, buffer = cv2.imencode(ext, blended)
        if not success:
            return {
                "status": "error",
                "message": f"Failed to encode overlay image with ext {ext}.",
            }
        with open(out_path, "wb") as f_out:
            f_out.write(buffer)

        metric = _calculate_alignment_metric(image, pixel_pts)

        return {
            "verification_image_path": str(out_path),
            "alignment_metric": metric,
        }

    except Exception as exc:
        return {
            "status": "error",
            "message": f"Verification overlay failed: {exc}",
        }
