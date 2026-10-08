"""検証用オーバーレイ画像描画ツールモジュール。

デジタイズされた CSV データを元画像の座標系へ逆変換して重ね合わせ描画
（オーバーレイ）を行い、適合度スコアを算出します。
複数曲線の一括描画および自動色分けに対応します。
"""

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from digitize_agent.tools.calibration import AxisCalibration

DEFAULT_PALETTE: list[tuple[int, int, int]] = [
    (255, 255, 0),  # シアン (BGR)
    (255, 0, 255),  # マゼンタ
    (0, 255, 255),  # イエロー
    (0, 255, 0),  # グリーン
    (0, 165, 255),  # オレンジ
    (255, 128, 0),  # ライトブルー
]


class RenderVerificationOverlayInput(BaseModel):
    """render_verification_overlay 関数の入力バリデーションモデル。"""

    original_image_path: str = Field(
        description="Path to the cropped original plot image."
    )
    csv_path: str = Field(
        description="Path to the digitized CSV file to verify."
    )
    x_calibration: AxisCalibration = Field(
        description=(
            "X axis calibration parameters (from auto_calibrate_axes)."
        )
    )
    y_calibration: AxisCalibration = Field(
        description=(
            "Y axis calibration parameters (from auto_calibrate_axes)."
        )
    )
    output_overlay_path: str = Field(
        description="Target output overlay image file path (PNG recommended)."
    )
    curve_column: str | None = Field(
        default="curve",
        description=(
            "Optional column name for curve grouping in long format CSV "
            "(default: 'curve')."
        ),
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
                f"Log scale values must be positive: v1={v1}, v2={v2}"
            )
        log_v1 = math.log10(v1)
        log_v2 = math.log10(v2)
        log_val = math.log10(val)
        if log_v1 == log_v2:
            raise ValueError("Log references cannot be equal.")
        t = (log_val - log_v1) / (log_v2 - log_v1)
        return float(p1 + t * (p2 - p1))

    t = (val - v1) / (v2 - v1)
    return float(p1 + t * (p2 - p1))


def _calculate_alignment_metric(
    image: np.ndarray,
    pixel_points: list[tuple[int, int]],
    window_size: int = 3,
) -> float:
    """逆変換ピクセル周辺のエッジ画素存在率からアライメント適合度を算出する。

    Args:
        image: 元画像 (BGR)。
        pixel_points: 重ね合わせるピクセル座標リスト。
        window_size: 探索ウィンドウの半径ピクセル。

    Returns:
        float: 0.0 から 1.0 の適合度指標（一致率）。
    """
    if not pixel_points:
        return 0.0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)

    h, w = gray.shape
    matched_count = 0
    valid_count = 0

    for px, py in pixel_points:
        if not (0 <= px < w and 0 <= py < h):
            continue
        valid_count += 1
        x_start = max(0, px - window_size)
        x_end = min(w, px + window_size + 1)
        y_start = max(0, py - window_size)
        y_end = min(h, py + window_size + 1)

        patch = edges[y_start:y_end, x_start:x_end]
        if np.any(patch > 0):
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
    curve_column: str | None = "curve",
) -> dict[str, Any]:
    """元画像とデジタイズデータを半透明重ね合わせして検証画像を生成する。

    デジタイズされた CSV データを元画像のピクセル座標系へ逆変換し、
    元画像上に鮮やかな色分けパレットで半透明オーバーレイ描画します。
    複数曲線の一括描画（long 形式・wide 形式双方に対応）をサポートし、
    エッジ検出画像との重なり一致率（alignment_metric: 0.0〜1.0）を算出します。
    LLM が抽出精度を自己検証（Visual Feedback）するための必須ツールです。

    推奨ワークフロー:
        1. calibrate_and_convert_coordinates で CSV を生成した直後に実行。
        2. output_overlay_path の画像を確認し、曲線が元画像と一致しているか
           検証する。
        3. alignment_metric が十分高いか（通常 0.6〜0.9 以上）を確認する。

    Args:
        original_image_path: クロップ元プロット画像のファイルパス。
        csv_path: デジタイズ結果 CSV ファイルパス。
        x_calibration: X軸のキャリブレーション辞書
            (auto_calibrate_axes の出力)。
        y_calibration: Y軸のキャリブレーション辞書
            (auto_calibrate_axes の出力)。
        output_overlay_path: 検証オーバーレイ画像の保存先ファイルパス。
        curve_column: long形式 CSV 時に曲線識別に使用する列名 (標準: 'curve')。

    Returns:
        dict[str, Any]:
            - status: "success" または "error"。
            - output_path: 保存された検証画像パス。
            - alignment_metric: エッジ重なり適合度スコア (0.0〜1.0)。
            - points_overlaid: オーバーレイ描画された総点数。
            - curves_count: 描画された曲線数。
    """
    try:
        validated = RenderVerificationOverlayInput(
            original_image_path=original_image_path,
            csv_path=csv_path,
            x_calibration=AxisCalibration(**x_calibration),
            y_calibration=AxisCalibration(**y_calibration),
            output_overlay_path=output_overlay_path,
            curve_column=curve_column,
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
                f"Failed to decode image: {validated.original_image_path}"
            ),
        }

    try:
        df = pd.read_csv(c_path)
        if len(df) == 0:
            return {
                "status": "error",
                "message": f"CSV file is empty: {validated.csv_path}",
            }

        # x, y カラムの特定（'x', 'y' または先頭2列）
        non_curve_cols = [c for c in df.columns if c != validated.curve_column]
        col_x = "x" if "x" in df.columns else non_curve_cols[0]
        col_y = "y" if "y" in df.columns else non_curve_cols[1]

        # 曲線グループの分割
        grouped_curves: list[list[tuple[int, int]]] = []
        has_group = (
            validated.curve_column is not None
            and validated.curve_column in df.columns
        )

        if has_group:
            for _, group_df in df.groupby(validated.curve_column, sort=False):
                group_pts: list[tuple[int, int]] = []
                for _, row in group_df.iterrows():
                    vx = float(row[col_x])
                    vy = float(row[col_y])
                    px = _convert_value_to_pixel(vx, validated.x_calibration)
                    py = _convert_value_to_pixel(vy, validated.y_calibration)
                    group_pts.append((int(round(px)), int(round(py))))
                grouped_curves.append(group_pts)
        else:
            single_pts: list[tuple[int, int]] = []
            for _, row in df.iterrows():
                vx = float(row[col_x])
                vy = float(row[col_y])
                px = _convert_value_to_pixel(vx, validated.x_calibration)
                py = _convert_value_to_pixel(vy, validated.y_calibration)
                single_pts.append((int(round(px)), int(round(py))))
            grouped_curves.append(single_pts)

        overlay = image.copy()
        all_pts: list[tuple[int, int]] = []

        for g_idx, pts in enumerate(grouped_curves):
            all_pts.extend(pts)
            if not pts:
                continue
            color_line = DEFAULT_PALETTE[g_idx % len(DEFAULT_PALETTE)]
            color_dot = (
                (255 - color_line[0], 255 - color_line[1], 255 - color_line[2])
                if color_line != (255, 255, 255)
                else (0, 0, 255)
            )

            if len(pts) >= 2:
                pts_array = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(
                    overlay,
                    [pts_array],
                    isClosed=False,
                    color=color_line,
                    thickness=2,
                )

            for px, py in pts:
                cv2.circle(
                    overlay, (px, py), radius=3, color=color_dot, thickness=-1
                )

        # 透過合成 (元画像 40% + オーバーレイ 60%)
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

        metric = _calculate_alignment_metric(image, all_pts)

        return {
            "status": "success",
            "verification_image_path": str(out_path),
            "alignment_metric": metric,
            "curves_rendered": len(grouped_curves),
        }

    except Exception as exc:
        return {
            "status": "error",
            "message": f"Verification overlay failed: {exc}",
        }
