"""座標キャリブレーションおよび実数値変換ツールモジュール。

画像上のピクセル座標を基準点（軸の目盛り値）に基づいて実世界ドメインの
物理値・統計値へと線形または対数変換し、CSV 形式で出力します。
"""

import math
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field


class AxisCalibration(BaseModel):
    """単一軸のキャリブレーションパラメータモデル。"""

    pixel_refs: list[float] = Field(
        min_length=2,
        max_length=2,
        description="Reference pixel positions [p1, p2].",
    )
    val_refs: list[float] = Field(
        min_length=2,
        max_length=2,
        description="Corresponding domain values [v1, v2].",
    )
    scale_type: Literal["linear", "log"] = Field(
        default="linear",
        description="Scale type: linear or log.",
    )


class CalibrateCoordinatesInput(BaseModel):
    """calibrate_and_convert_coordinates 関数の入力バリデーションモデル。"""

    pixel_points: list[list[float]] = Field(
        description="List of pixel points [[x, y], ...].",
    )
    x_calibration: AxisCalibration = Field(
        description="Calibration definition for X axis.",
    )
    y_calibration: AxisCalibration = Field(
        description="Calibration definition for Y axis.",
    )
    output_csv_path: str = Field(
        description="Output CSV file path.",
    )


def _convert_pixel_to_value(
    pixel: float,
    calib: AxisCalibration,
) -> float:
    """単一ピクセル値を実数値へ変換する。

    Args:
        pixel: 変換対象のピクセル値。
        calib: 軸キャリブレーション設定。

    Returns:
        float: 変換された実数値。

    Raises:
        ValueError: 基準点ピクセルが同一、
            または対数スケール基準値が不正な場合。
    """
    p1, p2 = calib.pixel_refs
    v1, v2 = calib.val_refs

    if p1 == p2:
        raise ValueError(
            f"Reference pixel values must be distinct: p1={p1}, p2={p2}"
        )

    t = (pixel - p1) / (p2 - p1)

    if calib.scale_type == "log":
        if v1 <= 0 or v2 <= 0:
            raise ValueError(
                f"Log scale values must be strictly positive: v1={v1}, v2={v2}"
            )
        log_v1 = math.log10(v1)
        log_v2 = math.log10(v2)
        log_v = log_v1 + t * (log_v2 - log_v1)
        return float(10.0**log_v)

    return float(v1 + t * (v2 - v1))


def calibrate_and_convert_coordinates(
    pixel_points: list[list[float]],
    x_calibration: dict[str, Any],
    y_calibration: dict[str, Any],
    output_csv_path: str,
) -> dict[str, Any]:
    """ピクセル座標列を実数値にマッピングし CSV ファイルとして出力する。

    Args:
        pixel_points: ピクセル座標配列 [[x, y], ...]。
        x_calibration: X軸の基準点およびスケール定義辞書。
        y_calibration: Y軸の基準点およびスケール定義辞書。
        output_csv_path: 出力先 CSV ファイルパス。

    Returns:
        dict[str, Any]: 変換成功ステータス、件数、プレビュー、またはエラー。
    """
    try:
        validated = CalibrateCoordinatesInput(
            pixel_points=pixel_points,
            x_calibration=AxisCalibration(**x_calibration),
            y_calibration=AxisCalibration(**y_calibration),
            output_csv_path=output_csv_path,
        )
    except Exception as exc:
        return {"status": "error", "message": f"Validation error: {exc}"}

    if not validated.pixel_points:
        return {
            "status": "error",
            "message": "Input pixel_points list is empty.",
        }

    try:
        x_vals: list[float] = []
        y_vals: list[float] = []

        for pt in validated.pixel_points:
            if len(pt) < 2:
                continue
            x_p, y_p = float(pt[0]), float(pt[1])
            x_val = _convert_pixel_to_value(x_p, validated.x_calibration)
            y_val = _convert_pixel_to_value(y_p, validated.y_calibration)
            x_vals.append(x_val)
            y_vals.append(y_val)

        df = pd.DataFrame({"x": x_vals, "y": y_vals})

        out_path = Path(validated.output_csv_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)

        preview = df.head(5).to_dict(orient="records")

        return {
            "status": "success",
            "csv_path": str(out_path),
            "row_count": len(df),
            "data_preview": preview,
        }

    except Exception as exc:
        return {
            "status": "error",
            "message": f"Calibration conversion failed: {exc}",
        }
