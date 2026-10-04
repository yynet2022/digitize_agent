"""座標キャリブレーションおよび実数値変換ツールモジュール。

画像上のピクセル座標を基準点（軸の目盛り値）に基づいて実世界ドメインの
物理値・統計値へと線形または対数変換し、CSV 形式で出力します。
列名指定、曲線ラベル付与、複数曲線の一括変換に対応します。
"""

import math
from pathlib import Path
from typing import Any, Literal

import numpy as np
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

    pixel_points: list[list[float]] | None = Field(
        default=None,
        description="List of pixel points [[x, y], ...] for single curve.",
    )
    curves: list[dict[str, Any]] | None = Field(
        default=None,
        description="List of curve dicts each having 'points' list.",
    )
    column_names: list[str] = Field(
        default=["x", "y"],
        min_length=2,
        max_length=2,
        description="Column names for X and Y in CSV.",
    )
    curve_label: str | None = Field(
        default=None,
        description="Optional curve label for single curve.",
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
    resample_x_grid: list[float] | None = Field(
        default=None,
        description="Optional explicit common X grid values for resampling.",
    )
    num_grid_points: int | None = Field(
        default=None,
        ge=2,
        le=5000,
        description="Optional number of equidistant X grid points.",
    )
    output_format: Literal["long", "wide"] = Field(
        default="long",
        description=(
            "Output CSV table layout: 'long' (X, Y, curve) or "
            "'wide' (X, curve1, curve2...)."
        ),
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


def _transform_point_list(
    points: list[list[float]],
    x_calib: AxisCalibration,
    y_calib: AxisCalibration,
) -> tuple[list[float], list[float]]:
    """ピクセル点列を物理値の X/Y リストへ変換する。"""
    x_vals: list[float] = []
    y_vals: list[float] = []
    for pt in points:
        if len(pt) < 2:
            continue
        xp, yp = float(pt[0]), float(pt[1])
        x_vals.append(_convert_pixel_to_value(xp, x_calib))
        y_vals.append(_convert_pixel_to_value(yp, y_calib))
    return x_vals, y_vals


def calibrate_and_convert_coordinates(
    x_calibration: dict[str, Any],
    y_calibration: dict[str, Any],
    output_csv_path: str,
    pixel_points: list[list[float]] | None = None,
    curves: list[dict[str, Any]] | None = None,
    column_names: list[str] = ["x", "y"],
    curve_label: str | None = None,
    resample_x_grid: list[float] | None = None,
    num_grid_points: int | None = None,
    output_format: Literal["long", "wide"] = "long",
) -> dict[str, Any]:
    """ピクセル座標列を実数値にマッピングし CSV ファイルとして出力する。

    単一曲線の点列または複数曲線のリストを受け取り、指定列名および
    曲線ラベル付きの CSV ファイルを生成します。

    Args:
        x_calibration: X軸の基準点およびスケール定義辞書。
        y_calibration: Y軸の基準点およびスケール定義辞書。
        output_csv_path: 出力先 CSV ファイルパス。
        pixel_points: 単一曲線のピクセル座標配列 [[x, y], ...]。
        curves: 複数曲線のリスト（各要素は 'points' を保持）。
        column_names: 出力 CSV の X/Y 列名 (デフォルト ['x', 'y'])。
        curve_label: 単一曲線時のラベル名。
        resample_x_grid: 共通 X 格子点リスト（リサンプリング用）。
        num_grid_points: X 最小〜最大間の等間隔補間点数。
        output_format: CSV 出力形式 ('long' または 'wide')。

    Returns:
        dict[str, Any]: 変換成功ステータス、件数、プレビュー、またはエラー。
    """
    try:
        validated = CalibrateCoordinatesInput(
            pixel_points=pixel_points,
            curves=curves,
            column_names=column_names,
            curve_label=curve_label,
            x_calibration=AxisCalibration(**x_calibration),
            y_calibration=AxisCalibration(**y_calibration),
            output_csv_path=output_csv_path,
            resample_x_grid=resample_x_grid,
            num_grid_points=num_grid_points,
            output_format=output_format,
        )
    except Exception as exc:
        return {"status": "error", "message": f"Validation error: {exc}"}

    if not validated.pixel_points and not validated.curves:
        return {
            "status": "error",
            "message": "Either pixel_points or curves must be provided.",
        }

    try:
        col_x, col_y = validated.column_names
        dfs: list[pd.DataFrame] = []

        if validated.curves:
            for idx, c_dict in enumerate(validated.curves):
                pts = c_dict.get("points", [])
                if not pts:
                    continue
                xs, ys = _transform_point_list(
                    pts,
                    validated.x_calibration,
                    validated.y_calibration,
                )
                c_lbl = (
                    c_dict.get("label")
                    or c_dict.get("name")
                    or c_dict.get("curve_name")
                    or f"curve_{idx}"
                )
                c_df = pd.DataFrame(
                    {col_x: xs, col_y: ys, "curve": [c_lbl] * len(xs)}
                )
                dfs.append(c_df)
        else:
            assert validated.pixel_points is not None
            xs, ys = _transform_point_list(
                validated.pixel_points,
                validated.x_calibration,
                validated.y_calibration,
            )
            data: dict[str, Any] = {col_x: xs, col_y: ys}
            if validated.curve_label:
                data["curve"] = [validated.curve_label] * len(xs)
            dfs.append(pd.DataFrame(data))

        if not dfs or sum(len(d) for d in dfs) == 0:
            return {
                "status": "error",
                "message": "No valid points to convert.",
            }

        need_resample = (
            validated.resample_x_grid is not None
            or validated.num_grid_points is not None
            or validated.output_format == "wide"
        )

        if need_resample:
            if validated.resample_x_grid is not None:
                target_x = np.array(validated.resample_x_grid, dtype=float)
            elif validated.num_grid_points is not None:
                all_x = np.concatenate([df[col_x].values for df in dfs])
                target_x = np.linspace(
                    float(np.min(all_x)),
                    float(np.max(all_x)),
                    validated.num_grid_points,
                )
            else:
                all_x = np.concatenate([df[col_x].values for df in dfs])
                target_x = np.linspace(
                    float(np.min(all_x)), float(np.max(all_x)), 100
                )

            if validated.output_format == "wide":
                wide_data: dict[str, Any] = {col_x: target_x.tolist()}
                for idx, df in enumerate(dfs):
                    lbl = (
                        df["curve"].iloc[0]
                        if "curve" in df.columns
                        else f"curve_{idx}"
                    )
                    df_sorted = df.sort_values(by=col_x).drop_duplicates(
                        subset=[col_x]
                    )
                    xp = df_sorted[col_x].values.astype(float)
                    yp = df_sorted[col_y].values.astype(float)
                    y_interp = np.interp(target_x, xp, yp)
                    wide_data[str(lbl)] = y_interp.tolist()
                combined_df = pd.DataFrame(wide_data)
            else:
                resampled_dfs: list[pd.DataFrame] = []
                for df in dfs:
                    lbl = (
                        df["curve"].iloc[0] if "curve" in df.columns else None
                    )
                    df_sorted = df.sort_values(by=col_x).drop_duplicates(
                        subset=[col_x]
                    )
                    xp = df_sorted[col_x].values.astype(float)
                    yp = df_sorted[col_y].values.astype(float)
                    y_interp = np.interp(target_x, xp, yp)
                    r_dict: dict[str, Any] = {
                        col_x: target_x.tolist(),
                        col_y: y_interp.tolist(),
                    }
                    if lbl:
                        r_dict["curve"] = [lbl] * len(target_x)
                    resampled_dfs.append(pd.DataFrame(r_dict))
                combined_df = pd.concat(resampled_dfs, ignore_index=True)
        else:
            combined_df = (
                pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
            )

        out_path = Path(validated.output_csv_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(out_path, index=False)

        preview = combined_df.head(5).to_dict(orient="records")

        return {
            "status": "success",
            "csv_path": str(out_path),
            "row_count": len(combined_df),
            "data_preview": preview,
        }

    except Exception as exc:
        return {
            "status": "error",
            "message": f"Calibration conversion failed: {exc}",
        }
