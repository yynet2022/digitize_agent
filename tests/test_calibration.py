"""座標キャリブレーションおよび数値変換ツールの単体テストモジュール。

calibrate_and_convert_coordinates の線形・対数変換精度、
CSV 出力、およびエラーハンドリングを検証します。
"""

from pathlib import Path

import pandas as pd

from digitize_agent.tools.calibration import calibrate_and_convert_coordinates


def test_calibrate_linear(temp_dir: Path) -> None:
    """線形スケールにおける座標変換および CSV 出力が
    正確であることを検証する。
    """
    out_csv = str(temp_dir / "linear.csv")

    # X: ピクセル 0 -> 値 0.0, ピクセル 100 -> 値 10.0
    # Y: ピクセル 100 (下) -> 値 0.0, ピクセル 0 (上) -> 値 50.0
    x_calib = {
        "pixel_refs": [0.0, 100.0],
        "val_refs": [0.0, 10.0],
        "scale_type": "linear",
    }
    y_calib = {
        "pixel_refs": [100.0, 0.0],
        "val_refs": [0.0, 50.0],
        "scale_type": "linear",
    }

    pts = [[0.0, 100.0], [50.0, 50.0], [100.0, 0.0]]

    res = calibrate_and_convert_coordinates(
        pixel_points=pts,
        x_calibration=x_calib,
        y_calibration=y_calib,
        output_csv_path=out_csv,
    )

    assert res["status"] == "success"
    assert res["row_count"] == 3
    assert len(res["data_preview"]) == 3
    assert Path(out_csv).is_file()

    df = pd.read_csv(out_csv)
    # (0, 100) -> x=0, y=0
    assert abs(df["x"][0] - 0.0) < 1e-4
    assert abs(df["y"][0] - 0.0) < 1e-4
    # (50, 50) -> x=5, y=25
    assert abs(df["x"][1] - 5.0) < 1e-4
    assert abs(df["y"][1] - 25.0) < 1e-4
    # (100, 0) -> x=10, y=50
    assert abs(df["x"][2] - 10.0) < 1e-4
    assert abs(df["y"][2] - 50.0) < 1e-4


def test_calibrate_log_scale(temp_dir: Path) -> None:
    """対数スケールにおける座標変換が正確であることを検証する。"""
    out_csv = str(temp_dir / "log.csv")

    # Y: ピクセル 100 -> 値 1.0 (10^0), ピクセル 0 -> 値 100.0 (10^2)
    x_calib = {
        "pixel_refs": [0.0, 100.0],
        "val_refs": [0.0, 10.0],
        "scale_type": "linear",
    }
    y_calib = {
        "pixel_refs": [100.0, 0.0],
        "val_refs": [1.0, 100.0],
        "scale_type": "log",
    }

    # 中間点 50 は対数中間値 10^1 = 10.0 になるはず
    pts = [[50.0, 50.0]]

    res = calibrate_and_convert_coordinates(
        pixel_points=pts,
        x_calibration=x_calib,
        y_calibration=y_calib,
        output_csv_path=out_csv,
    )

    assert res["status"] == "success"
    df = pd.read_csv(out_csv)
    assert abs(df["y"][0] - 10.0) < 1e-3


def test_calibrate_empty_points(temp_dir: Path) -> None:
    """空の点リストを指定した際にエラーを返すことを検証する。"""
    out_csv = str(temp_dir / "empty.csv")
    x_calib = {"pixel_refs": [0.0, 100.0], "val_refs": [0.0, 10.0]}
    y_calib = {"pixel_refs": [0.0, 100.0], "val_refs": [0.0, 10.0]}

    res = calibrate_and_convert_coordinates(
        pixel_points=[],
        x_calibration=x_calib,
        y_calibration=y_calib,
        output_csv_path=out_csv,
    )

    assert res["status"] == "error"
    assert "empty" in res["message"]


def test_calibrate_zero_division(temp_dir: Path) -> None:
    """基準ピクセルが同一でゼロ除算が発生する場合に安全にエラーを返すことを検証する。"""
    out_csv = str(temp_dir / "zero.csv")
    # p1 == p2
    x_calib = {"pixel_refs": [50.0, 50.0], "val_refs": [0.0, 10.0]}
    y_calib = {"pixel_refs": [0.0, 100.0], "val_refs": [0.0, 10.0]}

    res = calibrate_and_convert_coordinates(
        pixel_points=[[10.0, 20.0]],
        x_calibration=x_calib,
        y_calibration=y_calib,
        output_csv_path=out_csv,
    )

    assert res["status"] == "error"
    assert "distinct" in res["message"]
