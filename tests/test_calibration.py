"""座標キャリブレーションおよび数値変換ツールの単体テストモジュール。

calibrate_and_convert_coordinates の線形・対数変換精度、列名指定、
複数曲線の統合出力、およびエラーハンドリングを検証します。
"""

from pathlib import Path

import pandas as pd

from digitize_agent.tools.calibration import calibrate_and_convert_coordinates


def test_calibrate_linear(temp_dir: Path) -> None:
    """線形スケールにおける座標変換および CSV 出力を検証する。"""
    out_csv = str(temp_dir / "linear.csv")

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


def test_calibrate_custom_columns_and_label(temp_dir: Path) -> None:
    """列名指定および曲線ラベル付与が正常に反映されることを検証する。"""
    out_csv = str(temp_dir / "custom.csv")
    x_calib = {"pixel_refs": [0.0, 10.0], "val_refs": [0.0, 1.0]}
    y_calib = {"pixel_refs": [0.0, 10.0], "val_refs": [0.0, 1.0]}

    res = calibrate_and_convert_coordinates(
        pixel_points=[[5.0, 5.0]],
        column_names=["V_DS_V", "I_DS_mA"],
        curve_label="V_GS_large",
        x_calibration=x_calib,
        y_calibration=y_calib,
        output_csv_path=out_csv,
    )

    assert res["status"] == "success"
    df = pd.read_csv(out_csv)
    assert "V_DS_V" in df.columns
    assert "I_DS_mA" in df.columns
    assert "curve" in df.columns
    assert df["curve"][0] == "V_GS_large"


def test_calibrate_multiple_curves(temp_dir: Path) -> None:
    """複数曲線のリストが一括変換され単一 CSV に結合されることを検証。"""
    out_csv = str(temp_dir / "multi.csv")
    x_calib = {"pixel_refs": [0.0, 10.0], "val_refs": [0.0, 1.0]}
    y_calib = {"pixel_refs": [0.0, 10.0], "val_refs": [0.0, 1.0]}

    curves_input = [
        {"points": [[0.0, 0.0], [10.0, 10.0]], "label": "curve_a"},
        {"points": [[5.0, 5.0]], "label": "curve_b"},
    ]

    res = calibrate_and_convert_coordinates(
        curves=curves_input,
        column_names=["x_val", "y_val"],
        x_calibration=x_calib,
        y_calibration=y_calib,
        output_csv_path=out_csv,
    )

    assert res["status"] == "success"
    assert res["row_count"] == 3
    df = pd.read_csv(out_csv)
    assert len(df) == 3
    assert list(df["curve"]) == ["curve_a", "curve_a", "curve_b"]


def test_calibrate_name_fallback(temp_dir: Path) -> None:
    """curves で 'name' や 'curve_name' もラベルとして受理されることを検証。"""
    out_csv = str(temp_dir / "name_fallback.csv")
    x_calib = {"pixel_refs": [0.0, 10.0], "val_refs": [0.0, 1.0]}
    y_calib = {"pixel_refs": [0.0, 10.0], "val_refs": [0.0, 1.0]}

    curves_input = [
        {"points": [[0.0, 0.0]], "name": "alpha"},
        {"points": [[5.0, 5.0]], "curve_name": "beta"},
    ]

    res = calibrate_and_convert_coordinates(
        curves=curves_input,
        x_calibration=x_calib,
        y_calibration=y_calib,
        output_csv_path=out_csv,
    )

    assert res["status"] == "success"
    df = pd.read_csv(out_csv)
    assert list(df["curve"]) == ["alpha", "beta"]


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
    assert "Either pixel_points or curves" in res["message"]


def test_calibrate_zero_division(temp_dir: Path) -> None:
    """基準ピクセルが同一でゼロ除算が発生する場合にエラーを返すことを検証。"""
    out_csv = str(temp_dir / "zero.csv")
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


def test_calibrate_wide_format(temp_dir: Path) -> None:
    """output_format='wide' で横持ち形式の CSV が出力されることを検証。"""
    out_csv = str(temp_dir / "wide.csv")
    x_calib = {"pixel_refs": [0.0, 100.0], "val_refs": [0.0, 10.0]}
    y_calib = {"pixel_refs": [100.0, 0.0], "val_refs": [0.0, 50.0]}

    curves = [
        {"name": "c1", "points": [[0.0, 100.0], [50.0, 50.0]]},
        {"name": "c2", "points": [[0.0, 50.0], [50.0, 0.0]]},
    ]

    res = calibrate_and_convert_coordinates(
        curves=curves,
        x_calibration=x_calib,
        y_calibration=y_calib,
        output_csv_path=out_csv,
        output_format="wide",
    )
    assert res["status"] == "success"
    df = pd.read_csv(out_csv)
    assert "x" in df.columns
    assert "c1" in df.columns
    assert "c2" in df.columns
    assert len(df) == 100


def test_calibrate_resample_num_grid_points(temp_dir: Path) -> None:
    """num_grid_points で指定点数の共通 X 格子に補間されることを検証。"""
    out_csv = str(temp_dir / "resampled_pts.csv")
    x_calib = {"pixel_refs": [0.0, 100.0], "val_refs": [0.0, 10.0]}
    y_calib = {"pixel_refs": [100.0, 0.0], "val_refs": [0.0, 50.0]}

    curves = [
        {"name": "c1", "points": [[0.0, 100.0], [100.0, 0.0]]},
        {"name": "c2", "points": [[20.0, 80.0], [80.0, 20.0]]},
    ]

    res = calibrate_and_convert_coordinates(
        curves=curves,
        x_calibration=x_calib,
        y_calibration=y_calib,
        output_csv_path=out_csv,
        output_format="wide",
        num_grid_points=11,
    )
    assert res["status"] == "success"
    df = pd.read_csv(out_csv)
    assert len(df) == 11
    assert abs(df["x"].iloc[0] - 0.0) < 1e-4
    assert abs(df["x"].iloc[-1] - 10.0) < 1e-4


def test_calibrate_resample_explicit_grid(temp_dir: Path) -> None:
    """resample_x_grid リストで明示的な X 格子に補間されることを検証。"""
    out_csv = str(temp_dir / "resampled_grid.csv")
    x_calib = {"pixel_refs": [0.0, 100.0], "val_refs": [0.0, 10.0]}
    y_calib = {"pixel_refs": [100.0, 0.0], "val_refs": [0.0, 50.0]}

    curves = [
        {"name": "c1", "points": [[0.0, 100.0], [100.0, 0.0]]},
        {"name": "c2", "points": [[0.0, 50.0], [100.0, 50.0]]},
    ]

    custom_x = [0.0, 2.5, 5.0, 7.5, 10.0]
    res = calibrate_and_convert_coordinates(
        curves=curves,
        x_calibration=x_calib,
        y_calibration=y_calib,
        output_csv_path=out_csv,
        output_format="wide",
        resample_x_grid=custom_x,
    )
    assert res["status"] == "success"
    df = pd.read_csv(out_csv)
    assert len(df) == len(custom_x)
    assert list(df["x"]) == custom_x
