"""検証オーバーレイ描画ツールの単体テストモジュール。

render_verification_overlay の再プロット描画、透過合成、
複数曲線の色分けグループ化、適合度指標計算、およびエラー耐性を検証します。
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from digitize_agent.tools.calibration import AxisCalibration
from digitize_agent.tools.visual_verifier import (
    _calculate_alignment_metric,
    _convert_value_to_pixel,
    render_verification_overlay,
)


def test_render_verification_overlay_success(
    sample_plot_image: str, temp_dir: Path
) -> None:
    """元画像と CSV からオーバーレイ画像が正しく生成されることを検証する。"""
    csv_path = str(temp_dir / "data.csv")
    df = pd.DataFrame(
        {
            "x": [0.0, 5.0, 10.0],
            "y": [0.0, 25.0, 50.0],
        }
    )
    df.to_csv(csv_path, index=False)

    out_img = str(temp_dir / "overlay.png")
    x_calib = {"pixel_refs": [40.0, 180.0], "val_refs": [0.0, 10.0]}
    y_calib = {"pixel_refs": [160.0, 20.0], "val_refs": [0.0, 50.0]}

    res = render_verification_overlay(
        original_image_path=sample_plot_image,
        csv_path=csv_path,
        x_calibration=x_calib,
        y_calibration=y_calib,
        output_overlay_path=out_img,
    )

    assert res.get("status") != "error"
    assert "verification_image_path" in res
    assert Path(out_img).is_file()
    assert 0.0 <= res["alignment_metric"] <= 1.0


def test_render_verification_overlay_multiple_curves(
    sample_plot_image: str, temp_dir: Path
) -> None:
    """複数曲線を含む CSV で正しくグループ化描画されることを検証する。"""
    csv_path = str(temp_dir / "multi_curve.csv")
    df = pd.DataFrame(
        {
            "x": [0.0, 5.0, 10.0, 0.0, 5.0],
            "y": [0.0, 20.0, 40.0, 5.0, 25.0],
            "curve": ["c1", "c1", "c1", "c2", "c2"],
        }
    )
    df.to_csv(csv_path, index=False)

    out_img = str(temp_dir / "multi_overlay.png")
    x_calib = {"pixel_refs": [40.0, 180.0], "val_refs": [0.0, 10.0]}
    y_calib = {"pixel_refs": [160.0, 20.0], "val_refs": [0.0, 50.0]}

    res = render_verification_overlay(
        original_image_path=sample_plot_image,
        csv_path=csv_path,
        x_calibration=x_calib,
        y_calibration=y_calib,
        output_overlay_path=out_img,
    )

    assert res.get("status") != "error"
    assert res.get("curves_rendered") == 2
    assert Path(out_img).is_file()


def test_render_verification_file_not_found(temp_dir: Path) -> None:
    """元画像または CSV が存在しない場合に安全にエラーを返すことを検証。"""
    x_calib = {"pixel_refs": [0.0, 100.0], "val_refs": [0.0, 10.0]}
    y_calib = {"pixel_refs": [100.0, 0.0], "val_refs": [0.0, 10.0]}

    res = render_verification_overlay(
        original_image_path=str(temp_dir / "missing.png"),
        csv_path=str(temp_dir / "missing.csv"),
        x_calibration=x_calib,
        y_calibration=y_calib,
        output_overlay_path=str(temp_dir / "out.png"),
    )

    assert res["status"] == "error"
    assert "not found" in res["message"]


def test_render_verification_empty_csv(
    sample_plot_image: str, temp_dir: Path
) -> None:
    """空の CSV ファイルが与えられた場合にエラーを返すことを検証する。"""
    empty_csv = str(temp_dir / "empty.csv")
    pd.DataFrame(columns=["x", "y"]).to_csv(empty_csv, index=False)

    x_calib = {"pixel_refs": [0.0, 100.0], "val_refs": [0.0, 10.0]}
    y_calib = {"pixel_refs": [100.0, 0.0], "val_refs": [0.0, 10.0]}

    res = render_verification_overlay(
        original_image_path=sample_plot_image,
        csv_path=empty_csv,
        x_calibration=x_calib,
        y_calibration=y_calib,
        output_overlay_path=str(temp_dir / "out.png"),
    )

    assert res["status"] == "error"
    assert "empty" in res["message"]


def test_render_verification_overlay_wide_with_nan(
    sample_plot_image: str, temp_dir: Path
) -> None:
    """Wide 形式かつ NaN を含む CSV で正常に描画できることを検証する。"""
    csv_path = str(temp_dir / "wide_nan.csv")
    df = pd.DataFrame(
        {
            "x": [0.0, 5.0, 10.0],
            "curve_a": [10.0, 20.0, float("nan")],
            "curve_b": [float("nan"), 15.0, 30.0],
        }
    )
    df.to_csv(csv_path, index=False)

    out_img = str(temp_dir / "wide_overlay.png")
    x_calib = {"pixel_refs": [40.0, 180.0], "val_refs": [0.0, 10.0]}
    y_calib = {"pixel_refs": [160.0, 20.0], "val_refs": [0.0, 50.0]}

    res = render_verification_overlay(
        original_image_path=sample_plot_image,
        csv_path=csv_path,
        x_calibration=x_calib,
        y_calibration=y_calib,
        output_overlay_path=out_img,
    )

    assert res.get("status") != "error"
    assert res.get("curves_rendered") == 2
    assert Path(out_img).is_file()


def test_convert_value_to_pixel_and_alignment_helpers() -> None:
    """_convert_value_to_pixel と _calculate_alignment_metric の分岐を検証。"""
    # 1. 同一基準値で ValueError
    same_calib = AxisCalibration(pixel_refs=[0.0, 100.0], val_refs=[5.0, 5.0])
    with pytest.raises(ValueError, match="must be distinct"):
        _convert_value_to_pixel(5.0, same_calib)

    # 2. 対数スケール逆変換の正常系と非正値エラー
    log_calib = AxisCalibration(
        pixel_refs=[0.0, 100.0], val_refs=[1.0, 100.0], scale_type="log"
    )
    assert abs(_convert_value_to_pixel(10.0, log_calib) - 50.0) < 1e-4

    with pytest.raises(ValueError, match="must be positive"):
        _convert_value_to_pixel(-1.0, log_calib)

    with (
        patch("math.log10", return_value=1.0),
        pytest.raises(ValueError, match="Log references cannot be equal"),
    ):
        _convert_value_to_pixel(10.0, log_calib)

    # 3. _calculate_alignment_metric の空点列および全点画像範囲外
    dummy_img = np.zeros((20, 20, 3), dtype=np.uint8)
    assert _calculate_alignment_metric(dummy_img, []) == 0.0
    assert _calculate_alignment_metric(dummy_img, [(-5, -5), (50, 50)]) == 0.0


def test_render_verification_overlay_error_and_nan_branches(
    sample_plot_image: str, temp_dir: Path
) -> None:
    """render_verification_overlay の異常系と NaN・空グループ分岐を検証。"""
    x_calib = {"pixel_refs": [40.0, 180.0], "val_refs": [0.0, 10.0]}
    y_calib = {"pixel_refs": [160.0, 20.0], "val_refs": [0.0, 50.0]}
    out_img = str(temp_dir / "branch_overlay.png")

    # 1. バリデーションエラー (pixel_refs が 1 要素)
    res_val = render_verification_overlay(
        original_image_path=sample_plot_image,
        csv_path="dummy.csv",
        x_calibration={"pixel_refs": [0.0], "val_refs": [0.0, 1.0]},
        y_calibration=y_calib,
        output_overlay_path=out_img,
    )
    assert res_val["status"] == "error"
    assert "Validation error" in res_val["message"]

    # 2. 画像は存在するが CSV が存在しない場合
    res_no_csv = render_verification_overlay(
        original_image_path=sample_plot_image,
        csv_path=str(temp_dir / "no_file.csv"),
        x_calibration=x_calib,
        y_calibration=y_calib,
        output_overlay_path=out_img,
    )
    assert res_no_csv["status"] == "error"
    assert "CSV file not found" in res_no_csv["message"]

    # 3. 画像デコード失敗
    bad_img = temp_dir / "bad_ov.png"
    bad_img.write_bytes(b"invalid")
    csv_ok = temp_dir / "ok.csv"
    pd.DataFrame({"x": [1.0, float("nan")], "y": [2.0, 5.0]}).to_csv(
        csv_ok, index=False
    )
    res_bad_img = render_verification_overlay(
        original_image_path=str(bad_img),
        csv_path=str(csv_ok),
        x_calibration=x_calib,
        y_calibration=y_calib,
        output_overlay_path=out_img,
    )
    assert res_bad_img["status"] == "error"
    assert "Failed to decode image" in res_bad_img["message"]

    # 4. 単一曲線 (2列) に NaN 行が含まれる場合
    res_single_nan = render_verification_overlay(
        original_image_path=sample_plot_image,
        csv_path=str(csv_ok),
        x_calibration=x_calib,
        y_calibration=y_calib,
        output_overlay_path=out_img,
    )
    assert res_single_nan["status"] == "success"

    # 5. Long 形式で NaN のみのグループと有効グループがある場合
    long_nan_csv = temp_dir / "long_nan.csv"
    pd.DataFrame(
        {
            "x": [1.0, 2.0, float("nan")],
            "y": [10.0, 20.0, float("nan")],
            "curve": ["g1", "g1", "g_empty"],
        }
    ).to_csv(long_nan_csv, index=False)
    res_long_nan = render_verification_overlay(
        original_image_path=sample_plot_image,
        csv_path=str(long_nan_csv),
        x_calibration=x_calib,
        y_calibration=y_calib,
        output_overlay_path=out_img,
    )
    assert res_long_nan["status"] == "success"

    # 6. imencode 失敗および例外キャッチ
    with patch("cv2.imencode", return_value=(False, None)):
        res_enc = render_verification_overlay(
            original_image_path=sample_plot_image,
            csv_path=str(csv_ok),
            x_calibration=x_calib,
            y_calibration=y_calib,
            output_overlay_path=out_img,
        )
        assert res_enc["status"] == "error"
        assert "Failed to encode overlay" in res_enc["message"]

    with patch("pandas.read_csv", side_effect=RuntimeError("csv read error")):
        res_exc = render_verification_overlay(
            original_image_path=sample_plot_image,
            csv_path=str(csv_ok),
            x_calibration=x_calib,
            y_calibration=y_calib,
            output_overlay_path=out_img,
        )
        assert res_exc["status"] == "error"
        assert "Verification overlay failed" in res_exc["message"]

    # 7. _convert_value_to_pixel が NaN を返した場合のスキップ分岐
    wide_csv = temp_dir / "wide_for_nan.csv"
    pd.DataFrame({"x": [1.0], "c1": [2.0], "c2": [3.0]}).to_csv(
        wide_csv, index=False
    )
    with patch(
        "digitize_agent.tools.visual_verifier._convert_value_to_pixel",
        return_value=float("nan"),
    ):
        for c_file in (csv_ok, long_nan_csv, wide_csv):
            res_nan_px = render_verification_overlay(
                original_image_path=sample_plot_image,
                csv_path=str(c_file),
                x_calibration=x_calib,
                y_calibration=y_calib,
                output_overlay_path=out_img,
            )
            assert res_nan_px["status"] == "success"
