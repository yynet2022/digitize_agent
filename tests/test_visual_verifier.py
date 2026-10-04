"""検証オーバーレイ描画ツールの単体テストモジュール。

render_verification_overlay の再プロット描画、透過合成、
複数曲線の色分けグループ化、適合度指標計算、およびエラー耐性を検証します。
"""

from pathlib import Path

import pandas as pd

from digitize_agent.tools.visual_verifier import render_verification_overlay


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
