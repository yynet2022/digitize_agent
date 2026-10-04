"""幾何構造（軸・目盛り）検出ツールの単体テストモジュール。

detect_axes_and_ticks の軸特定、目盛り候補抽出、およびエラー耐性を検証します。
"""

from pathlib import Path

from digitize_agent.tools.geometry_detect import detect_axes_and_ticks


def test_detect_axes_and_ticks_basic(sample_plot_image: str) -> None:
    """合成グラフ画像から X 軸、Y 軸、目盛り候補が検出されることを検証する。"""
    res = detect_axes_and_ticks(
        image_path=sample_plot_image, min_line_length_ratio=0.3
    )

    assert res.get("status") != "error"
    assert "x_axis" in res
    assert "y_axis" in res

    # X軸は y=160 付近に描画
    assert abs(res["x_axis"]["y_pixel"] - 160) <= 5
    # Y軸は x=40 付近に描画
    assert abs(res["y_axis"]["x_pixel"] - 40) <= 5

    # 目盛り候補が存在すること
    assert len(res["x_tick_candidates"]) > 0
    assert len(res["y_tick_candidates"]) > 0


def test_detect_axes_file_not_found(temp_dir: Path) -> None:
    """存在しない画像パスを指定した際にエラーを返すことを検証する。"""
    res = detect_axes_and_ticks(image_path=str(temp_dir / "non_existent.png"))

    assert res["status"] == "error"
    assert "not found" in res["message"]


def test_detect_axes_invalid_ratio(sample_plot_image: str) -> None:
    """範囲外の比率パラメータを指定した際にエラーを返すことを検証する。"""
    res = detect_axes_and_ticks(
        image_path=sample_plot_image, min_line_length_ratio=2.5
    )

    assert res["status"] == "error"
    assert "Validation error" in res["message"]
