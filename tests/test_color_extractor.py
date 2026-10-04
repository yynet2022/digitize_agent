"""特定色プロット点抽出ツールの単体テストモジュール。

extract_plot_pixels_by_color の連続線抽出、散布図重心抽出、
およびエラー耐性を検証します。
"""

from pathlib import Path

import cv2
import numpy as np

from digitize_agent.tools.color_extractor import extract_plot_pixels_by_color


def test_extract_continuous_line(sample_plot_image: str) -> None:
    """青色の連続線プロットからピクセル点群が正しく抽出されることを検証する。"""
    # 青色の HSV 範囲: H 100〜140, S 100〜255, V 100〜255
    res = extract_plot_pixels_by_color(
        image_path=sample_plot_image,
        hsv_lower=[100, 100, 100],
        hsv_upper=[140, 255, 255],
        extract_mode="continuous_line",
    )

    assert res.get("status") != "error"
    assert res["point_count"] > 0
    assert len(res["pixel_points"]) == res["point_count"]

    # X座標が昇順にソートされていること
    xs = [pt[0] for pt in res["pixel_points"]]
    assert xs == sorted(xs)


def test_extract_scatter_centroids(temp_dir: Path) -> None:
    """赤色の孤立マーカー群から各重心が正しく抽出されることを検証する。"""
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    # 赤色の円 (BGR: (0, 0, 255)) を (30, 30), (70, 70) に描画
    cv2.circle(img, (30, 30), radius=4, color=(0, 0, 255), thickness=-1)
    cv2.circle(img, (70, 70), radius=4, color=(0, 0, 255), thickness=-1)

    img_path = str(temp_dir / "scatter.png")
    cv2.imwrite(img_path, img)

    # 赤色の HSV 範囲 (H 0〜10 または 170〜179)
    res = extract_plot_pixels_by_color(
        image_path=img_path,
        hsv_lower=[0, 150, 150],
        hsv_upper=[10, 255, 255],
        extract_mode="scatter_centroids",
    )

    assert res.get("status") != "error"
    assert res["point_count"] == 2
    # 重心が (30, 30) と (70, 70) 近傍にあること
    pts = res["pixel_points"]
    assert abs(pts[0][0] - 30) <= 1 and abs(pts[0][1] - 30) <= 1
    assert abs(pts[1][0] - 70) <= 1 and abs(pts[1][1] - 70) <= 1


def test_extract_file_not_found(temp_dir: Path) -> None:
    """存在しない画像パスを指定した際にエラーを返すことを検証する。"""
    res = extract_plot_pixels_by_color(
        image_path=str(temp_dir / "none.png"),
        hsv_lower=[0, 0, 0],
        hsv_upper=[10, 10, 10],
    )

    assert res["status"] == "error"
    assert "not found" in res["message"]


def test_extract_invalid_mode(sample_plot_image: str) -> None:
    """不正な extract_mode を指定した際に
    バリデーションエラーを返すことを検証する。
    """
    res = extract_plot_pixels_by_color(
        image_path=sample_plot_image,
        hsv_lower=[0, 0, 0],
        hsv_upper=[255, 255, 255],
        extract_mode="unknown_mode",
    )

    assert res["status"] == "error"
    assert "Validation error" in res["message"]
