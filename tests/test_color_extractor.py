"""特定色プロット点抽出および主要色検出ツールの単体テストモジュール。

extract_plot_pixels_by_color の連続線抽出、散布図重心抽出、
色プリセット抽出、および detect_plot_colors の主要色検出を検証します。
"""

from pathlib import Path

import cv2
import numpy as np

from digitize_agent.tools.color_extractor import (
    detect_plot_colors,
    extract_plot_pixels_by_color,
)


def test_extract_continuous_line(sample_plot_image: str) -> None:
    """青色の連続線プロットからピクセル点群が正しく抽出されることを検証。"""
    res = extract_plot_pixels_by_color(
        image_path=sample_plot_image,
        hsv_lower=[100, 100, 100],
        hsv_upper=[140, 255, 255],
        extract_mode="continuous_line",
    )

    assert res.get("status") != "error"
    assert res["point_count"] > 0
    assert len(res["pixel_points"]) == res["point_count"]

    xs = [pt[0] for pt in res["pixel_points"]]
    assert xs == sorted(xs)


def test_extract_color_preset_blue(sample_plot_image: str) -> None:
    """色プリセット 'blue' で青色プロット線が正しく抽出できることを検証。"""
    res = extract_plot_pixels_by_color(
        image_path=sample_plot_image,
        color_preset="blue",
        extract_mode="continuous_line",
    )

    assert res.get("status") == "success"
    assert res["point_count"] > 0


def test_detect_plot_colors(sample_plot_image: str) -> None:
    """画像内のプロット色（青線や黒軸）が正しく自動検出されることを検証。"""
    res = detect_plot_colors(
        image_path=sample_plot_image,
        max_colors=3,
    )

    assert res.get("status") == "success"
    assert res["detected_colors_count"] > 0
    color_names = [c["color_name"] for c in res["dominant_colors"]]
    # 青線と黒軸が含まれていること
    assert "blue" in color_names or "black" in color_names


def test_extract_scatter_centroids(temp_dir: Path) -> None:
    """赤色の孤立マーカー群から各重心が正しく抽出されることを検証する。"""
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    cv2.circle(img, (30, 30), radius=4, color=(0, 0, 255), thickness=-1)
    cv2.circle(img, (70, 70), radius=4, color=(0, 0, 255), thickness=-1)

    img_path = str(temp_dir / "scatter.png")
    cv2.imwrite(img_path, img)

    res = extract_plot_pixels_by_color(
        image_path=img_path,
        color_preset="red",
        extract_mode="scatter_centroids",
    )

    assert res.get("status") == "success"
    assert res["point_count"] == 2
    pts = res["pixel_points"]
    assert abs(pts[0][0] - 30) <= 1 and abs(pts[0][1] - 30) <= 1
    assert abs(pts[1][0] - 70) <= 1 and abs(pts[1][1] - 70) <= 1


def test_extract_file_not_found(temp_dir: Path) -> None:
    """存在しない画像パスを指定した際にエラーを返すことを検証する。"""
    res = extract_plot_pixels_by_color(
        image_path=str(temp_dir / "none.png"),
        color_preset="blue",
    )

    assert res["status"] == "error"
    assert "not found" in res["message"]


def test_extract_invalid_mode(sample_plot_image: str) -> None:
    """不正な extract_mode を指定した際にエラーを返すことを検証する。"""
    res = extract_plot_pixels_by_color(
        image_path=sample_plot_image,
        color_preset="blue",
        extract_mode="unknown_mode",
    )

    assert res["status"] == "error"
    assert "Validation error" in res["message"]


def test_extract_by_target_rgb(temp_dir: Path) -> None:
    """target_rgb 指定により特定RGB色のプロット点が抽出されることを検証。"""
    img = np.full((50, 50, 3), 255, dtype=np.uint8)
    # 純青色 (BGR: 255, 0, 0)
    cv2.circle(img, (25, 25), radius=3, color=(255, 0, 0), thickness=-1)
    img_path = str(temp_dir / "rgb_test.png")
    cv2.imwrite(img_path, img)

    res = extract_plot_pixels_by_color(
        image_path=img_path,
        target_rgb=[0, 0, 255],
        extract_mode="scatter_centroids",
    )
    assert res.get("status") == "success"
    assert res["point_count"] == 1
    assert abs(res["pixel_points"][0][0] - 25) <= 1


def test_extract_by_target_hex(temp_dir: Path) -> None:
    """target_hex 指定により16進カラーコードで抽出できることを検証。"""
    img = np.full((50, 50, 3), 255, dtype=np.uint8)
    # 赤色 (BGR: 0, 0, 255)
    cv2.circle(img, (20, 20), radius=3, color=(0, 0, 255), thickness=-1)
    img_path = str(temp_dir / "hex_test.png")
    cv2.imwrite(img_path, img)

    res = extract_plot_pixels_by_color(
        image_path=img_path,
        target_hex="#FF0000",
        extract_mode="scatter_centroids",
    )
    assert res.get("status") == "success"
    assert res["point_count"] == 1


def test_extract_with_bbox_and_exclude(temp_dir: Path) -> None:
    """bbox による範囲限定および exclude_bboxes による除外を検証。"""
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    # (25, 25) と (75, 75) に青点を配置
    cv2.circle(img, (25, 25), radius=3, color=(255, 0, 0), thickness=-1)
    cv2.circle(img, (75, 75), radius=3, color=(255, 0, 0), thickness=-1)
    img_path = str(temp_dir / "bbox_test.png")
    cv2.imwrite(img_path, img)

    # 1. bbox で左上のみ指定
    res1 = extract_plot_pixels_by_color(
        image_path=img_path,
        color_preset="blue",
        bbox=[0, 0, 50, 50],
        extract_mode="scatter_centroids",
    )
    assert res1.get("status") == "success"
    assert res1["point_count"] == 1
    assert abs(res1["pixel_points"][0][0] - 25) <= 1

    # 2. exclude_bboxes で左上を除外指定
    res2 = extract_plot_pixels_by_color(
        image_path=img_path,
        color_preset="blue",
        exclude_bboxes=[[20, 20, 30, 30]],
        extract_mode="scatter_centroids",
    )
    assert res2.get("status") == "success"
    assert res2["point_count"] == 1
    assert abs(res2["pixel_points"][0][0] - 75) <= 1


def test_extract_cyan_preset(temp_dir: Path) -> None:
    """新プリセット 'cyan' でシアン色のプロットが抽出できることを検証。"""
    img = np.full((50, 50, 3), 255, dtype=np.uint8)
    # シアン色 (BGR: 255, 255, 0)
    cv2.circle(img, (25, 25), radius=3, color=(255, 255, 0), thickness=-1)
    img_path = str(temp_dir / "cyan_test.png")
    cv2.imwrite(img_path, img)

    res = extract_plot_pixels_by_color(
        image_path=img_path,
        color_preset="cyan",
        extract_mode="scatter_centroids",
    )
    assert res.get("status") == "success"
    assert res["point_count"] >= 1


def test_color_tolerance_auto_scaling(temp_dir: Path) -> None:
    """color_tolerance が正規化値と 255 距離の両方で動作することを検証。"""
    img = np.full((60, 60, 3), 255, dtype=np.uint8)
    # BGR で (200, 20, 20) -> RGB (20, 20, 200) 赤み
    img[25:35, 10:50] = (200, 20, 20)
    img_path = str(temp_dir / "tolerance_test.png")
    cv2.imwrite(img_path, img)

    # 1. 0.0〜1.0 スケール (0.15)
    res_norm = extract_plot_pixels_by_color(
        image_path=img_path,
        target_rgb=[20, 20, 200],
        color_tolerance=0.15,
        extract_mode="continuous_line",
    )
    assert res_norm.get("status") == "success"
    assert res_norm["point_count"] > 0

    # 2. 0〜255 スケール (40.0)
    res_255 = extract_plot_pixels_by_color(
        image_path=img_path,
        target_rgb=[20, 20, 200],
        color_tolerance=40.0,
        extract_mode="continuous_line",
    )
    assert res_255.get("status") == "success"
    assert res_255["point_count"] > 0
