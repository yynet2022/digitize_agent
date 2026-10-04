"""幾何構造（軸・目盛り）検出ツールの単体テストモジュール。

detect_axes_and_ticks の軸特定、目盛り候補抽出、およびエラー耐性を検証します。
"""

from pathlib import Path

from digitize_agent.tools.geometry_detect import (
    detect_axes_and_ticks,
    detect_legend_region,
)


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


def test_detect_legend_region_with_box(temp_dir: Path) -> None:
    """プロット枠内の凡例ボックス矩形が正しく検出されることを検証する。"""
    import cv2
    import numpy as np

    # 200x200 の合成プロット画像を作成
    img = np.full((200, 200, 3), 255, dtype=np.uint8)
    # 外枠プロット線 (x: 30..180, y: 20..170)
    cv2.rectangle(img, (30, 20), (180, 170), (0, 0, 0), 2)
    # 右上に凡例ボックスを描画 (x: 110..170, y: 30..70)
    cv2.rectangle(img, (110, 30), (170, 70), (50, 50, 50), 1)
    # 凡例内部にダミーの線とテキスト
    cv2.line(img, (115, 45), (130, 45), (255, 0, 0), 2)
    cv2.putText(
        img, "Data", (135, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1
    )

    img_path = str(temp_dir / "legend_plot.png")
    cv2.imwrite(img_path, img)

    res = detect_legend_region(
        image_path=img_path,
        plot_bbox=[30, 20, 180, 170],
    )
    assert res.get("status") == "success"
    assert res["legend_detected"] is True
    assert res["legend_bbox"] is not None
    bx0, by0, bx1, by1 = res["legend_bbox"]
    # 凡例ボックス (110, 30, 170, 70) に近いこと
    assert abs(bx0 - 110) <= 8
    assert abs(by0 - 30) <= 8
    assert abs(bx1 - 170) <= 8
    assert abs(by1 - 70) <= 8


def test_detect_legend_region_missing_file(temp_dir: Path) -> None:
    """存在しない画像パスを指定した際にエラーを返すことを検証する。"""
    res = detect_legend_region(image_path=str(temp_dir / "missing.png"))
    assert res["status"] == "error"
    assert "not found" in res["message"]


def test_detect_legend_items(temp_dir: Path) -> None:
    """凡例領域から項目ごとの色サンプリング情報が取得できることを検証。"""
    import cv2
    import numpy as np

    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (90, 60), (0, 0, 0), 1)
    cv2.line(img, (15, 25), (35, 25), (0, 0, 255), 2)
    cv2.line(img, (15, 45), (35, 45), (255, 0, 0), 2)

    img_path = str(temp_dir / "legend_items_test.png")
    cv2.imwrite(img_path, img)

    res = detect_legend_region(image_path=img_path)
    assert res.get("status") == "success"
    assert "legend_items" in res
    assert isinstance(res["legend_items"], list)


def test_auto_calibrate_axes_synthetic(temp_dir: Path) -> None:
    """目盛りとテキストの自動ペアリング校正パラメータ推定を検証。"""
    import cv2
    import numpy as np
    import pymupdf as fitz

    from digitize_agent.tools.geometry_detect import auto_calibrate_axes

    doc = fitz.open()
    page = doc.new_page(width=200, height=200)
    page.insert_text(fitz.Point(60, 170), "10", fontsize=8)
    page.insert_text(fitz.Point(100, 170), "20", fontsize=8)
    page.insert_text(fitz.Point(140, 170), "30", fontsize=8)
    page.insert_text(fitz.Point(20, 122), "5", fontsize=8)
    page.insert_text(fitz.Point(20, 82), "10", fontsize=8)
    page.insert_text(fitz.Point(20, 42), "15", fontsize=8)

    pdf_path = str(temp_dir / "calib_test.pdf")
    doc.save(pdf_path)
    doc.close()

    img = np.full((200, 200, 3), 255, dtype=np.uint8)
    cv2.line(img, (30, 160), (180, 160), (0, 0, 0), 2)
    cv2.line(img, (40, 20), (40, 170), (0, 0, 0), 2)
    for xt in [60, 100, 140]:
        cv2.line(img, (xt, 160), (xt, 165), (0, 0, 0), 1)
    for yt in [40, 80, 120]:
        cv2.line(img, (35, yt), (40, yt), (0, 0, 0), 1)

    img_path = str(temp_dir / "calib_img.png")
    cv2.imwrite(img_path, img)

    res = auto_calibrate_axes(
        image_path=img_path,
        pdf_path=pdf_path,
        dpi=72.0,
    )
    assert res.get("status") == "success"
    assert "x_calibration" in res
    assert "y_calibration" in res
