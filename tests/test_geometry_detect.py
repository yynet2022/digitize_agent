"""幾何構造（軸・目盛り）検出ツールの単体テストモジュール。

detect_axes_and_ticks の軸特定、目盛り候補抽出、およびエラー耐性を検証します。
"""

from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pymupdf as fitz

from digitize_agent.tools.geometry_detect import (
    _detect_box_frame,
    _extract_legend_items,
    _find_axis_line,
    auto_calibrate_axes,
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


def test_detect_axes_box_frame(temp_dir: Path) -> None:
    """四角い外枠（Box Frame）を持つプロット画像から枠情報が検出されること。"""
    import cv2
    import numpy as np

    img = np.full((300, 300, 3), 255, dtype=np.uint8)
    # 外枠線 (x: 40..260, y: 30..270)
    cv2.rectangle(img, (40, 30), (260, 270), (0, 0, 0), 2)
    img_path = str(temp_dir / "box_frame_test.png")
    cv2.imwrite(img_path, img)

    res = detect_axes_and_ticks(img_path)
    assert res.get("status") != "error"
    assert res.get("box_frame") is not None
    bf = res["box_frame"]
    assert "inner_bbox" in bf
    assert "outer_bbox" in bf
    assert abs(bf["outer_bbox"][0] - 40) <= 3
    assert abs(bf["outer_bbox"][2] - 260) <= 3


def test_auto_calibrate_axes_normalized(temp_dir: Path) -> None:
    """目盛りのない画像に対し正規化モードでキャリブレーションが行われること。"""
    import cv2
    import numpy as np

    from digitize_agent.tools.geometry_detect import auto_calibrate_axes

    img = np.full((300, 300, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (40, 30), (260, 270), (0, 0, 0), 2)
    img_path = str(temp_dir / "norm_calib_test.png")
    cv2.imwrite(img_path, img)

    res = auto_calibrate_axes(
        image_path=img_path,
        mode="normalized",
        normalized_domain_x=[0.0, 1.0],
        normalized_domain_y=[0.0, 1.0],
    )
    assert res.get("status") == "success"
    assert res.get("calibration_mode") == "normalized"
    assert "x_calibration" in res
    assert "y_calibration" in res
    x_cal = res["x_calibration"]
    assert x_cal["val_refs"] == [0.0, 1.0]
    assert abs(x_cal["pixel_refs"][0] - 40) <= 5


def test_geometry_helpers_and_blank_axes(temp_dir: Path) -> None:
    """_find_axis_line, _detect_box_frame, _extract_legend_items 等を検証。"""
    # 1. _find_axis_line のピークなし・下端/上端優先・最大ピーク選択
    proj_zero = np.zeros(50, dtype=float)
    assert _find_axis_line(proj_zero, min_length=5) == 0

    proj_peaks = np.zeros(100, dtype=float)
    proj_peaks[20] = 30.0
    proj_peaks[70] = 15.0
    assert (
        _find_axis_line(proj_peaks, min_length=10, prefer_lower_or_left=True)
        == 70
    )
    assert (
        _find_axis_line(proj_peaks, min_length=10, prefer_lower_or_left=False)
        == 20
    )

    # 2. _detect_box_frame で輪郭がない白紙画像
    blank = np.full((100, 100, 3), 255, dtype=np.uint8)
    assert _detect_box_frame(blank) is None

    # 塗りつぶし矩形（子輪郭がないため pad で inner_box を作る分岐）
    filled = np.full((200, 200, 3), 255, dtype=np.uint8)
    cv2.rectangle(filled, (30, 30), (170, 170), (0, 0, 0), -1)
    bf_filled = _detect_box_frame(filled)
    assert bf_filled is not None
    assert bf_filled["inner_bbox"][0] > bf_filled["outer_bbox"][0]

    # 3. _extract_legend_items の空 ROI
    assert _extract_legend_items(blank, [50, 50, 50, 50]) == []

    # 4. 白紙画像に対する detect_axes_and_ticks と画像デコード失敗
    blank_path = str(temp_dir / "blank_axes.png")
    cv2.imwrite(blank_path, blank)
    res_blank = detect_axes_and_ticks(blank_path, detect_box_frame=False)
    assert "x_axis" in res_blank
    assert "y_axis" in res_blank

    bad_path = temp_dir / "bad_axes.png"
    bad_path.write_bytes(b"corrupt")
    res_bad = detect_axes_and_ticks(str(bad_path))
    assert res_bad["status"] == "error"
    assert "Failed to load image" in res_bad["message"]


def test_detect_legend_region_frameless_and_not_detected(
    temp_dir: Path,
) -> None:
    """枠なしテキスト塊からの凡例検出・未検出・異常系分岐を検証する。"""
    # 1. バリデーションエラー & デコードエラー
    res_val = detect_legend_region(image_path="x.png", plot_bbox=[1, 2])
    assert res_val["status"] == "error"
    assert "Validation error" in res_val["message"]

    bad_path = temp_dir / "bad_leg.png"
    bad_path.write_bytes(b"corrupt")
    res_bad = detect_legend_region(image_path=str(bad_path))
    assert res_bad["status"] == "error"
    assert "Failed to load image" in res_bad["message"]

    # 2. 凡例が一切ない白紙画像 (legend_detected=False)
    blank = np.full((200, 200, 3), 255, dtype=np.uint8)
    blank_path = str(temp_dir / "no_legend.png")
    cv2.imwrite(blank_path, blank)
    res_none = detect_legend_region(
        image_path=blank_path, plot_bbox=[10, 10, 190, 190]
    )
    assert res_none["status"] == "success"
    assert res_none["legend_detected"] is False

    # 3. 枠線なしで2行のテキスト塊がある画像 (text_rects >= 2 の分岐)
    frameless = np.full((200, 200, 3), 255, dtype=np.uint8)
    # 2行の水平テキスト風ブロブ（幅40, 高さ12）をプロット内に配置
    cv2.rectangle(frameless, (100, 40), (145, 52), (0, 0, 0), -1)
    cv2.rectangle(frameless, (100, 65), (145, 77), (0, 0, 0), -1)
    fl_path = str(temp_dir / "frameless_legend.png")
    cv2.imwrite(fl_path, frameless)
    res_fl = detect_legend_region(
        image_path=fl_path, plot_bbox=[20, 20, 180, 180]
    )
    assert res_fl["status"] == "success"
    assert res_fl["legend_detected"] is True
    assert res_fl["confidence"] == 0.65


def test_auto_calibrate_axes_scale_multipliers_and_errors(
    temp_dir: Path,
) -> None:
    """auto_calibrate_axes のスケール倍率とフォールバック分岐を検証する。"""
    # 1. バリデーションエラー・未存在・デコード失敗
    res_val = auto_calibrate_axes(image_path="x.png", dpi=-5.0)
    assert res_val["status"] == "error"

    res_missing = auto_calibrate_axes(image_path=str(temp_dir / "no.png"))
    assert res_missing["status"] == "error"

    bad_img = temp_dir / "bad_cal.png"
    bad_img.write_bytes(b"corrupt")
    res_bad = auto_calibrate_axes(image_path=str(bad_img))
    assert res_bad["status"] == "error"

    # 2. PDF 内に "1e-3", "10^4", 非数値ラベルがある場合のスケール倍率検出と
    #    box_frame がない画像の normalized_fallback を検証
    doc1 = fitz.open()
    p1 = doc1.new_page(width=200, height=200)
    p1.insert_text(fitz.Point(20, 20), "×1e-3", fontsize=9)
    p1.insert_text(fitz.Point(50, 20), " Label ", fontsize=9)
    pdf1 = str(temp_dir / "sci1.pdf")
    doc1.save(pdf1)
    doc1.close()

    doc2 = fitz.open()
    p2 = doc2.new_page(width=200, height=200)
    p2.insert_text(fitz.Point(20, 20), "10^4", fontsize=9)
    pdf2 = str(temp_dir / "sci2.pdf")
    doc2.save(pdf2)
    doc2.close()

    # 外枠 (box_frame) のない単純なL字軸画像
    l_img = np.full((200, 200, 3), 255, dtype=np.uint8)
    cv2.line(l_img, (30, 160), (180, 160), (0, 0, 0), 2)
    cv2.line(l_img, (30, 20), (30, 160), (0, 0, 0), 2)
    l_path = str(temp_dir / "l_axes.png")
    cv2.imwrite(l_path, l_img)

    res_sci1 = auto_calibrate_axes(
        image_path=l_path,
        pdf_path=pdf1,
        crop_bbox_points=[0.0, 0.0, 200.0, 200.0],
        dpi=72.0,
    )
    assert res_sci1["status"] == "success"
    assert res_sci1["scale_multiplier"] == 1e-3
    assert res_sci1["calibration_mode"] == "normalized_fallback"

    res_sci2 = auto_calibrate_axes(
        image_path=l_path,
        pdf_path=pdf2,
        box_frame_bbox=[30, 20, 180, 160],
        dpi=72.0,
    )
    assert res_sci2["status"] == "success"
    assert res_sci2["scale_multiplier"] == 1e4

    # PDF オープン時に例外が発生してもフォールバックで継続すること
    with patch("pymupdf.open", side_effect=RuntimeError("pdf open error")):
        res_pdf_err = auto_calibrate_axes(image_path=l_path, pdf_path=pdf1)
        assert res_pdf_err["status"] == "success"
