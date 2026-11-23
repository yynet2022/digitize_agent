"""画像変換・前処理ツールの単体テストモジュール。

crop_and_transform_region の画像および PDF からの切り出し、
傾き補正、コントラスト強調、およびエラー耐性を検証します。
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from digitize_agent.tools.image_transforms import (
    _detect_and_correct_skew,
    _estimate_figure_bbox_above_caption,
    _load_image_from_pdf,
    crop_and_transform_region,
)


def test_crop_basic(sample_plot_image: str, temp_dir: Path) -> None:
    """指定バウンディングボックスの領域が正常に切り出せることを検証する。"""
    out_path = str(temp_dir / "cropped.png")
    res = crop_and_transform_region(
        image_path=sample_plot_image,
        bbox=[20, 20, 100, 120],
        output_path=out_path,
    )

    assert res["status"] == "success"
    assert res["dimensions"]["width"] == 80
    assert res["dimensions"]["height"] == 100
    assert Path(out_path).is_file()

    img = cv2.imread(out_path)
    assert img.shape == (100, 80, 3)


def test_crop_with_deskew(temp_dir: Path) -> None:
    """傾いた直線の画像に対して傾き検出と補正が機能することを検証する。"""
    center = (75, 75)
    rot_mat = cv2.getRotationMatrix2D(center, 5.0, 1.0)
    # 水平線を描いて回転
    base_line = np.full((150, 150, 3), 255, dtype=np.uint8)
    cv2.line(base_line, (20, 75), (130, 75), (0, 0, 0), 3)
    rotated = cv2.warpAffine(
        base_line, rot_mat, (150, 150), borderValue=(255, 255, 255)
    )

    in_path = str(temp_dir / "skewed.png")
    out_path = str(temp_dir / "deskewed.png")
    cv2.imwrite(in_path, rotated)

    res = crop_and_transform_region(
        image_path=in_path,
        bbox=[0, 0, 150, 150],
        deskew=True,
        output_path=out_path,
    )

    assert res["status"] == "success"
    assert abs(res["skew_angle_detected"]) > 0.5


def test_crop_with_enhance_contrast(
    sample_plot_image: str, temp_dir: Path
) -> None:
    """コントラスト強調 (CLAHE) オプションが正常に適用されることを検証する。"""
    out_path = str(temp_dir / "enhanced.png")
    res = crop_and_transform_region(
        image_path=sample_plot_image,
        bbox=[10, 10, 80, 80],
        enhance_contrast=True,
        output_path=out_path,
    )

    assert res["status"] == "success"
    assert Path(out_path).is_file()


def test_crop_from_pdf_direct(
    sample_pdf_with_text_and_lines: str, temp_dir: Path
) -> None:
    """PDF から直接指定 DPI で指定領域がクロップできることを検証する。"""
    out_path = str(temp_dir / "pdf_cropped.png")
    res = crop_and_transform_region(
        pdf_path=sample_pdf_with_text_and_lines,
        page_number=0,
        dpi=144.0,
        bbox=[50, 50, 200, 250],
        output_path=out_path,
    )

    assert res["status"] == "success"
    assert res["dimensions"]["width"] == 150
    assert res["dimensions"]["height"] == 200
    assert Path(out_path).is_file()


def test_crop_from_pdf_bbox_mode_point(
    sample_pdf_with_text_and_lines: str, temp_dir: Path
) -> None:
    """PDF 座標 (pt) 単位の bbox が DPI に応じて自動換算されることを検証。"""
    out_path = str(temp_dir / "pdf_point_cropped.png")
    # 72 DPI では 1 pt = 1 px, 144 DPI では 1 pt = 2 px
    # bbox=[10, 10, 60, 80] pt -> 幅 50 pt (100 px), 高さ 70 pt (140 px)
    res = crop_and_transform_region(
        pdf_path=sample_pdf_with_text_and_lines,
        page_number=0,
        dpi=144.0,
        bbox=[10, 10, 60, 80],
        bbox_mode="point",
        output_path=out_path,
    )

    assert res["status"] == "success"
    assert res["dimensions"]["width"] == 100
    assert res["dimensions"]["height"] == 140
    assert Path(out_path).is_file()


def test_crop_neither_image_nor_pdf(temp_dir: Path) -> None:
    """image_path も pdf_path も未指定の場合にエラーを返すことを検証。"""
    out_path = str(temp_dir / "fail.png")
    res = crop_and_transform_region(
        bbox=[0, 0, 50, 50],
        output_path=out_path,
    )

    assert res["status"] == "error"
    assert "Either image_path or pdf_path" in res["message"]


def test_crop_file_not_found(temp_dir: Path) -> None:
    """存在しない元画像パスを指定した際にエラーを返すことを検証する。"""
    out_path = str(temp_dir / "fail.png")
    res = crop_and_transform_region(
        image_path=str(temp_dir / "none.png"),
        bbox=[0, 0, 50, 50],
        output_path=out_path,
    )

    assert res["status"] == "error"
    assert "not found" in res["message"]


def test_crop_invalid_bbox(sample_plot_image: str, temp_dir: Path) -> None:
    """要素数が不正な bbox を指定した際に
    バリデーションエラーを返すことを検証する。
    """
    out_path = str(temp_dir / "fail.png")
    res = crop_and_transform_region(
        image_path=sample_plot_image,
        bbox=[0, 0, 50],  # 3つしかない
        output_path=out_path,
    )

    assert res["status"] == "error"
    assert "Validation error" in res["message"]


def test_crop_with_caption_bbox(
    sample_pdf_with_text_and_lines: str, temp_dir: Path
) -> None:
    """caption_bbox から図表領域が自動推定されてクロップされることを検証。"""
    out_path = str(temp_dir / "caption_auto_crop.png")
    res = crop_and_transform_region(
        pdf_path=sample_pdf_with_text_and_lines,
        page_number=0,
        caption_bbox=[50.0, 250.0, 200.0, 270.0],
        output_path=out_path,
    )

    assert res["status"] == "success"
    assert "estimated_bbox" in res
    assert len(res["estimated_bbox"]) == 4
    assert res["dimensions"]["width"] > 0
    assert res["dimensions"]["height"] > 0
    assert Path(out_path).is_file()


def test_detect_and_correct_skew_edge_branches() -> None:
    """_detect_and_correct_skew の直線なし・垂直線のみ・微小角度を検証。"""
    blank = np.full((100, 100, 3), 255, dtype=np.uint8)
    out_blank, ang_blank = _detect_and_correct_skew(blank)
    assert ang_blank == 0.0
    assert out_blank.shape == blank.shape

    # 垂直線のみ (-45〜45度外の直線ばかりの場合)
    vert = np.full((100, 100, 3), 255, dtype=np.uint8)
    cv2.line(vert, (50, 5), (50, 95), (0, 0, 0), 3)
    _, ang_vert = _detect_and_correct_skew(vert)
    assert ang_vert == 0.0

    # 水平線のみ (傾き < 0.5 度)
    horiz = np.full((100, 100, 3), 255, dtype=np.uint8)
    cv2.line(horiz, (5, 50), (95, 50), (0, 0, 0), 3)
    _, ang_horiz = _detect_and_correct_skew(horiz)
    assert abs(ang_horiz) < 0.5

    # HoughLinesP が4要素未満の配列を返した場合のスキップ分岐
    with patch("cv2.HoughLinesP", return_value=[np.array([1, 2])]):
        _, ang_short = _detect_and_correct_skew(horiz)
        assert ang_short == 0.0


def test_load_image_from_pdf_and_estimate_bbox_branches(
    sample_scanned_pdf: str,
) -> None:
    """_load_image_from_pdf と _estimate_figure_bbox_above_caption を検証。"""
    with pytest.raises(FileNotFoundError):
        _load_image_from_pdf("missing_file.pdf", 0, 72.0)

    with pytest.raises(IndexError):
        _load_image_from_pdf(sample_scanned_pdf, 99, 72.0)

    # pix.n == 1 (グレースケール) と pix.n == 2 の分岐
    for channels in (1, 2):
        fake_pix = MagicMock()
        fake_pix.height = 10
        fake_pix.width = 10
        fake_pix.n = channels
        fake_pix.samples = np.zeros(
            (10, 10, channels), dtype=np.uint8
        ).tobytes()
        fake_page = MagicMock()
        fake_page.get_pixmap.return_value = fake_pix
        fake_doc = MagicMock()
        fake_doc.__enter__.return_value = [fake_page]
        with patch("pymupdf.open", return_value=fake_doc):
            loaded = _load_image_from_pdf(sample_scanned_pdf, 0, 72.0)
            assert loaded.shape[:2] == (10, 10)

    # 描画要素がないページ等のフォールバック
    import pymupdf as fitz

    fake_page2 = MagicMock()
    fake_page2.rect = fitz.Rect(0, 0, 300, 400)
    fake_page2.get_drawings.return_value = [
        {"rect": None},
        {"rect": fitz.Rect(0, 100, 295, 200)},  # width > 95% でスキップ
        {"rect": fitz.Rect(10, 300, 100, 350)},  # r.y1 > c_y0 + 5.0 でスキップ
    ]
    est = _estimate_figure_bbox_above_caption(
        fake_page2, [50.0, 250.0, 200.0, 270.0]
    )
    assert len(est) == 4
    assert est[3] == 245.0


def test_crop_and_transform_error_branches(
    sample_plot_image: str,
    sample_scanned_pdf: str,
    temp_dir: Path,
) -> None:
    """crop_and_transform_region の各種エラーハンドリング分岐を検証する。"""
    out_path = str(temp_dir / "err_out.png")

    # 1. bbox も caption_bbox も未指定
    res_no_box = crop_and_transform_region(
        image_path=sample_plot_image,
        output_path=out_path,
    )
    assert res_no_box["status"] == "error"
    assert "Either bbox or caption_bbox" in res_no_box["message"]

    # 2. 画像デコード失敗
    bad_img = temp_dir / "corrupt_crop.png"
    bad_img.write_bytes(b"corrupt")
    res_decode = crop_and_transform_region(
        image_path=str(bad_img),
        bbox=[0, 0, 10, 10],
        output_path=out_path,
    )
    assert res_decode["status"] == "error"
    assert "Failed to decode image" in res_decode["message"]

    # 3. caption_bbox のみ指定して pdf_path がない場合
    res_cap_no_pdf = crop_and_transform_region(
        image_path=sample_plot_image,
        caption_bbox=[10.0, 50.0, 100.0, 70.0],
        output_path=out_path,
    )
    assert res_cap_no_pdf["status"] == "error"
    assert "requires pdf_path" in res_cap_no_pdf["message"]

    # 4. imencode 失敗時
    with patch("cv2.imencode", return_value=(False, None)):
        res_enc = crop_and_transform_region(
            image_path=sample_plot_image,
            bbox=[0, 0, 50, 50],
            output_path=out_path,
        )
        assert res_enc["status"] == "error"
        assert "Failed to encode image" in res_enc["message"]

    # 5. PDF ページ範囲外で例外キャッチ
    res_oob = crop_and_transform_region(
        pdf_path=sample_scanned_pdf,
        page_number=99,
        bbox=[0, 0, 50, 50],
        output_path=out_path,
    )
    assert res_oob["status"] == "error"
    assert "Failed to process and save cropped image" in res_oob["message"]

    # 6. caption_bbox 推定時のページ範囲外分岐と空画像分岐
    dummy_img = np.zeros((10, 10, 3), dtype=np.uint8)
    with patch(
        "digitize_agent.tools.image_transforms._load_image_from_pdf",
        return_value=dummy_img,
    ):
        res_cap_oob = crop_and_transform_region(
            pdf_path=sample_scanned_pdf,
            page_number=99,
            caption_bbox=[10.0, 50.0, 100.0, 70.0],
            output_path=out_path,
        )
        assert res_cap_oob["status"] == "error"
        assert "out of range" in res_cap_oob["message"]

    empty_3d = np.zeros((0, 0, 3), dtype=np.uint8)
    with patch("cv2.imread", return_value=empty_3d):
        res_empty = crop_and_transform_region(
            image_path=sample_plot_image,
            bbox=[0, 0, 10, 10],
            output_path=out_path,
        )
        assert res_empty["status"] == "error"
        assert "Cropped region is empty" in res_empty["message"]
