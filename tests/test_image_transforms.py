"""画像変換・前処理ツールの単体テストモジュール。

crop_and_transform_region の画像および PDF からの切り出し、
傾き補正、コントラスト強調、およびエラー耐性を検証します。
"""

from pathlib import Path

import cv2
import numpy as np

from digitize_agent.tools.image_transforms import crop_and_transform_region


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
