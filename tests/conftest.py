"""pytest テスト用共通フィクスチャモジュール。

テスト用の合成画像、合成 PDF、一時ファイル生成ヘルパーを提供します。
"""

from pathlib import Path

import cv2
import fitz
import numpy as np
import pytest


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """テスト用の一時ディレクトリパスを提供する。

    Args:
        tmp_path: pytest 提供の一時ディレクトリパス。

    Returns:
        Path: 一時ディレクトリパス。
    """
    return tmp_path


@pytest.fixture
def sample_plot_image(tmp_path: Path) -> str:
    """軸と目盛り線、青色の線プロットを持つ合成画像を生成する。

    Args:
        tmp_path: 一時ディレクトリパス。

    Returns:
        str: 生成された合成画像の絶対パス。
    """
    width, height = 200, 200
    img = np.full((height, width, 3), 255, dtype=np.uint8)

    # 黒い X 軸 (y=160, x=30..180) と Y 軸 (x=40, y=20..170)
    cv2.line(img, (30, 160), (180, 160), (0, 0, 0), 2)
    cv2.line(img, (40, 20), (40, 170), (0, 0, 0), 2)

    # X 目盛り (x=60, 100, 140)
    for x_tick in [60, 100, 140]:
        cv2.line(img, (x_tick, 160), (x_tick, 165), (0, 0, 0), 1)

    # Y 目盛り (y=40, 80, 120)
    for y_tick in [40, 80, 120]:
        cv2.line(img, (35, y_tick), (40, y_tick), (0, 0, 0), 1)

    # 青色プロット線 (BGR: (255, 0, 0), HSV: H~120)
    pts = np.array(
        [[50, 140], [80, 110], [120, 70], [150, 50]], dtype=np.int32
    )
    cv2.polylines(img, [pts], isClosed=False, color=(255, 0, 0), thickness=2)

    file_path = tmp_path / "sample_plot.png"
    cv2.imwrite(str(file_path), img)
    return str(file_path)


@pytest.fixture
def sample_pdf_with_text_and_lines(tmp_path: Path) -> str:
    """テキストとベクター線を含むテスト用 PDF を生成する。

    Args:
        tmp_path: 一時ディレクトリパス。

    Returns:
        str: 生成された PDF の絶対パス。
    """
    doc = fitz.open()
    page = doc.new_page(width=300, height=300)

    # テキストの挿入
    page.insert_text(
        fitz.Point(50, 50),
        "Title: Sample Plot",
        fontsize=12,
        color=(0, 0, 0),
    )
    page.insert_text(
        fitz.Point(50, 80),
        "X-axis label",
        fontsize=10,
        color=(0, 0, 0),
    )

    # 線の描画
    shape = page.new_shape()
    shape.draw_line(fitz.Point(50, 200), fitz.Point(250, 200))
    shape.draw_line(fitz.Point(50, 50), fitz.Point(50, 200))
    shape.finish(color=(0, 0, 0), width=1.5)
    shape.commit()

    pdf_path = tmp_path / "sample_vector.pdf"
    doc.save(str(pdf_path))
    doc.close()
    return str(pdf_path)


@pytest.fixture
def sample_scanned_pdf(tmp_path: Path) -> str:
    """テキストなし（画像のみ想定）のテスト用 PDF を生成する。

    Args:
        tmp_path: 一時ディレクトリパス。

    Returns:
        str: 生成された PDF の絶対パス。
    """
    doc = fitz.open()
    doc.new_page(width=200, height=200)
    pdf_path = tmp_path / "scanned_like.pdf"
    doc.save(str(pdf_path))
    doc.close()
    return str(pdf_path)
