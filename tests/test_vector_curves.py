"""PDF ベクター曲線抽出ツールの単体テストモジュール。

3次ベジェ曲線のサンプリング評価、セグメントの連結処理、
PDF からの座標抽出および DPI・クロップ変換を検証します。
"""

from pathlib import Path

import pymupdf as fitz
import pytest

from digitize_agent.tools.vector_curves import (
    _chain_segments,
    _eval_cubic_bezier,
    extract_vector_curve_points,
)


def test_eval_cubic_bezier() -> None:
    """3次ベジェ曲線が始点・終点を通り指定点数でサンプリングされることを検証。"""
    p0 = (0.0, 0.0)
    p1 = (0.0, 10.0)
    p2 = (10.0, 10.0)
    p3 = (10.0, 0.0)

    pts = _eval_cubic_bezier(p0, p1, p2, p3, num_samples=5)
    assert len(pts) == 5
    assert pts[0] == [0.0, 0.0]
    assert pts[-1] == [10.0, 0.0]
    # 対称性より中点 (t=0.5) の x=5.0
    assert abs(pts[2][0] - 5.0) < 1e-6


def test_chain_segments() -> None:
    """近接端点を持つセグメントが自動連結されX昇順に整列されることを検証。"""
    # 2つのセグメント: [10, 0] -> [5, 5] と [5, 5] -> [0, 0] (逆順で配置)
    seg1 = {"item_index": 0, "points": [[10.0, 0.0], [5.0, 5.0]]}
    seg2 = {"item_index": 1, "points": [[5.0, 5.0], [0.0, 0.0]]}

    chains = _chain_segments(
        [seg1, seg2], tolerance=1.0, sort_x_ascending=True
    )
    assert len(chains) == 1
    # X 昇順なので [0, 0] から始まり [10, 0] で終わる
    assert chains[0]["points"][0][0] == 0.0
    assert chains[0]["points"][-1][0] == 10.0
    assert len(chains[0]["points"]) == 3  # [0,0], [5,5], [10,0]


def test_extract_vector_curve_points_file_not_found() -> None:
    """存在しない PDF を指定した際にエラーが返ることを検証。"""
    res = extract_vector_curve_points(pdf_path="non_existent.pdf")
    assert res["status"] == "error"
    assert "not found" in res["message"]


def test_extract_vector_curve_points_invalid_page(tmp_path: Path) -> None:
    """ページ範囲外を指定した際にエラーが返ることを検証。"""
    doc = fitz.open()
    doc.new_page(width=100, height=100)
    pdf_path = tmp_path / "single_page.pdf"
    doc.save(str(pdf_path))
    doc.close()

    res = extract_vector_curve_points(pdf_path=str(pdf_path), page_number=5)
    assert res["status"] == "error"
    assert "out of range" in res["message"]


def test_extract_vector_curve_points_synthetic_pdf(tmp_path: Path) -> None:
    """合成 PDF からベジェ曲線が正常に抽出・スケーリングされることを検証。"""
    doc = fitz.open()
    page = doc.new_page(width=200, height=200)
    shape = page.new_shape()
    shape.draw_bezier(
        fitz.Point(10, 10),
        fitz.Point(10, 50),
        fitz.Point(50, 50),
        fitz.Point(50, 10),
    )
    shape.finish(color=(0, 0, 0), width=1.0)
    shape.commit()

    pdf_path = tmp_path / "bezier_test.pdf"
    doc.save(str(pdf_path))
    doc.close()

    # pt 単位での抽出
    res = extract_vector_curve_points(
        pdf_path=str(pdf_path),
        page_number=0,
        num_samples_per_segment=10,
    )
    assert res["status"] == "success"
    assert res["total_curves"] == 1
    assert res["curves"][0]["num_points"] == 10

    # DPI スケーリングとクロップ座標適用の検証
    # DPI=144 (scale=2.0), crop_bbox=[10, 10, 100, 100]
    res_scaled = extract_vector_curve_points(
        pdf_path=str(pdf_path),
        page_number=0,
        num_samples_per_segment=10,
        dpi=144.0,
        crop_bbox_pixels=[10.0, 10.0, 100.0, 100.0],
    )
    assert res_scaled["status"] == "success"
    first_pt = res_scaled["curves"][0]["points"][0]
    # (10 pt * (144/72)) - 10 = 10 px
    assert abs(first_pt[0] - 10.0) < 1e-3


def test_extract_vector_curve_points_real_pdf() -> None:
    """a.pdf の図2.4から3本の特性曲線が抽出できることを検証。"""
    pdf_path = Path("a.pdf")
    if not pdf_path.exists():
        pytest.skip("a.pdf is not available in workspace")

    res = extract_vector_curve_points(
        pdf_path=str(pdf_path),
        page_number=1,
        drawing_index=31,
        item_indices=[16, 17, 18, 19, 20],
        num_samples_per_segment=50,
        dpi=300.0,
        crop_bbox_pixels=[890.0, 225.0, 1610.0, 945.0],
    )
    assert res["status"] == "success"
    # 3本の連続曲線 (V_GS 大, 中, 小) に自動連結されること
    assert res["total_curves"] == 3


def test_vector_curves_min_length_and_color(tmp_path: Path) -> None:
    """min_length による微小目盛り線除外と stroke_color 絞り込みを検証。"""
    import pymupdf as fitz

    doc = fitz.open()
    page = doc.new_page(width=300, height=300)

    # 1. 短い目盛り線 (黒、長さ 4 pt)
    s1 = page.new_shape()
    s1.draw_line(fitz.Point(10, 10), fitz.Point(14, 10))
    s1.finish(color=(0, 0, 0), width=1.0)
    s1.commit()

    # 2. 長い主曲線 (青、長さ 80 pt)
    s2 = page.new_shape()
    s2.draw_line(fitz.Point(20, 20), fitz.Point(100, 20))
    s2.finish(color=(0, 0, 1), width=1.0)
    s2.commit()

    pdf_path = tmp_path / "filter_test.pdf"
    doc.save(str(pdf_path))
    doc.close()

    # min_length=10.0 で短い目盛り線がスキップされること
    res_len = extract_vector_curve_points(
        pdf_path=str(pdf_path),
        page_number=0,
        curve_types=["l"],
        min_length=10.0,
    )
    assert res_len.get("status") == "success"
    assert res_len["total_curves"] >= 1
    # 抽出されたのは長い主曲線 (長さ 80)
    assert all(
        abs(c["bounds"][2] - c["bounds"][0] - 80) < 1e-3
        for c in res_len["curves"]
    )

    # stroke_color=[0, 0, 1] で青色のみ抽出されること
    res_col = extract_vector_curve_points(
        pdf_path=str(pdf_path),
        page_number=0,
        curve_types=["l"],
        stroke_color=[0.0, 0.0, 1.0],
    )
    assert res_col.get("status") == "success"
    assert res_col["total_curves"] >= 1
    assert all(
        abs(c["bounds"][2] - c["bounds"][0] - 80) < 1e-3
        for c in res_col["curves"]
    )

    # extract_all_matching=True で全描画リストが返ること
    res_all = extract_vector_curve_points(
        pdf_path=str(pdf_path),
        page_number=0,
        curve_types=["l"],
        extract_all_matching=True,
    )
    assert res_all.get("status") == "success"
    assert res_all["total_drawings"] >= 2
