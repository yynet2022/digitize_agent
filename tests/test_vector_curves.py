"""PDF ベクター曲線抽出ツールの単体テストモジュール。

3次ベジェ曲線のサンプリング評価、セグメントの連結処理、
PDF からの座標抽出および DPI・クロップ変換を検証します。
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pymupdf as fitz

from digitize_agent.tools.vector_curves import (
    _calc_segment_length,
    _chain_segments,
    _color_matches,
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


def test_extract_vector_curve_drawing_indices(tmp_path: Path) -> None:
    """drawing_indices リストによる複数描画の一括抽出を検証。"""
    doc = fitz.open()
    page = doc.new_page(width=200, height=200)

    s0 = page.new_shape()
    s0.draw_line(fitz.Point(10, 10), fitz.Point(50, 10))
    s0.finish(color=(1, 0, 0), width=1)
    s0.commit()

    s1 = page.new_shape()
    s1.draw_line(fitz.Point(50, 10), fitz.Point(90, 10))
    s1.finish(color=(0, 0, 1), width=1)
    s1.commit()

    pdf_path = tmp_path / "indices_test.pdf"
    doc.save(str(pdf_path))
    doc.close()

    res = extract_vector_curve_points(
        pdf_path=str(pdf_path),
        page_number=0,
        curve_types=["l"],
        drawing_indices=[0, 1],
    )
    assert res.get("status") == "success"
    assert res["total_curves"] == 2


def test_extract_vector_curve_group_by_color(tmp_path: Path) -> None:
    """group_by_color=True により同色描画が統合されることを検証。"""
    doc = fitz.open()
    page = doc.new_page(width=200, height=200)

    s0 = page.new_shape()
    s0.draw_line(fitz.Point(10, 20), fitz.Point(50, 20))
    s0.finish(color=(0, 0, 1), width=1)
    s0.commit()

    s1 = page.new_shape()
    s1.draw_line(fitz.Point(50, 20), fitz.Point(100, 20))
    s1.finish(color=(0, 0, 1), width=1)
    s1.commit()

    pdf_path = tmp_path / "group_color_test.pdf"
    doc.save(str(pdf_path))
    doc.close()

    res = extract_vector_curve_points(
        pdf_path=str(pdf_path),
        page_number=0,
        curve_types=["l"],
        drawing_indices=[0, 1],
        group_by_color=True,
    )
    assert res.get("status") == "success"
    assert res["total_groups"] == 1
    group = res["color_grouped_curves"][0]
    assert group["points"][0][0] == 10.0
    assert group["points"][-1][0] == 100.0


def test_vector_curves_helpers_and_chain_modes() -> None:
    """ヘルパー関数と _chain_segments の全連結モードを検証する。"""
    assert _color_matches(None, [1.0, 0.0, 0.0]) is False
    assert _color_matches([1.0, 0.0], [1.0, 0.0, 0.0]) is False
    assert _color_matches([255, 0, 0], [1.0, 0.0, 0.0], tolerance=0.1) is True

    assert _calc_segment_length([]) == 0.0
    assert _calc_segment_length([[0.0, 0.0]]) == 0.0
    assert abs(_calc_segment_length([[0.0, 0.0], [3.0, 4.0]]) - 5.0) < 1e-6

    # append_reverse: curr=[0->5], cand=[10->5]
    c_app_rev = _chain_segments(
        [
            {"item_index": 0, "points": [[0.0, 0.0], [5.0, 0.0]]},
            {"item_index": 1, "points": [[10.0, 0.0], [5.0, 0.0]]},
        ],
        tolerance=1.0,
    )
    assert len(c_app_rev) == 1
    assert c_app_rev[0]["points"] == [[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]]

    # prepend_reverse: curr=[5->10], cand=[5->0]
    c_pre_rev = _chain_segments(
        [
            {"item_index": 0, "points": [[5.0, 0.0], [10.0, 0.0]]},
            {"item_index": 1, "points": [[5.0, 0.0], [0.0, 0.0]]},
        ],
        tolerance=1.0,
    )
    assert len(c_pre_rev) == 1
    assert c_pre_rev[0]["points"] == [[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]]

    # prepend_normal: curr=[5->10], cand=[0->5], sort_x_ascending=False
    c_pre_norm = _chain_segments(
        [
            {"item_index": 0, "points": [[5.0, 0.0], [10.0, 0.0]]},
            {"item_index": 1, "points": [[0.0, 0.0], [5.0, 0.0]]},
        ],
        tolerance=1.0,
        sort_x_ascending=False,
    )
    assert len(c_pre_norm) == 1
    assert c_pre_norm[0]["points"] == [[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]]

    # ヘアピン折り返し（内積 < 0）の場合は連結されないこと
    c_hairpin = _chain_segments(
        [
            {"item_index": 0, "points": [[0.0, 0.0], [5.0, 0.0]]},
            {"item_index": 1, "points": [[0.0, 0.0], [5.0, 0.0]]},
        ],
        tolerance=1.0,
    )
    assert len(c_hairpin) == 2


def test_extract_vector_curve_points_edge_and_error_branches(
    tmp_path: Path,
) -> None:
    """extract_vector_curve_points のエッジケース・エラー分岐を検証する。"""
    # 1. バリデーションエラー
    res_val = extract_vector_curve_points(
        pdf_path="dummy.pdf", num_samples_per_segment=1
    )
    assert res_val["status"] == "error"
    assert "Validation error" in res_val["message"]

    # 2. ベクター描画が1つもないページ
    doc_empty = fitz.open()
    doc_empty.new_page(width=100, height=100)
    empty_pdf = tmp_path / "no_drawings.pdf"
    doc_empty.save(str(empty_pdf))
    doc_empty.close()

    res_no_dwg = extract_vector_curve_points(pdf_path=str(empty_pdf))
    assert res_no_dwg["status"] == "error"
    assert "No vector drawings found" in res_no_dwg["message"]

    # 3. 描画はあるが指定インデックスが範囲外、または条件不一致
    doc = fitz.open()
    page = doc.new_page(width=200, height=200)
    sh = page.new_shape()
    sh.draw_line(fitz.Point(10, 10), fitz.Point(50, 10))
    sh.draw_line(fitz.Point(50, 10), fitz.Point(90, 10))
    sh.finish(color=(0, 0, 1), width=1.0)
    sh.commit()
    line_pdf = tmp_path / "line_only.pdf"
    doc.save(str(line_pdf))
    doc.close()

    # 無効な drawing_indices
    res_bad_indices = extract_vector_curve_points(
        pdf_path=str(line_pdf), drawing_indices=[99, 100]
    )
    assert res_bad_indices["status"] == "error"
    assert (
        "None of the specified drawing_indices" in res_bad_indices["message"]
    )

    # 無効な drawing_index
    res_bad_idx = extract_vector_curve_points(
        pdf_path=str(line_pdf), drawing_index=99
    )
    assert res_bad_idx["status"] == "error"
    assert "out of range" in res_bad_idx["message"]

    # bbox_filter で範囲外になり一致描画が0件になる場合
    res_bbox_miss = extract_vector_curve_points(
        pdf_path=str(line_pdf),
        curve_types=["l"],
        bbox_filter=[150.0, 150.0, 190.0, 190.0],
    )
    assert res_bbox_miss["status"] == "error"
    assert (
        "No drawings containing requested curves" in res_bbox_miss["message"]
    )

    # drawing_index=0 を明示したが curve_types=['c'] (線分しかないので抽出0件)
    res_no_items = extract_vector_curve_points(
        pdf_path=str(line_pdf),
        drawing_index=0,
        curve_types=["c"],
    )
    assert res_no_items["status"] == "error"
    assert "No curve or line segments extracted" in res_no_items["message"]

    # curve_segments と item_indices の明示指定（未存在グループ含む）
    res_segs = extract_vector_curve_points(
        pdf_path=str(line_pdf),
        drawing_index=0,
        curve_types=["l"],
        item_indices=[0, 1],
        curve_segments=[[0], [1], [99]],
    )
    assert res_segs["status"] == "success"
    assert res_segs["total_curves"] == 2

    # group_by_color / extract_all_matching / bbox_filter の残り分岐
    fake_p0 = MagicMock(x=10.0, y=20.0)
    fake_p1 = MagicMock(x=50.0, y=20.0)
    fake_rect = fitz.Rect(0.0, 0.0, 100.0, 100.0)
    fake_page = MagicMock()
    fake_page.get_drawings.return_value = [
        {"color": None, "rect": None, "items": [("re", fake_p0)]},
        {
            "color": None,
            "rect": fake_rect,
            "items": [("l", fake_p0, fake_p1), ("re", fake_p0)],
        },
        {
            "color": (1.0, 0.0, 0.0),
            "rect": fake_rect,
            "items": [("l", fake_p0, fake_p0)],
        },
    ]
    fake_doc = MagicMock()
    fake_doc.__enter__.return_value = [fake_page]
    with patch("pymupdf.open", return_value=fake_doc):
        res_none_col = extract_vector_curve_points(
            pdf_path=str(line_pdf),
            curve_types=["l", "re"],
            bbox_filter=[0.0, 0.0, 100.0, 100.0],
            min_length=1.0,
            sort_x_ascending=False,
            group_by_color=True,
        )
        assert res_none_col["status"] == "success"
        assert res_none_col["color_grouped_curves"][0]["color_hex"] is None

        res_all_match = extract_vector_curve_points(
            pdf_path=str(line_pdf),
            curve_types=["l"],
            min_length=1.0,
            extract_all_matching=True,
        )
        assert res_all_match["status"] == "success"
        assert res_all_match["total_drawings"] == 1

    with patch("pymupdf.open", side_effect=RuntimeError("fitz failure")):
        res_exc = extract_vector_curve_points(pdf_path=str(line_pdf))
        assert res_exc["status"] == "error"
        assert "Failed to extract vector curve points" in res_exc["message"]
