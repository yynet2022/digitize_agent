"""PDF プリミティブ抽出ツールの単体テストモジュール。

inspect_pdf_primitives の正常動作、フィルタリング、
およびエラーハンドリングを検証します。
"""

from pathlib import Path

from digitize_agent.tools.pdf_tools import (
    inspect_pdf_primitives,
    search_pdf_primitives,
)


def test_inspect_pdf_primitives_success(
    sample_pdf_with_text_and_lines: str,
) -> None:
    """埋め込みテキストおよびベクター線が正しく抽出されることを検証する。"""
    res = inspect_pdf_primitives(
        pdf_path=sample_pdf_with_text_and_lines, page_number=0
    )

    assert res.get("status") != "error"
    assert res["is_scanned"] is False
    assert len(res["text_elements"]) >= 2
    texts = [elem["text"] for elem in res["text_elements"]]
    assert any("Sample Plot" in t for t in texts)
    assert any("X-axis" in t for t in texts)
    assert len(res["vector_lines"]) >= 2


def test_inspect_pdf_primitives_scanned(sample_scanned_pdf: str) -> None:
    """テキストのない PDF において is_scanned が True になることを検証する。"""
    res = inspect_pdf_primitives(pdf_path=sample_scanned_pdf, page_number=0)

    assert res.get("status") != "error"
    assert res["is_scanned"] is True
    assert len(res["text_elements"]) == 0


def test_inspect_pdf_primitives_bbox_filter(
    sample_pdf_with_text_and_lines: str,
) -> None:
    """bbox_filter で指定した範囲外の要素が除外されることを検証する。"""
    # 上部タイトル付近のみを対象にするフィルタ
    res = inspect_pdf_primitives(
        pdf_path=sample_pdf_with_text_and_lines,
        page_number=0,
        bbox_filter=[40.0, 40.0, 200.0, 60.0],
    )

    assert res.get("status") != "error"
    texts = [elem["text"] for elem in res["text_elements"]]
    assert any("Sample Plot" in t for t in texts)
    # 範囲外の X-axis label (y=80) は除外されること
    assert not any("X-axis label" in t for t in texts)


def test_inspect_pdf_primitives_file_not_found(temp_dir: Path) -> None:
    """存在しない PDF を指定した際に安全にエラーを返すことを検証する。"""
    non_existent = str(temp_dir / "missing.pdf")
    res = inspect_pdf_primitives(pdf_path=non_existent)

    assert res["status"] == "error"
    assert "not found" in res["message"]


def test_inspect_pdf_primitives_page_out_of_range(
    sample_scanned_pdf: str,
) -> None:
    """範囲外のページ番号を指定した際にエラーを返すことを検証する。"""
    res = inspect_pdf_primitives(pdf_path=sample_scanned_pdf, page_number=99)

    assert res["status"] == "error"
    assert "out of range" in res["message"]


def test_inspect_pdf_primitives_invalid_filter(
    sample_scanned_pdf: str,
) -> None:
    """不正な bbox_filter を指定した際に
    バリデーションエラーを返すことを検証する。
    """
    # 要素数が 3 つしかない不正なフィルタ
    res = inspect_pdf_primitives(
        pdf_path=sample_scanned_pdf,
        bbox_filter=[10.0, 20.0, 30.0],
    )

    assert res["status"] == "error"
    assert "Validation error" in res["message"]


def test_search_pdf_primitives_success(
    sample_pdf_with_text_and_lines: str,
) -> None:
    """PDF 内のキーワード検索で該当ページと bbox が正しく返ることを検証。"""
    res = search_pdf_primitives(
        pdf_path=sample_pdf_with_text_and_lines,
        query="Sample Plot",
    )
    assert res.get("status") == "success"
    assert res["total_matches"] >= 1
    m0 = res["matches"][0]
    assert m0["page_number"] == 0
    assert len(m0["bbox"]) == 4
    assert "Sample Plot" in m0["snippet"]


def test_search_pdf_primitives_not_found(
    sample_pdf_with_text_and_lines: str,
) -> None:
    """ヒットしないキーワードを指定した際に 0 件を返すことを検証。"""
    res = search_pdf_primitives(
        pdf_path=sample_pdf_with_text_and_lines,
        query="NonExistentFigure999",
    )
    assert res.get("status") == "success"
    assert res["total_matches"] == 0
    assert len(res["matches"]) == 0


def test_search_pdf_primitives_missing_file(temp_dir: Path) -> None:
    """存在しない PDF ファイルを指定した際にエラーを返すことを検証。"""
    res = search_pdf_primitives(
        pdf_path=str(temp_dir / "missing.pdf"),
        query="test",
    )
    assert res["status"] == "error"
    assert "not found" in res["message"]
