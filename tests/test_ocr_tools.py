"""領域 OCR 文字認識ツールの単体テストモジュール。

ocr_region_text のテキスト抽出、信頼度集計、モック動作、
およびエラー耐性を検証します。
"""

from pathlib import Path
from unittest.mock import patch

from digitize_agent.tools.ocr_tools import (
    _ensure_tesseract_configured,
    _extract_text_from_pdf_region,
    ocr_region_text,
)


def test_ocr_region_text_mocked_success(
    sample_plot_image: str,
) -> None:
    """pytesseract の動作をモックしてパースおよび信頼度計算を検証する。"""
    mock_data = {
        "text": ["", "100.0", "", "200.0"],
        "conf": [-1, 95.0, -1, 85.0],
    }

    with (
        patch(
            "digitize_agent.tools.ocr_tools._ensure_tesseract_configured",
            return_value=True,
        ),
        patch("pytesseract.image_to_data", return_value=mock_data),
        patch("pytesseract.image_to_string", return_value="100.0 200.0\n"),
    ):
        res = ocr_region_text(
            image_path=sample_plot_image,
            psm=6,
            whitelist="0123456789.",
        )

        assert res.get("status") != "error"
        assert res["text"] == "100.0 200.0"
        assert res["clean_text"] == "100.0 200.0"
        # 信頼度平均: (95.0 + 85.0) / 2 = 90.0
        assert res["confidence"] == 90.0


def test_ocr_region_text_file_not_found(temp_dir: Path) -> None:
    """存在しない画像パスを指定した際にエラーを返すことを検証する。"""
    res = ocr_region_text(image_path=str(temp_dir / "missing.png"))

    assert res["status"] == "error"
    assert "not found" in res["message"]


def test_ocr_region_text_invalid_psm(sample_plot_image: str) -> None:
    """不正な PSM 値を指定した際にバリデーションエラーを返すことを検証する。"""
    res = ocr_region_text(image_path=sample_plot_image, psm=99)

    assert res["status"] == "error"
    assert "Validation error" in res["message"]


def test_ocr_not_installed_error(sample_plot_image: str) -> None:
    """Tesseract が未導入の環境下で
    安全にエラーメッセージを返すことを検証する。
    """
    with patch(
        "digitize_agent.tools.ocr_tools._ensure_tesseract_configured",
        return_value=False,
    ):
        res = ocr_region_text(image_path=sample_plot_image)

        assert res["status"] == "error"
        assert "Tesseract OCR is not installed" in res["message"]


def test_ocr_pdf_fallback(sample_pdf_with_text_and_lines: str) -> None:
    """Tesseract 未導入時でも pdf_path 指定で文字抽出ができることを検証。"""
    with patch(
        "digitize_agent.tools.ocr_tools._ensure_tesseract_configured",
        return_value=False,
    ):
        res = ocr_region_text(
            pdf_path=sample_pdf_with_text_and_lines,
            page_number=0,
            bbox=[40.0, 40.0, 200.0, 60.0],
        )

        assert res.get("status") != "error"
        assert "Sample Plot" in res["text"]
        assert res["confidence"] == 100.0


def test_ensure_tesseract_configured_branches() -> None:
    """_ensure_tesseract_configured の PATH 検出・候補探索を検証する。"""
    with patch("shutil.which", return_value="C:/tesseract.exe"):
        assert _ensure_tesseract_configured() is True

    with (
        patch("shutil.which", return_value=None),
        patch("pathlib.Path.is_file", return_value=True),
    ):
        assert _ensure_tesseract_configured() is True

    with (
        patch("shutil.which", return_value=None),
        patch("pathlib.Path.is_file", return_value=False),
    ):
        assert _ensure_tesseract_configured() is False


def test_extract_text_from_pdf_region_branches(
    sample_pdf_with_text_and_lines: str,
) -> None:
    """PDF 領域抽出のページ範囲外・bbox 省略・例外処理を検証する。"""
    # ページ範囲外
    res_out = _extract_text_from_pdf_region(
        pdf_path=sample_pdf_with_text_and_lines,
        page_number=99,
    )
    assert res_out["status"] == "error"
    assert "out of range" in res_out["message"]

    # bbox=None でページ全体抽出
    res_full = _extract_text_from_pdf_region(
        pdf_path=sample_pdf_with_text_and_lines,
        page_number=0,
        bbox=None,
    )
    assert "Sample Plot" in res_full["text"]

    # 存在しないファイルで例外発生
    res_err = _extract_text_from_pdf_region(
        pdf_path="non_existent_pdf.pdf",
        page_number=0,
    )
    assert res_err["status"] == "error"
    assert "PDF text fallback failed" in res_err["message"]


def test_ocr_region_text_additional_branches(
    sample_plot_image: str,
    sample_pdf_with_text_and_lines: str,
) -> None:
    """ocr_region_text の未指定エラー・フォールバック・例外分岐を検証。"""
    # image_path も pdf_path も未指定
    res_none = ocr_region_text()
    assert res_none["status"] == "error"
    assert "Either image_path or valid pdf_path" in res_none["message"]

    # image_path と pdf_path 両方指定時に 1回目のチェックは True、2回目で False
    with patch(
        "digitize_agent.tools.ocr_tools._ensure_tesseract_configured",
        side_effect=[True, False],
    ):
        res_fb2 = ocr_region_text(
            image_path=sample_plot_image,
            pdf_path=sample_pdf_with_text_and_lines,
            page_number=0,
        )
        assert "Sample Plot" in res_fb2["text"]

    # OCR 実行中の予期せぬ例外
    with (
        patch(
            "digitize_agent.tools.ocr_tools._ensure_tesseract_configured",
            return_value=True,
        ),
        patch("PIL.Image.open", side_effect=RuntimeError("corrupt image")),
    ):
        res_exc = ocr_region_text(image_path=sample_plot_image)
        assert res_exc["status"] == "error"
        assert "OCR processing failed" in res_exc["message"]

    # 存在しない pdf_path と whitelist=None での OCR 実行
    mock_empty_data = {"text": [], "conf": []}
    with (
        patch(
            "digitize_agent.tools.ocr_tools._ensure_tesseract_configured",
            return_value=True,
        ),
        patch("pytesseract.image_to_data", return_value=mock_empty_data),
        patch("pytesseract.image_to_string", return_value=""),
    ):
        res_no_wl = ocr_region_text(
            image_path=sample_plot_image,
            pdf_path="non_existent_for_ocr.pdf",
            whitelist=None,
        )
        assert res_no_wl["confidence"] == 0.0
