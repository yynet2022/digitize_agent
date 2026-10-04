"""領域 OCR 文字認識ツールの単体テストモジュール。

ocr_region_text のテキスト抽出、信頼度集計、モック動作、
およびエラー耐性を検証します。
"""

from pathlib import Path
from unittest.mock import patch

from digitize_agent.tools.ocr_tools import ocr_region_text


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
