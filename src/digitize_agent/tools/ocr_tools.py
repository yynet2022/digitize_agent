"""領域 OCR 文字認識ツールモジュール。

切り出し画像スニペットから軸ラベル、目盛り数値、表セル文字列などの
テキストおよび認識信頼度スコアを抽出します。
"""

import shutil
from pathlib import Path
from typing import Any

import pymupdf as fitz
import pytesseract
from PIL import Image
from pydantic import BaseModel, Field
from pytesseract import Output


class OcrRegionTextInput(BaseModel):
    """ocr_region_text 関数の入力バリデーションモデル。"""

    image_path: str | None = Field(
        default=None, description="Path to the image snippet file."
    )
    pdf_path: str | None = Field(
        default=None,
        description="Optional PDF file path for native vector text fallback.",
    )
    page_number: int = Field(default=0, ge=0, description="0-indexed page.")
    bbox: list[float] | None = Field(
        default=None,
        min_length=4,
        max_length=4,
        description="Bounding box [x0, y0, x1, y1] in points.",
    )
    psm: int = Field(
        default=6,
        ge=0,
        le=13,
        description="Tesseract Page Segmentation Mode (PSM).",
    )
    whitelist: str | None = Field(
        default=None,
        description="Optional character whitelist for OCR.",
    )


def _ensure_tesseract_configured() -> bool:
    """Tesseract 実行コマンドの配置を確認し必要に応じて設定する。

    Returns:
        bool: tesseract コマンドが利用可能であれば True。
    """
    if shutil.which("tesseract"):
        return True

    # Windows 環境の代表的なデフォルトインストール先を確認
    candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for cand in candidates:
        if Path(cand).is_file():
            pytesseract.pytesseract.tesseract_cmd = cand
            return True

    return False


def _extract_text_from_pdf_region(
    pdf_path: str,
    page_number: int,
    bbox: list[float] | None = None,
) -> dict[str, Any]:
    """PDF 内部のテキストオブジェクトから指定矩形領域内の文字を抽出する。"""
    try:
        with fitz.open(pdf_path) as doc:
            if page_number >= len(doc):
                return {
                    "status": "error",
                    "message": f"Page number {page_number} out of range.",
                }
            page = doc[page_number]
            if bbox:
                rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
                text = page.get_text("text", clip=rect).strip()
            else:
                text = page.get_text("text").strip()

            clean_text = " ".join(text.split())
            return {
                "text": text,
                "clean_text": clean_text,
                "confidence": 100.0,
            }
    except Exception as exc:
        return {
            "status": "error",
            "message": f"PDF text fallback failed: {exc}",
        }


def ocr_region_text(
    image_path: str | None = None,
    pdf_path: str | None = None,
    page_number: int = 0,
    bbox: list[float] | None = None,
    psm: int = 6,
    whitelist: str | None = None,
) -> dict[str, Any]:
    """画像スニペットまたは PDF 領域からテキストを認識・抽出する。

    Args:
        image_path: 入力画像のファイルパス。
        pdf_path: フォールバックまたは直接取得用 PDF パス。
        page_number: PDF 対象ページ番号。
        bbox: PDF 内の抽出矩形 [x0, y0, x1, y1] (pt単位)。
        psm: Tesseract PSM モード番号。
        whitelist: 認識対象を限定する文字ホワイトリスト。

    Returns:
        dict[str, Any]: 生テキスト、整形テキスト、信頼度スコア、またはエラー。
    """
    try:
        validated = OcrRegionTextInput(
            image_path=image_path,
            pdf_path=pdf_path,
            page_number=page_number,
            bbox=bbox,
            psm=psm,
            whitelist=whitelist,
        )
    except Exception as exc:
        return {"status": "error", "message": f"Validation error: {exc}"}

    # PDF パスが指定されており、PDF からの直接テキスト抽出が
    # 優先または可能な場合
    if validated.pdf_path is not None:
        p_obj = Path(validated.pdf_path)
        if p_obj.is_file():
            # Tesseract 未設定、または画像未指定の場合はフォールバック
            if not _ensure_tesseract_configured() or not validated.image_path:
                return _extract_text_from_pdf_region(
                    pdf_path=str(p_obj),
                    page_number=validated.page_number,
                    bbox=validated.bbox,
                )

    if not validated.image_path:
        return {
            "status": "error",
            "message": "Either image_path or valid pdf_path must be provided.",
        }

    img_path = Path(validated.image_path)
    if not img_path.is_file():
        return {
            "status": "error",
            "message": f"Snippet image not found: {validated.image_path}",
        }

    if not _ensure_tesseract_configured():
        # PDF があればフォールバックを試行
        if validated.pdf_path and Path(validated.pdf_path).is_file():
            return _extract_text_from_pdf_region(
                pdf_path=validated.pdf_path,
                page_number=validated.page_number,
                bbox=validated.bbox,
            )
        return {
            "status": "error",
            "message": (
                "Tesseract OCR is not installed or not found in system PATH. "
                "Please install tesseract-ocr or provide pdf_path for "
                "vector text fallback."
            ),
        }

    try:
        image = Image.open(str(img_path))

        config_parts = [f"--psm {validated.psm}"]
        if validated.whitelist:
            config_parts.append(
                f"-c tessedit_char_whitelist={validated.whitelist}"
            )
        config = " ".join(config_parts)

        # 単語レベルの座標と確信度を取得
        data = pytesseract.image_to_data(
            image, config=config, output_type=Output.DICT
        )

        words: list[str] = []
        confidences: list[float] = []

        num_items = len(data.get("text", []))
        for i in range(num_items):
            word = str(data["text"][i]).strip()
            conf_val = float(data["conf"][i])
            if word and conf_val >= 0.0:
                words.append(word)
                confidences.append(conf_val)

        raw_text = pytesseract.image_to_string(image, config=config)
        clean_text = " ".join(words)
        mean_conf = (
            float(sum(confidences) / len(confidences)) if confidences else 0.0
        )

        return {
            "text": raw_text.strip(),
            "clean_text": clean_text,
            "confidence": round(mean_conf, 2),
        }

    except Exception as exc:
        return {
            "status": "error",
            "message": f"OCR processing failed: {exc}",
        }
