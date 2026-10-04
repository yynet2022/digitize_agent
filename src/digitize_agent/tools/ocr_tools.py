"""領域 OCR 文字認識ツールモジュール。

切り出し画像スニペットから軸ラベル、目盛り数値、表セル文字列などの
テキストおよび認識信頼度スコアを抽出します。
"""

import shutil
from pathlib import Path
from typing import Any

import pytesseract
from PIL import Image
from pydantic import BaseModel, Field
from pytesseract import Output


class OcrRegionTextInput(BaseModel):
    """ocr_region_text 関数の入力バリデーションモデル。"""

    image_path: str = Field(description="Path to the image snippet file.")
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


def ocr_region_text(
    image_path: str,
    psm: int = 6,
    whitelist: str | None = None,
) -> dict[str, Any]:
    """画像スニペットに対して Tesseract OCR を実行し認識結果を返す。

    Args:
        image_path: 入力画像のファイルパス。
        psm: Tesseract PSM モード番号。
        whitelist: 認識対象を限定する文字ホワイトリスト。

    Returns:
        dict[str, Any]: 生テキスト、整形テキスト、信頼度スコア、またはエラー。
    """
    try:
        validated = OcrRegionTextInput(
            image_path=image_path,
            psm=psm,
            whitelist=whitelist,
        )
    except Exception as exc:
        return {"status": "error", "message": f"Validation error: {exc}"}

    img_path = Path(validated.image_path)
    if not img_path.is_file():
        return {
            "status": "error",
            "message": f"Snippet image not found: {validated.image_path}",
        }

    if not _ensure_tesseract_configured():
        return {
            "status": "error",
            "message": (
                "Tesseract OCR is not installed or not found in system PATH. "
                "Please install tesseract-ocr on the system."
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
