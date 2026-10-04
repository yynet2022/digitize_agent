"""PDF プリミティブ要素（テキスト・ベクター線）抽出ツールモジュール。

ラスター画像変換を行わず、デジタル PDF から直接テキスト境界ボックスや
罫線などのベクターパスを抽出します。
"""

from pathlib import Path
from typing import Any

import pdfplumber
import pymupdf as fitz
from pydantic import BaseModel, Field


class InspectPdfPrimitivesInput(BaseModel):
    """inspect_pdf_primitives 関数の入力バリデーションモデル。"""

    pdf_path: str = Field(description="Path to the PDF file.")
    page_number: int = Field(default=0, ge=0, description="0-indexed page.")
    bbox_filter: list[float] | None = Field(
        default=None,
        min_length=4,
        max_length=4,
        description="Optional [x0, y0, x1, y1] bounding box in points.",
    )


def _is_bbox_inside(
    target: tuple[float, float, float, float] | list[float],
    bbox_filter: list[float],
) -> bool:
    """要素のバウンディングボックスがフィルタ枠内にあるか判定する。

    Args:
        target: 対象要素の [x0, y0, x1, y1]。
        bbox_filter: フィルタ枠 [fx0, fy0, fx1, fy1]。

    Returns:
        bool: フィルタ枠と交差または包含されていれば True。
    """
    tx0, ty0, tx1, ty1 = target
    fx0, fy0, fx1, fy1 = bbox_filter
    return not (tx1 < fx0 or tx0 > fx1 or ty1 < fy0 or ty0 > fy1)


def inspect_pdf_primitives(
    pdf_path: str,
    page_number: int = 0,
    bbox_filter: list[float] | None = None,
) -> dict[str, Any]:
    """PDF から直接テキスト要素とベクター罫線を抽出する。

    Args:
        pdf_path: 対象 PDF ファイルのパス。
        page_number: 0 から始まる対象ページ番号。
        bbox_filter: [x0, y0, x1, y1] で指定する抽出範囲フィルタ。

    Returns:
        dict[str, Any]: 抽出結果辞書、またはエラー情報辞書。
    """
    try:
        validated = InspectPdfPrimitivesInput(
            pdf_path=pdf_path,
            page_number=page_number,
            bbox_filter=bbox_filter,
        )
    except Exception as exc:
        return {"status": "error", "message": f"Validation error: {exc}"}

    path_obj = Path(validated.pdf_path)
    if not path_obj.is_file():
        return {
            "status": "error",
            "message": f"PDF file not found: {validated.pdf_path}",
        }

    text_elements: list[dict[str, Any]] = []
    vector_lines: list[dict[str, Any]] = []

    try:
        # PyMuPDF によるテキストスパンの抽出
        with fitz.open(str(path_obj)) as doc:
            if validated.page_number >= len(doc):
                return {
                    "status": "error",
                    "message": (
                        f"Page number {validated.page_number} is out of "
                        f"range (total pages: {len(doc)})."
                    ),
                }

            page = doc[validated.page_number]
            text_dict = page.get_text("dict")

            for block in text_dict.get("blocks", []):
                if "lines" not in block:
                    continue
                for line in block["lines"]:
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if not text:
                            continue
                        bbox = [float(v) for v in span.get("bbox", [])]
                        if len(bbox) != 4:
                            continue

                        if validated.bbox_filter and not _is_bbox_inside(
                            bbox, validated.bbox_filter
                        ):
                            continue

                        text_elements.append(
                            {
                                "text": text,
                                "bbox": bbox,
                                "font_size": float(span.get("size", 0.0)),
                            }
                        )

        # pdfplumber によるベクター線分の抽出
        with pdfplumber.open(str(path_obj)) as plum_pdf:
            if validated.page_number < len(plum_pdf.pages):
                plum_page = plum_pdf.pages[validated.page_number]
                for line in plum_page.lines:
                    coords = [
                        float(line["x0"]),
                        float(line["top"]),
                        float(line["x1"]),
                        float(line["bottom"]),
                    ]
                    if validated.bbox_filter and not _is_bbox_inside(
                        coords, validated.bbox_filter
                    ):
                        continue
                    width = float(line.get("width", 1.0))
                    vector_lines.append({"coords": coords, "width": width})

        is_scanned = len(text_elements) == 0

        return {
            "is_scanned": is_scanned,
            "text_elements": text_elements,
            "vector_lines": vector_lines,
        }

    except Exception as exc:
        return {
            "status": "error",
            "message": f"Failed to extract PDF primitives: {exc}",
        }


class SearchPdfPrimitivesInput(BaseModel):
    """search_pdf_primitives 関数の入力バリデーションモデル。"""

    pdf_path: str = Field(description="Path to the PDF file.")
    query: str = Field(
        description="Search text query (e.g. 'Figure 5', 'Fig.')."
    )
    case_sensitive: bool = Field(
        default=False, description="Whether search is case-sensitive."
    )
    max_matches: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum number of matches to return.",
    )


def search_pdf_primitives(
    pdf_path: str,
    query: str,
    case_sensitive: bool = False,
    max_matches: int = 50,
) -> dict[str, Any]:
    """PDF 全ページから指定キーワードを検索し、出現ページと領域を返す。

    Args:
        pdf_path: 対象 PDF ファイルのパス。
        query: 検索対象テキスト（例: 'Figure 5', 'Table 1'）。
        case_sensitive: 大文字小文字を区別するかどうか。
        max_matches: 返却する最大一致件数。

    Returns:
        dict[str, Any]: 一致リスト（ページ番号、bbox、テキスト文脈）。
    """
    try:
        validated = SearchPdfPrimitivesInput(
            pdf_path=pdf_path,
            query=query,
            case_sensitive=case_sensitive,
            max_matches=max_matches,
        )
    except Exception as exc:
        return {"status": "error", "message": f"Validation error: {exc}"}

    path_obj = Path(validated.pdf_path)
    if not path_obj.is_file():
        return {
            "status": "error",
            "message": f"PDF file not found: {validated.pdf_path}",
        }

    matches: list[dict[str, Any]] = []

    try:
        with fitz.open(str(path_obj)) as doc:
            for page_idx, page in enumerate(doc):
                if len(matches) >= validated.max_matches:
                    break

                # 精密矩形の検索
                flags = 0 if validated.case_sensitive else 2
                rects = page.search_for(validated.query, flags=flags)
                if not rects:
                    continue

                # 該当ページのテキストブロックから文脈スニペットを特定
                blocks = page.get_text("blocks")
                for r in rects:
                    if len(matches) >= validated.max_matches:
                        break

                    matched_bbox = [
                        round(float(r.x0), 2),
                        round(float(r.y0), 2),
                        round(float(r.x1), 2),
                        round(float(r.y1), 2),
                    ]

                    snippet = ""
                    block_bbox: list[float] | None = None
                    for b in blocks:
                        bx0, by0, bx1, by1 = b[0], b[1], b[2], b[3]
                        btext = b[4]
                        if not (
                            r.x1 < bx0
                            or r.x0 > bx1
                            or r.y1 < by0
                            or r.y0 > by1
                        ):
                            snippet = " ".join(btext.split())
                            block_bbox = [
                                round(float(bx0), 2),
                                round(float(by0), 2),
                                round(float(bx1), 2),
                                round(float(by1), 2),
                            ]
                            break

                    matches.append(
                        {
                            "page_number": page_idx,
                            "bbox": matched_bbox,
                            "block_bbox": block_bbox or matched_bbox,
                            "snippet": snippet,
                        }
                    )

        return {
            "status": "success",
            "query": validated.query,
            "total_matches": len(matches),
            "matches": matches,
        }

    except Exception as exc:
        return {
            "status": "error",
            "message": f"Search in PDF failed: {exc}",
        }
