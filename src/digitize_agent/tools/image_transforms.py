"""画像領域の切り出しおよび前処理ツールモジュール。

画像ファイルまたは PDF ページから指定領域をクロップし、傾き（Deskew）検出・
補正、コントラスト強調（CLAHE）を行い、解析用サブ画像を生成します。
"""

from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import pymupdf as fitz
from pydantic import BaseModel, Field


class CropAndTransformInput(BaseModel):
    """crop_and_transform_region 関数の入力バリデーションモデル。"""

    image_path: str | None = Field(
        default=None,
        description="Path to the source image file.",
    )
    pdf_path: str | None = Field(
        default=None,
        description="Path to the source PDF file.",
    )
    page_number: int = Field(
        default=0,
        ge=0,
        description="0-indexed PDF page number.",
    )
    dpi: float = Field(
        default=300.0,
        gt=0,
        description="Rendering DPI when extracting from PDF.",
    )
    bbox: list[float] = Field(
        min_length=4,
        max_length=4,
        description="Bounding box [x_min, y_min, x_max, y_max].",
    )
    bbox_mode: Literal["pixel", "point"] = Field(
        default="pixel",
        description="Coordinate unit of bbox: 'pixel' or 'point' (72 DPI pt).",
    )
    deskew: bool = Field(default=False, description="Apply deskewing.")
    enhance_contrast: bool = Field(
        default=False, description="Apply CLAHE enhancement."
    )
    output_path: str = Field(
        description="Destination path for the cropped image."
    )


def _detect_and_correct_skew(
    image: np.ndarray,
) -> tuple[np.ndarray, float]:
    """ハフ変換により直線の傾き中央値を算出し画像を補正する。

    Args:
        image: 対象の入力画像 (BGR)。

    Returns:
        tuple[np.ndarray, float]: 補正後画像と検出された傾斜角度（度数法）。
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=80,
        minLineLength=30,
        maxLineGap=10,
    )

    if lines is None or len(lines) == 0:
        return image, 0.0

    angles: list[float] = []
    for line in lines:
        coords = np.asarray(line).ravel()
        if len(coords) < 4:
            continue
        x1, y1, x2, y2 = coords[:4]
        dx = x2 - x1
        dy = y2 - y1
        angle = np.degrees(np.arctan2(dy, dx))
        if -45.0 <= angle <= 45.0:
            angles.append(angle)

    if not angles:
        return image, 0.0

    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.5:
        return image, median_angle

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(
        image,
        rot_mat,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated, median_angle


def _apply_clahe(image: np.ndarray) -> np.ndarray:
    """LAB色空間のLチャンネルに対して適応的ヒストグラム均等化を行う。

    Args:
        image: 対象の入力画像 (BGR)。

    Returns:
        np.ndarray: コントラスト強調された画像 (BGR)。
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    merged_lab = cv2.merge([enhanced_l, a_channel, b_channel])
    return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)


def _load_image_from_pdf(
    pdf_path: str,
    page_number: int,
    dpi: float,
) -> np.ndarray:
    """PDF の指定ページを指定 DPI で画像としてレンダリングする。

    Args:
        pdf_path: 対象 PDF のファイルパス。
        page_number: 0 から始まるページ番号。
        dpi: レンダリング解像度。

    Returns:
        np.ndarray: レンダリングされた OpenCV BGR 画像。

    Raises:
        FileNotFoundError: PDF が存在しない場合。
        IndexError: ページ番号が範囲外の場合。
    """
    path_obj = Path(pdf_path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    with fitz.open(str(path_obj)) as doc:
        if page_number >= len(doc):
            raise IndexError(
                f"Page {page_number} out of range (total: {len(doc)})."
            )
        page = doc[page_number]
        pix = page.get_pixmap(dpi=int(round(dpi)))
        img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        if pix.n == 1:
            return cv2.cvtColor(img_data, cv2.COLOR_GRAY2BGR)
        if pix.n >= 3:
            return cv2.cvtColor(img_data[:, :, :3], cv2.COLOR_RGB2BGR)
        return img_data


def crop_and_transform_region(
    bbox: list[float],
    output_path: str,
    image_path: str | None = None,
    pdf_path: str | None = None,
    page_number: int = 0,
    dpi: float = 300.0,
    bbox_mode: Literal["pixel", "point"] = "pixel",
    deskew: bool = False,
    enhance_contrast: bool = False,
) -> dict[str, Any]:
    """指定バウンディングボックスの領域を切り出し、傾き補正や強調を行う。

    画像ファイル、または PDF ファイルから直接指定 DPI でレンダリングして
    指定領域をクロップ保存します。

    Args:
        bbox: [x_min, y_min, x_max, y_max] の切り出し範囲。
        output_path: 保存先のファイルパス。
        image_path: 元画像ファイルへのパス (pdf_path と排他または優先)。
        pdf_path: 元 PDF ファイルへのパス。
        page_number: PDF の対象ページ番号 (0-indexed)。
        dpi: PDF レンダリング解像度 (DPI)。
        bbox_mode: bbox の単位 ("pixel" または "point")。
        deskew: 傾き検出と補正を行うフラグ。
        enhance_contrast: CLAHE によるコントラスト強調フラグ。

    Returns:
        dict[str, Any]: 処理結果情報またはエラーメッセージ。
    """
    try:
        validated = CropAndTransformInput(
            image_path=image_path,
            pdf_path=pdf_path,
            page_number=page_number,
            dpi=dpi,
            bbox=bbox,
            bbox_mode=bbox_mode,
            deskew=deskew,
            enhance_contrast=enhance_contrast,
            output_path=output_path,
        )
    except Exception as exc:
        return {"status": "error", "message": f"Validation error: {exc}"}

    if not validated.image_path and not validated.pdf_path:
        return {
            "status": "error",
            "message": "Either image_path or pdf_path must be specified.",
        }

    try:
        if validated.pdf_path:
            image = _load_image_from_pdf(
                validated.pdf_path,
                validated.page_number,
                validated.dpi,
            )
        else:
            assert validated.image_path is not None
            img_path = Path(validated.image_path)
            if not img_path.is_file():
                return {
                    "status": "error",
                    "message": (
                        f"Source image not found: {validated.image_path}"
                    ),
                }
            image = cv2.imread(str(img_path))
            if image is None:
                return {
                    "status": "error",
                    "message": (
                        f"Failed to decode image: {validated.image_path}"
                    ),
                }

        img_h, img_w = image.shape[:2]

        # bbox_mode に応じたピクセル座標変換
        if validated.bbox_mode == "point":
            scale = validated.dpi / 72.0
            bx0 = int(round(validated.bbox[0] * scale))
            by0 = int(round(validated.bbox[1] * scale))
            bx1 = int(round(validated.bbox[2] * scale))
            by1 = int(round(validated.bbox[3] * scale))
        else:
            bx0 = int(round(validated.bbox[0]))
            by0 = int(round(validated.bbox[1]))
            bx1 = int(round(validated.bbox[2]))
            by1 = int(round(validated.bbox[3]))

        # 境界値のクリッピングと整合性チェック
        x_min = max(0, min(bx0, img_w - 1))
        x_max = max(x_min + 1, min(bx1, img_w))
        y_min = max(0, min(by0, img_h - 1))
        y_max = max(y_min + 1, min(by1, img_h))

        cropped = image[y_min:y_max, x_min:x_max].copy()
        if cropped.size == 0:
            return {
                "status": "error",
                "message": (
                    "Cropped region is empty after clamping coordinates."
                ),
            }

        skew_angle = 0.0
        if validated.deskew:
            cropped, skew_angle = _detect_and_correct_skew(cropped)

        if validated.enhance_contrast:
            cropped = _apply_clahe(cropped)

        out_path = Path(validated.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ext = out_path.suffix if out_path.suffix else ".png"
        success, buffer = cv2.imencode(ext, cropped)
        if not success:
            return {
                "status": "error",
                "message": f"Failed to encode image with extension {ext}.",
            }
        with open(out_path, "wb") as f_out:
            f_out.write(buffer)

        out_h, out_w = cropped.shape[:2]
        return {
            "status": "success",
            "output_path": str(out_path),
            "dimensions": {"width": out_w, "height": out_h},
            "skew_angle_detected": round(skew_angle, 3),
        }
    except Exception as exc:
        return {
            "status": "error",
            "message": f"Failed to process and save cropped image: {exc}",
        }
