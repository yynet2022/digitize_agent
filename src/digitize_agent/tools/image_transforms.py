"""画像領域の切り出しおよび前処理ツールモジュール。

指定領域のクロップ、傾き（Deskew）検出および補正、コントラスト強調（CLAHE）
を行い、解析用サブ画像を生成します。
"""

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pydantic import BaseModel, Field


class CropAndTransformInput(BaseModel):
    """crop_and_transform_region 関数の入力バリデーションモデル。"""

    image_path: str = Field(description="Path to the source image file.")
    bbox: list[int] = Field(
        min_length=4,
        max_length=4,
        description="Bounding box [x_min, y_min, x_max, y_max].",
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
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        if dx == 0:
            continue
        angle = np.degrees(np.arctan2(dy, dx))
        # 水平に近い線分（-45度から45度）の傾斜を採用
        if -45.0 <= angle <= 45.0:
            angles.append(angle)

    if not angles:
        return image, 0.0

    median_angle = float(np.median(angles))
    # 微小な傾き（0.01度未満）は補正しない
    if abs(median_angle) < 0.01:
        return image, median_angle

    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    rot_mat = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(
        image,
        rot_mat,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated, median_angle


def _apply_clahe(image: np.ndarray) -> np.ndarray:
    """LAB色空間のLチャンネルに対してCLAHEを適用する。

    Args:
        image: 対象の入力画像 (BGR)。

    Returns:
        np.ndarray: コントラスト強調後の画像 (BGR)。
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    if len(image.shape) == 2:
        return clahe.apply(image)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    l_enhanced = clahe.apply(l_chan)
    merged = cv2.merge((l_enhanced, a_chan, b_chan))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def crop_and_transform_region(
    image_path: str,
    bbox: list[int],
    deskew: bool = False,
    enhance_contrast: bool = False,
    output_path: str = "",
) -> dict[str, Any]:
    """指定バウンディングボックスの領域を切り出し、傾き補正や強調を行う。

    Args:
        image_path: 元画像ファイルへのパス。
        bbox: [x_min, y_min, x_max, y_max] の切り出し範囲。
        deskew: 傾き検出と補正を行うフラグ。
        enhance_contrast: CLAHE によるコントラスト強調フラグ。
        output_path: 保存先のファイルパス。

    Returns:
        dict[str, Any]: 処理結果情報またはエラーメッセージ。
    """
    try:
        validated = CropAndTransformInput(
            image_path=image_path,
            bbox=bbox,
            deskew=deskew,
            enhance_contrast=enhance_contrast,
            output_path=output_path,
        )
    except Exception as exc:
        return {"status": "error", "message": f"Validation error: {exc}"}

    img_path = Path(validated.image_path)
    if not img_path.is_file():
        return {
            "status": "error",
            "message": f"Source image not found: {validated.image_path}",
        }

    try:
        image = cv2.imread(str(img_path))
        if image is None:
            return {
                "status": "error",
                "message": f"Failed to decode image: {validated.image_path}",
            }

        img_h, img_w = image.shape[:2]
        x_min, y_min, x_max, y_max = validated.bbox

        # 境界値のクリッピングと整合性チェック
        x_min = max(0, min(x_min, img_w - 1))
        x_max = max(x_min + 1, min(x_max, img_w))
        y_min = max(0, min(y_min, img_h - 1))
        y_max = max(y_min + 1, min(y_max, img_h))

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
        # Windows 環境の非ASCII文字パス対策として imencode を使用
        ext = out_path.suffix if out_path.suffix else ".png"
        success, buffer = cv2.imencode(ext, cropped)
        if not success:
            return {
                "status": "error",
                "message": (f"Failed to encode image with extension {ext}."),
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
