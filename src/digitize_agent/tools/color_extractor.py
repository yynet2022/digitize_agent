"""特定色によるプロットピクセル抽出および主要色検出ツールモジュール。

HSV 色空間の閾値処理や代表色プリセットにより、連続線プロットまたは
散布図マーカーのピクセル座標を抽出します。
また、画像内の主要プロット色を自動検出する機能を提供します。
"""

from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from pydantic import BaseModel, Field

ColorPresetType = Literal[
    "blue",
    "red",
    "green",
    "orange",
    "black",
    "cyan",
    "magenta",
    "yellow",
    "purple",
    "brown",
    "pink",
    "gray",
]

COLOR_PRESETS: dict[str, tuple[list[int], list[int]]] = {
    "blue": ([90, 50, 50], [130, 255, 255]),
    "red": ([0, 70, 50], [10, 255, 255]),
    "green": ([35, 50, 50], [85, 255, 255]),
    "orange": ([11, 70, 50], [25, 255, 255]),
    "black": ([0, 0, 0], [180, 255, 60]),
    "cyan": ([80, 50, 50], [105, 255, 255]),
    "magenta": ([140, 50, 50], [170, 255, 255]),
    "yellow": ([25, 70, 50], [35, 255, 255]),
    "purple": ([125, 50, 50], [150, 255, 255]),
    "brown": ([10, 80, 20], [25, 200, 150]),
    "pink": ([160, 40, 150], [175, 200, 255]),
    "gray": ([0, 0, 50], [180, 35, 200]),
}


def _parse_hex_color(hex_str: str) -> tuple[int, int, int]:
    """16進数カラーコードをRGBタプルに変換する。"""
    s = hex_str.strip().lstrip("#")
    if len(s) == 3:
        s = "".join([c * 2 for c in s])
    if len(s) != 6:
        raise ValueError(f"Invalid hex color format: {hex_str}")
    r = int(s[0:2], 16)
    g = int(s[2:4], 16)
    b = int(s[4:6], 16)
    return (r, g, b)


def _rgb_to_hsv_bounds(
    rgb: tuple[int, int, int],
    tolerance: float = 0.15,
) -> tuple[list[int], list[int]]:
    """RGB色から許容誤差を考慮したHSV上限・下限値を算出する。"""
    r, g, b = rgb
    pixel_bgr = np.uint8([[[b, g, r]]])
    hsv_pixel = cv2.cvtColor(pixel_bgr, cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = int(hsv_pixel[0]), int(hsv_pixel[1]), int(hsv_pixel[2])

    h_tol = max(6, int(180 * tolerance))
    s_tol = max(35, int(255 * tolerance))
    v_tol = max(35, int(255 * tolerance))

    h_low = max(0, h - h_tol)
    h_high = min(179, h + h_tol)
    s_low = max(30, s - s_tol)
    s_high = min(255, s + s_tol)
    v_low = max(30, v - v_tol)
    v_high = min(255, v + v_tol)

    return ([h_low, s_low, v_low], [h_high, s_high, v_high])


class ExtractPlotPixelsInput(BaseModel):
    """extract_plot_pixels_by_color 関数の入力バリデーションモデル。"""

    image_path: str = Field(description="Path to the plot image file.")
    color_preset: ColorPresetType | None = Field(
        default=None,
        description="Preset color name ('blue', 'red', 'cyan', etc.).",
    )
    target_rgb: list[int] | None = Field(
        default=None,
        min_length=3,
        max_length=3,
        description="Target RGB color [R, G, B] (0-255 each).",
    )
    target_hex: str | None = Field(
        default=None,
        description="Target color in hex format (e.g., '#0072BD', 'FF0000').",
    )
    color_tolerance: float = Field(
        default=0.15,
        ge=0.01,
        le=0.5,
        description="Tolerance factor (0.01-0.5) when matching RGB/Hex.",
    )
    hsv_lower: list[int] | None = Field(
        default=None,
        min_length=3,
        max_length=3,
        description="HSV lower bound [H (0-179), S (0-255), V (0-255)].",
    )
    hsv_upper: list[int] | None = Field(
        default=None,
        min_length=3,
        max_length=3,
        description="HSV upper bound [H (0-179), S (0-255), V (0-255)].",
    )
    bbox: list[int] | None = Field(
        default=None,
        min_length=4,
        max_length=4,
        description="Optional bounding box [x_min, y_min, x_max, y_max].",
    )
    exclude_bboxes: list[list[int]] | None = Field(
        default=None,
        description=(
            "Optional list of bounding boxes to mask out (e.g. legends)."
        ),
    )
    extract_mode: Literal["continuous_line", "scatter_centroids"] = Field(
        default="continuous_line",
        description="Extraction mode: continuous_line or scatter_centroids.",
    )


class DetectPlotColorsInput(BaseModel):
    """detect_plot_colors 関数の入力バリデーションモデル。"""

    image_path: str = Field(description="Path to the plot image file.")
    max_colors: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum number of dominant plot colors to detect.",
    )
    min_pixel_ratio: float = Field(
        default=0.002,
        ge=0.0001,
        le=0.5,
        description="Minimum pixel population ratio to consider as plot line.",
    )


def _classify_hsv_color_name(h: float, s: float, v: float) -> str:
    """HSV値から一般的な色名ラベルを判定する。"""
    if v < 60:
        return "black"
    if s < 35 and v > 200:
        return "white"
    if s < 35:
        return "gray"
    if h < 10 or h >= 170:
        return "red"
    if 10 <= h < 25:
        return "orange"
    if 25 <= h < 35:
        return "yellow"
    if 35 <= h < 85:
        return "green"
    if 85 <= h < 135:
        return "blue"
    if 135 <= h < 170:
        return "purple"
    return "unknown"


def extract_plot_pixels_by_color(
    image_path: str,
    color_preset: ColorPresetType | None = None,
    target_rgb: list[int] | None = None,
    target_hex: str | None = None,
    color_tolerance: float = 0.15,
    hsv_lower: list[int] | None = None,
    hsv_upper: list[int] | None = None,
    bbox: list[int] | None = None,
    exclude_bboxes: list[list[int]] | None = None,
    extract_mode: str = "continuous_line",
) -> dict[str, Any]:
    """HSV色閾値、プリセット、RGB/Hexによりプロットの座標群を抽出する。

    Args:
        image_path: 対象画像のファイルパス。
        color_preset: 代表色プリセット ('blue', 'cyan', 'magenta' 等)。
        target_rgb: 対象 RGB 色 [R, G, B] (0-255)。
        target_hex: 対象 16 進カラーコード (例: '#0072BD')。
        color_tolerance: RGB/Hex 照合時の許容誤差係数 (0.01-0.5)。
        hsv_lower: HSV 下限値リスト [H, S, V] (手動指定時)。
        hsv_upper: HSV 上限値リスト [H, S, V] (手動指定時)。
        bbox: 抽出対象の矩形範囲 [x_min, y_min, x_max, y_max]。
        exclude_bboxes: 除外する矩形範囲のリスト (凡例枠等)。
        extract_mode: 抽出モード
            ("continuous_line" または "scatter_centroids")。

    Returns:
        dict[str, Any]: 抽出点数およびピクセル座標配列、またはエラー。
    """
    try:
        validated = ExtractPlotPixelsInput(
            image_path=image_path,
            color_preset=color_preset,
            target_rgb=target_rgb,
            target_hex=target_hex,
            color_tolerance=color_tolerance,
            hsv_lower=hsv_lower,
            hsv_upper=hsv_upper,
            bbox=bbox,
            exclude_bboxes=exclude_bboxes,
            extract_mode=extract_mode,  # type: ignore[arg-type]
        )
    except Exception as exc:
        return {"status": "error", "message": f"Validation error: {exc}"}

    # プリセット、RGB/Hex、または手動パラメータの解決
    effective_lower = validated.hsv_lower
    effective_upper = validated.hsv_upper

    # 1. Hex 指定の解決
    resolved_rgb = validated.target_rgb
    if validated.target_hex is not None and resolved_rgb is None:
        try:
            resolved_rgb = list(_parse_hex_color(validated.target_hex))
        except ValueError as v_err:
            return {"status": "error", "message": str(v_err)}

    # 2. RGB 指定の解決
    if resolved_rgb is not None and effective_lower is None:
        r, g, b = resolved_rgb
        effective_lower, effective_upper = _rgb_to_hsv_bounds(
            (r, g, b), validated.color_tolerance
        )

    # 3. プリセット指定の解決
    if validated.color_preset is not None and effective_lower is None:
        p_lower, p_upper = COLOR_PRESETS[validated.color_preset]
        effective_lower = p_lower
        effective_upper = p_upper

    if effective_lower is None or effective_upper is None:
        return {
            "status": "error",
            "message": (
                "Either color_preset, target_rgb, target_hex, or "
                "(hsv_lower, hsv_upper) must be provided."
            ),
        }

    img_path = Path(validated.image_path)
    if not img_path.is_file():
        return {
            "status": "error",
            "message": f"Image file not found: {validated.image_path}",
        }

    image = cv2.imread(str(img_path))
    if image is None:
        return {
            "status": "error",
            "message": f"Failed to load image: {validated.image_path}",
        }

    try:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_bound = np.array(effective_lower, dtype=np.uint8)
        upper_bound = np.array(effective_upper, dtype=np.uint8)

        # 赤色プリセット特有のラップアラウンド対応
        if validated.color_preset == "red" and validated.hsv_lower is None:
            m1 = cv2.inRange(hsv, lower_bound, upper_bound)
            m2 = cv2.inRange(
                hsv,
                np.array([170, 70, 50], dtype=np.uint8),
                np.array([179, 255, 255], dtype=np.uint8),
            )
            mask = cv2.bitwise_or(m1, m2)
        else:
            mask = cv2.inRange(hsv, lower_bound, upper_bound)

        # モルフォロジー開閉演算で微細ノイズを除去
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 指定領域 (bbox) 以外のピクセルをゼロマスク
        h_img, w_img = mask.shape[:2]
        if validated.bbox is not None:
            bx0, by0, bx1, by1 = validated.bbox
            bx0 = max(0, min(w_img - 1, bx0))
            bx1 = max(0, min(w_img, bx1))
            by0 = max(0, min(h_img - 1, by0))
            by1 = max(0, min(h_img, by1))
            roi_mask = np.zeros_like(mask)
            roi_mask[by0:by1, bx0:bx1] = 255
            mask = cv2.bitwise_and(mask, roi_mask)

        # 除外領域 (exclude_bboxes, 凡例など) をゼロマスク
        if validated.exclude_bboxes is not None:
            for ex_box in validated.exclude_bboxes:
                if len(ex_box) >= 4:
                    ex0, ey0, ex1, ey1 = ex_box[:4]
                    cv2.rectangle(mask, (ex0, ey0), (ex1, ey1), 0, -1)

        pixel_points: list[list[int]] = []

        if validated.extract_mode == "continuous_line":
            y_indices, x_indices = np.where(mask > 0)
            if len(x_indices) > 0:
                unique_xs = np.unique(x_indices)
                for x in unique_xs:
                    ys = y_indices[x_indices == x]
                    y_median = int(np.median(ys))
                    pixel_points.append([int(x), y_median])
                pixel_points.sort(key=lambda pt: pt[0])
        else:
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for cnt in contours:
                if cv2.contourArea(cnt) < 1.0:
                    continue
                mom = cv2.moments(cnt)
                if mom["m00"] != 0:
                    cx = int(round(mom["m10"] / mom["m00"]))
                    cy = int(round(mom["m01"] / mom["m00"]))
                else:
                    pt = cnt[0][0]
                    cx, cy = int(pt[0]), int(pt[1])
                pixel_points.append([cx, cy])
            pixel_points.sort(key=lambda pt: (pt[0], pt[1]))

        return {
            "status": "success",
            "point_count": len(pixel_points),
            "pixel_points": pixel_points,
        }

    except Exception as exc:
        return {
            "status": "error",
            "message": f"Pixel extraction failed: {exc}",
        }


def detect_plot_colors(
    image_path: str,
    max_colors: int = 5,
    min_pixel_ratio: float = 0.002,
) -> dict[str, Any]:
    """画像内の主要プロット色を自動検出し、色抽出用パラメータ候補を返す。

    白背景や無彩色ノイズを除外し、曲線プロット候補となる代表色クラスタを
    解析して HSV 範囲と推奨プリセット名を返します。

    Args:
        image_path: 対象画像のファイルパス。
        max_colors: 検出する最大色数。
        min_pixel_ratio: プロット線とみなす最小画素割合。

    Returns:
        dict[str, Any]: 検出された代表色リストおよび推奨情報。
    """
    try:
        validated = DetectPlotColorsInput(
            image_path=image_path,
            max_colors=max_colors,
            min_pixel_ratio=min_pixel_ratio,
        )
    except Exception as exc:
        return {"status": "error", "message": f"Validation error: {exc}"}

    img_path = Path(validated.image_path)
    if not img_path.is_file():
        return {
            "status": "error",
            "message": f"Image file not found: {validated.image_path}",
        }

    image = cv2.imread(str(img_path))
    if image is None:
        return {
            "status": "error",
            "message": f"Failed to load image: {validated.image_path}",
        }

    try:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h_channel, s_channel, v_channel = cv2.split(hsv)

        # 白・極薄グレー背景（S < 30 かつ V > 210）を除外
        non_bg_mask = ~((s_channel < 30) & (v_channel > 210))
        fg_pixels_hsv = hsv[non_bg_mask]

        total_pixels = image.shape[0] * image.shape[1]
        if len(fg_pixels_hsv) == 0:
            return {
                "status": "success",
                "dominant_colors": [],
                "message": "No foreground plot pixels detected.",
            }

        # K-Means による代表色クラスタリング
        k = min(validated.max_colors, len(fg_pixels_hsv))
        data = np.float32(fg_pixels_hsv)
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            15,
            1.0,
        )
        _, labels, centers = cv2.kmeans(
            data, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS
        )

        counts = np.bincount(labels.flatten())
        results: list[dict[str, Any]] = []

        for idx, count in enumerate(counts):
            ratio = float(count / total_pixels)
            if ratio < validated.min_pixel_ratio:
                continue

            h_c, s_c, v_c = centers[idx]
            color_name = _classify_hsv_color_name(h_c, s_c, v_c)

            # 推奨 HSV 範囲の算出
            cluster_pts = fg_pixels_hsv[labels.flatten() == idx]
            lower_h = int(max(0, np.percentile(cluster_pts[:, 0], 5) - 5))
            upper_h = int(min(179, np.percentile(cluster_pts[:, 0], 95) + 5))
            lower_s = int(max(20, np.percentile(cluster_pts[:, 1], 5) - 10))
            upper_s = int(min(255, np.percentile(cluster_pts[:, 1], 95) + 10))
            lower_v = int(max(20, np.percentile(cluster_pts[:, 2], 5) - 10))
            upper_v = int(min(255, np.percentile(cluster_pts[:, 2], 95) + 10))

            results.append(
                {
                    "color_name": color_name,
                    "pixel_count": int(count),
                    "pixel_ratio": round(ratio, 4),
                    "representative_hsv": [
                        round(float(h_c), 1),
                        round(float(s_c), 1),
                        round(float(v_c), 1),
                    ],
                    "suggested_hsv_lower": [lower_h, lower_s, lower_v],
                    "suggested_hsv_upper": [upper_h, upper_s, upper_v],
                }
            )

        # 画素割合の降順にソート
        results.sort(key=lambda x: x["pixel_ratio"], reverse=True)

        return {
            "status": "success",
            "detected_colors_count": len(results),
            "dominant_colors": results,
        }

    except Exception as exc:
        return {
            "status": "error",
            "message": f"Plot color detection failed: {exc}",
        }
