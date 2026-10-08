"""PDF ベクターパス（ベジェ曲線・折れ線）抽出ツールモジュール。

PDF 内部のベクター描画命令から3次ベジェ曲線や線分要素を抽出し、
等間隔にサンプリングした座標列を返します。
DPI やクロップ領域を指定することで、ピクセル座標への自動変換も行います。
"""

import math
from pathlib import Path
from typing import Any

import pymupdf as fitz
from pydantic import BaseModel, Field


class ExtractVectorCurvePointsInput(BaseModel):
    """extract_vector_curve_points 関数の入力バリデーションモデル。"""

    pdf_path: str = Field(description="Path to the PDF file.")
    page_number: int = Field(
        default=0, ge=0, description="0-indexed target page number."
    )
    bbox_filter: list[float] | None = Field(
        default=None,
        min_length=4,
        max_length=4,
        description=(
            "Optional [x0, y0, x1, y1] bounding box in PDF points to filter "
            "drawings within target plot area."
        ),
    )
    drawing_index: int | None = Field(
        default=None,
        ge=0,
        description="Optional index of a single specific drawing to extract.",
    )
    drawing_indices: list[int] | None = Field(
        default=None,
        description=(
            "Optional list of drawing indices to extract together (e.g. "
            "[31, 32, 33])."
        ),
    )
    curve_types: list[str] = Field(
        default=["c"],
        description=(
            "Element types to extract: 'c' for Bézier curves (default), "
            "'l' for line segments."
        ),
    )
    item_indices: list[int] | None = Field(
        default=None,
        description="Optional list of item indices within the drawing.",
    )
    curve_segments: list[list[int]] | None = Field(
        default=None,
        description=(
            "Optional explicit grouping of item indices into distinct curves."
        ),
    )
    num_samples_per_segment: int = Field(
        default=50,
        ge=2,
        le=500,
        description=(
            "Number of sampled points per Bézier curve segment (default: 50)."
        ),
    )
    sort_x_ascending: bool = Field(
        default=True,
        description="If True, orient points so X increases monotonically.",
    )
    dpi: float | None = Field(
        default=None,
        gt=0,
        description=(
            "Rendering resolution (DPI) to convert PDF points to pixel "
            "coordinates (match the crop DPI, e.g. 300.0)."
        ),
    )
    crop_bbox_pixels: list[float] | None = Field(
        default=None,
        min_length=4,
        max_length=4,
        description=(
            "Optional [x0, y0, x1, y1] in pixel coordinates of a cropped "
            "region. If provided with dpi, returned coordinates will be "
            "relative to this crop origin."
        ),
    )
    min_length: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Minimum path length in points to filter out short tick marks "
            "or noise dots (e.g. 5.0)."
        ),
    )
    stroke_color: list[float] | None = Field(
        default=None,
        min_length=3,
        max_length=4,
        description="Target stroke color [R, G, B] (0.0-1.0 or 0-255).",
    )
    color_tolerance: float = Field(
        default=0.15,
        ge=0.01,
        le=0.5,
        description="Tolerance factor when matching stroke_color.",
    )
    extract_all_matching: bool = Field(
        default=False,
        description=(
            "If True, return extracted curves from all matching drawings."
        ),
    )
    group_by_color: bool = Field(
        default=False,
        description=(
            "If True, groups matching drawings by identical stroke color and "
            "merges curve segments into continuous curves. Highly recommended."
        ),
    )


def _color_matches(
    dwg_color: tuple[float, ...] | list[float] | None,
    target_color: list[float],
    tolerance: float = 0.15,
) -> bool:
    """描画色と対象色のRGBが許容誤差範囲内で一致するか判定する。"""
    if dwg_color is None:
        return False
    t_rgb = [
        c / 255.0 if any(v > 1.0 for v in target_color) else c
        for c in target_color[:3]
    ]
    d_rgb = [
        c / 255.0 if any(v > 1.0 for v in dwg_color) else c
        for c in dwg_color[:3]
    ]
    if len(d_rgb) < 3 or len(t_rgb) < 3:
        return False
    diffs = [abs(d - t) for d, t in zip(d_rgb, t_rgb)]
    return all(d <= tolerance for d in diffs)


def _calc_segment_length(points: list[list[float]]) -> float:
    """点列の折れ線長さを算出する。"""
    if len(points) < 2:
        return 0.0
    total = 0.0
    for i in range(len(points) - 1):
        dx = points[i + 1][0] - points[i][0]
        dy = points[i + 1][1] - points[i][1]
        total += math.hypot(dx, dy)
    return total


def _eval_cubic_bezier(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    num_samples: int,
) -> list[list[float]]:
    """3次ベジェ曲線を評価して指定数の点列を生成する。

    Args:
        p0: 始点 (x, y)。
        p1: 制御点1 (x, y)。
        p2: 制御点2 (x, y)。
        p3: 終点 (x, y)。
        num_samples: 生成するサンプリング点数 (>= 2)。

    Returns:
        list[list[float]]: [[x, y], ...] 形式の点列リスト。
    """
    points: list[list[float]] = []
    x0, y0 = p0
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    for i in range(num_samples):
        t = i / (num_samples - 1)
        u = 1.0 - t
        tt = t * t
        uu = u * u
        uuu = uu * u
        ttt = tt * t

        x = uuu * x0 + 3.0 * uu * t * x1 + 3.0 * u * tt * x2 + ttt * x3
        y = uuu * y0 + 3.0 * uu * t * y1 + 3.0 * u * tt * y2 + ttt * y3
        points.append([x, y])

    return points


def _dist(p1: list[float], p2: list[float]) -> float:
    """2点間のユークリッド距離を計算する。"""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _dot_product(v1: list[float], v2: list[float]) -> float:
    """2次元ベクトルの内積を計算する。"""
    return v1[0] * v2[0] + v1[1] * v2[1]


def _chain_segments(
    segments: list[dict[str, Any]],
    tolerance: float = 2.0,
    sort_x_ascending: bool = True,
) -> list[dict[str, Any]]:
    """近接するセグメント同士を連結して連続曲線を構築する。

    端点間の距離が tolerance 以内で、かつ進行方向の急変（折り返し、
    内積が負となるヘアピン）が生じないセグメント同士のみを連結します。

    Args:
        segments: 各セグメント情報辞書 (points, item_index)。
        tolerance: 連結判定の最大距離 (pt 単位)。
        sort_x_ascending: X座標を昇順に整列するかどうか。

    Returns:
        list[dict[str, Any]]: 連結された曲線のリスト。
    """
    remaining = list(segments)
    chains: list[dict[str, Any]] = []

    while remaining:
        current = remaining.pop(0)
        curr_points: list[list[float]] = list(current["points"])
        curr_items: list[int] = [current["item_index"]]

        extended = True
        while extended:
            extended = False
            best_idx = -1
            best_mode = None
            best_d = tolerance

            # 現在のチェーンの両端での進行方向ベクトル
            v_end = [
                curr_points[-1][0] - curr_points[-2][0],
                curr_points[-1][1] - curr_points[-2][1],
            ]
            v_start = [
                curr_points[1][0] - curr_points[0][0],
                curr_points[1][1] - curr_points[0][1],
            ]

            for idx, candidate in enumerate(remaining):
                cand_pts = candidate["points"]
                d_end_start = _dist(curr_points[-1], cand_pts[0])
                d_end_end = _dist(curr_points[-1], cand_pts[-1])
                d_start_end = _dist(curr_points[0], cand_pts[-1])
                d_start_start = _dist(curr_points[0], cand_pts[0])

                # 進行方向の内積を判定して折り返し（負の内積）を排除
                if d_end_start < best_d:
                    v_cand = [
                        cand_pts[1][0] - cand_pts[0][0],
                        cand_pts[1][1] - cand_pts[0][1],
                    ]
                    if _dot_product(v_end, v_cand) >= 0:
                        best_d, best_idx, best_mode = (
                            d_end_start,
                            idx,
                            "append_normal",
                        )

                if d_end_end < best_d:
                    v_cand = [
                        cand_pts[-2][0] - cand_pts[-1][0],
                        cand_pts[-2][1] - cand_pts[-1][1],
                    ]
                    if _dot_product(v_end, v_cand) >= 0:
                        best_d, best_idx, best_mode = (
                            d_end_end,
                            idx,
                            "append_reverse",
                        )

                if d_start_end < best_d:
                    v_cand = [
                        cand_pts[-1][0] - cand_pts[-2][0],
                        cand_pts[-1][1] - cand_pts[-2][1],
                    ]
                    if _dot_product(v_cand, v_start) >= 0:
                        best_d, best_idx, best_mode = (
                            d_start_end,
                            idx,
                            "prepend_normal",
                        )

                if d_start_start < best_d:
                    v_cand = [
                        cand_pts[0][0] - cand_pts[1][0],
                        cand_pts[0][1] - cand_pts[1][1],
                    ]
                    if _dot_product(v_cand, v_start) >= 0:
                        best_d, best_idx, best_mode = (
                            d_start_start,
                            idx,
                            "prepend_reverse",
                        )

            if best_idx != -1 and best_mode is not None:
                cand = remaining.pop(best_idx)
                cand_pts = cand["points"]
                cand_item = cand["item_index"]

                if best_mode == "append_normal":
                    curr_points.extend(cand_pts[1:])
                    curr_items.append(cand_item)
                elif best_mode == "append_reverse":
                    curr_points.extend(list(reversed(cand_pts))[1:])
                    curr_items.append(cand_item)
                elif best_mode == "prepend_normal":
                    curr_points = cand_pts[:-1] + curr_points
                    curr_items.insert(0, cand_item)
                elif best_mode == "prepend_reverse":
                    curr_points = list(reversed(cand_pts))[:-1] + curr_points
                    curr_items.insert(0, cand_item)
                extended = True

        if sort_x_ascending and len(curr_points) > 1:
            if curr_points[0][0] > curr_points[-1][0]:
                curr_points = list(reversed(curr_points))

        chains.append({"item_indices": curr_items, "points": curr_points})

    return chains


def _extract_curves_from_drawing(
    drawing: dict[str, Any],
    curve_types: list[str],
    item_indices: list[int] | None,
    curve_segments: list[list[int]] | None,
    num_samples_per_segment: int,
    sort_x_ascending: bool,
    scale_factor: float,
    crop_x0: float,
    crop_y0: float,
    dpi: float | None,
    min_length: float = 0.0,
) -> list[dict[str, Any]]:
    """単一の描画辞書から曲線群を抽出してスケーリング変換する。"""
    items = drawing.get("items", [])
    item_map: dict[int, dict[str, Any]] = {}

    for i, item in enumerate(items):
        if item_indices and i not in item_indices:
            continue

        kind = item[0]
        if kind not in curve_types:
            continue

        if kind == "c":
            p0 = (float(item[1].x), float(item[1].y))
            p1 = (float(item[2].x), float(item[2].y))
            p2 = (float(item[3].x), float(item[3].y))
            p3 = (float(item[4].x), float(item[4].y))
            pts = _eval_cubic_bezier(p0, p1, p2, p3, num_samples_per_segment)
        elif kind == "l":
            p0 = (float(item[1].x), float(item[1].y))
            p1 = (float(item[2].x), float(item[2].y))
            pts = [[p0[0], p0[1]], [p1[0], p1[1]]]
        else:
            continue

        if min_length > 0.0 and _calc_segment_length(pts) < min_length:
            continue

        item_map[i] = {"item_index": i, "points": pts}

    if not item_map:
        return []

    if curve_segments:
        chains = []
        for group in curve_segments:
            group_segs = [item_map[idx] for idx in group if idx in item_map]
            if group_segs:
                sub_chains = _chain_segments(
                    group_segs, sort_x_ascending=sort_x_ascending
                )
                chains.extend(sub_chains)
    else:
        chains = _chain_segments(
            list(item_map.values()), sort_x_ascending=sort_x_ascending
        )

    result_curves: list[dict[str, Any]] = []
    for c_idx, ch in enumerate(chains):
        transformed_pts: list[list[float]] = []
        for pt in ch["points"]:
            if dpi is not None:
                px = pt[0] * scale_factor - crop_x0
                py = pt[1] * scale_factor - crop_y0
            else:
                px = pt[0]
                py = pt[1]
            transformed_pts.append([round(px, 4), round(py, 4)])

        xs = [p[0] for p in transformed_pts]
        ys = [p[1] for p in transformed_pts]

        result_curves.append(
            {
                "curve_index": c_idx,
                "item_indices": ch["item_indices"],
                "num_points": len(transformed_pts),
                "bounds": [min(xs), min(ys), max(xs), max(ys)],
                "points": transformed_pts,
            }
        )

    return result_curves


def extract_vector_curve_points(
    pdf_path: str,
    page_number: int = 0,
    bbox_filter: list[float] | None = None,
    drawing_index: int | None = None,
    curve_types: list[str] = ["c"],
    drawing_indices: list[int] | None = None,
    item_indices: list[int] | None = None,
    curve_segments: list[list[int]] | None = None,
    num_samples_per_segment: int = 50,
    sort_x_ascending: bool = True,
    dpi: float | None = None,
    crop_bbox_pixels: list[float] | None = None,
    min_length: float = 0.0,
    stroke_color: list[float] | None = None,
    color_tolerance: float = 0.15,
    extract_all_matching: bool = False,
    group_by_color: bool = False,
) -> dict[str, Any]:
    """PDF のベクター描画パスから曲線の座標列を直接サンプリング抽出する。

    PDF 内部に埋め込まれたベクター描画命令（ベジェ曲線 'c'、線分 'l'）を
    解析し、幾何学的に等間隔な座標点列 [[x, y], ...] を抽出します。
    ラスタ画像変換を介さないため解像度劣化や色にじみがなく、最高精度です。
    複数描画の一括抽出（drawing_indices）、同一線色の自動マージ（group_by_color）、
    目盛り線の除外（min_length）に対応しています。

    推奨ワークフロー:
        1. inspect_pdf_primitives でベクター線の存在とインデックスを確認。
        2. 抽出対象の描画インデックス群を drawing_indices に渡し、
           group_by_color=True, dpi=300.0 を指定して本関数を実行。
        3. 出力の color_grouped_curves をそのまま
           calibrate_and_convert_coordinates の curves に渡して CSV 変換。

    Args:
        pdf_path: 対象 PDF ファイルのパス。
        page_number: 0 から始まる対象ページ番号 (標準: 0)。
        bbox_filter: 描画抽出範囲 [x0, y0, x1, y1] (PDF pt単位)。
        drawing_index: 特定の単一描画オブジェクト番号。
        drawing_indices: 複数の特定描画オブジェクト番号リスト (推奨)。
        curve_types: 抽出対象要素タイプ ('c': ベジェ, 'l': 線分、標準: ['c'])。
        item_indices: 描画内の特定アイテム番号リスト。
        curve_segments: 曲線ごとにアイテム番号をグループ化したリスト。
        num_samples_per_segment: ベジェ曲線1区間あたりのサンプル数 (標準: 50)。
        sort_x_ascending: X座標を昇順に整列するかどうか (標準: True)。
        dpi: ピクセル変換用の解像度 (DPI、クロップ画像と一致させる)。
        crop_bbox_pixels: クロップ画像の [x0, y0, x1, y1] (px単位)。
        min_length: 短い目盛り線等を除外する最小パス長 (pt単位、例: 5.0)。
        stroke_color: 抽出対象の線色 [R, G, B]。
        color_tolerance: 線色照合時の許容誤差係数。
        extract_all_matching: マッチする全描画を一括返却するかどうか。
        group_by_color: 同一線色の描画曲線を自動統合するか (標準: False)。

    Returns:
        dict[str, Any]:
            - status: "success" または "error"。
            - curves: 抽出された各曲線のリスト。
            - color_grouped_curves: group_by_color 有効時に同一色で統合された
              曲線リスト (calibrate_and_convert_coordinates に直接入力可能)。
    """
    try:
        validated = ExtractVectorCurvePointsInput(
            pdf_path=pdf_path,
            page_number=page_number,
            bbox_filter=bbox_filter,
            drawing_index=drawing_index,
            drawing_indices=drawing_indices,
            curve_types=curve_types,
            item_indices=item_indices,
            curve_segments=curve_segments,
            num_samples_per_segment=num_samples_per_segment,
            sort_x_ascending=sort_x_ascending,
            dpi=dpi,
            crop_bbox_pixels=crop_bbox_pixels,
            min_length=min_length,
            stroke_color=stroke_color,
            color_tolerance=color_tolerance,
            extract_all_matching=extract_all_matching,
            group_by_color=group_by_color,
        )
    except Exception as exc:
        return {"status": "error", "message": f"Validation error: {exc}"}

    path_obj = Path(validated.pdf_path)
    if not path_obj.is_file():
        return {
            "status": "error",
            "message": f"PDF file not found: {validated.pdf_path}",
        }

    try:
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
            drawings = page.get_drawings()

            if not drawings:
                return {
                    "status": "error",
                    "message": "No vector drawings found on this page.",
                }

            scale_factor = (
                (validated.dpi / 72.0) if validated.dpi is not None else 1.0
            )
            crop_x0 = (
                validated.crop_bbox_pixels[0]
                if validated.crop_bbox_pixels
                else 0.0
            )
            crop_y0 = (
                validated.crop_bbox_pixels[1]
                if validated.crop_bbox_pixels
                else 0.0
            )

            # 対象 drawing の選定
            if validated.drawing_indices is not None:
                matching_drawings = [
                    idx
                    for idx in validated.drawing_indices
                    if 0 <= idx < len(drawings)
                ]
                if not matching_drawings:
                    return {
                        "status": "error",
                        "message": (
                            "None of the specified drawing_indices are valid."
                        ),
                    }
            elif validated.drawing_index is not None:
                if 0 <= validated.drawing_index < len(drawings):
                    matching_drawings = [validated.drawing_index]
                else:
                    return {
                        "status": "error",
                        "message": (
                            f"Drawing index {validated.drawing_index} out of "
                            f"range (total drawings: {len(drawings)})."
                        ),
                    }
            else:
                matching_drawings = []
                for idx, dwg in enumerate(drawings):
                    rect = dwg.get("rect")
                    if validated.bbox_filter:
                        bx0, by0, bx1, by1 = validated.bbox_filter
                        if rect is None:
                            continue
                        if (
                            rect.x1 < bx0
                            or rect.x0 > bx1
                            or rect.y1 < by0
                            or rect.y0 > by1
                        ):
                            continue

                    if validated.stroke_color is not None:
                        dwg_col = dwg.get("color")
                        if not _color_matches(
                            dwg_col,
                            validated.stroke_color,
                            validated.color_tolerance,
                        ):
                            continue

                    items_list = dwg.get("items", [])
                    has_curves = any(
                        it[0] in validated.curve_types for it in items_list
                    )
                    if has_curves:
                        matching_drawings.append(idx)

            if not matching_drawings:
                return {
                    "status": "error",
                    "message": (
                        "No drawings containing requested curves found "
                        "matching the criteria."
                    ),
                }

            # 色別グループ化一括統合モード
            if validated.group_by_color:
                color_groups: dict[str, dict[str, Any]] = {}
                for d_idx in matching_drawings:
                    dwg = drawings[d_idx]
                    col = dwg.get("color")
                    if col is None:
                        col_key = "none"
                        col_rgb = None
                        col_hex = None
                    else:
                        r = int(round(col[0] * 255))
                        g = int(round(col[1] * 255))
                        b = int(round(col[2] * 255))
                        col_key = f"{col[0]:.2f},{col[1]:.2f},{col[2]:.2f}"
                        col_rgb = [r, g, b]
                        col_hex = f"#{r:02X}{g:02X}{b:02X}"

                    curves = _extract_curves_from_drawing(
                        drawing=dwg,
                        curve_types=validated.curve_types,
                        item_indices=validated.item_indices,
                        curve_segments=validated.curve_segments,
                        num_samples_per_segment=(
                            validated.num_samples_per_segment
                        ),
                        sort_x_ascending=validated.sort_x_ascending,
                        scale_factor=scale_factor,
                        crop_x0=crop_x0,
                        crop_y0=crop_y0,
                        dpi=validated.dpi,
                        min_length=validated.min_length,
                    )
                    if not curves:
                        continue

                    if col_key not in color_groups:
                        color_groups[col_key] = {
                            "stroke_color": list(col) if col else None,
                            "color_rgb": col_rgb,
                            "color_hex": col_hex,
                            "drawing_indices": [],
                            "all_curves": [],
                        }
                    color_groups[col_key]["drawing_indices"].append(d_idx)
                    color_groups[col_key]["all_curves"].extend(curves)

                grouped_results = []
                for entry in color_groups.values():
                    merged_pts: list[list[float]] = []
                    for c in entry["all_curves"]:
                        merged_pts.extend(c["points"])
                    if validated.sort_x_ascending and merged_pts:
                        merged_pts.sort(key=lambda p: p[0])
                        deduped = [merged_pts[0]]
                        for p in merged_pts[1:]:
                            if (
                                abs(p[0] - deduped[-1][0]) > 1e-4
                                or abs(p[1] - deduped[-1][1]) > 1e-4
                            ):
                                deduped.append(p)
                        merged_pts = deduped

                    grouped_results.append(
                        {
                            "stroke_color": entry["stroke_color"],
                            "color_rgb": entry["color_rgb"],
                            "color_hex": entry["color_hex"],
                            "drawing_indices": entry["drawing_indices"],
                            "total_points": len(merged_pts),
                            "points": merged_pts,
                        }
                    )

                return {
                    "status": "success",
                    "total_groups": len(grouped_results),
                    "color_grouped_curves": grouped_results,
                }

            # 全描画一括抽出モード
            if validated.extract_all_matching:
                all_results: list[dict[str, Any]] = []
                for d_idx in matching_drawings:
                    dwg = drawings[d_idx]
                    curves = _extract_curves_from_drawing(
                        drawing=dwg,
                        curve_types=validated.curve_types,
                        item_indices=validated.item_indices,
                        curve_segments=validated.curve_segments,
                        num_samples_per_segment=(
                            validated.num_samples_per_segment
                        ),
                        sort_x_ascending=validated.sort_x_ascending,
                        scale_factor=scale_factor,
                        crop_x0=crop_x0,
                        crop_y0=crop_y0,
                        dpi=validated.dpi,
                        min_length=validated.min_length,
                    )
                    if curves:
                        d_rect = dwg.get("rect")
                        rect_l = (
                            [d_rect.x0, d_rect.y0, d_rect.x1, d_rect.y1]
                            if d_rect is not None
                            else None
                        )
                        all_results.append(
                            {
                                "drawing_index": d_idx,
                                "drawing_rect": rect_l,
                                "stroke_color": dwg.get("color"),
                                "total_curves": len(curves),
                                "curves": curves,
                            }
                        )

                return {
                    "status": "success",
                    "total_drawings": len(all_results),
                    "drawings": all_results,
                }

            # 単一描画抽出モード (有効な曲線が得られる最初の描画を探索)
            result_curves = []
            target_idx = -1
            target_dwg = None

            for d_idx in matching_drawings:
                dwg = drawings[d_idx]
                curves = _extract_curves_from_drawing(
                    drawing=dwg,
                    curve_types=validated.curve_types,
                    item_indices=validated.item_indices,
                    curve_segments=validated.curve_segments,
                    num_samples_per_segment=validated.num_samples_per_segment,
                    sort_x_ascending=validated.sort_x_ascending,
                    scale_factor=scale_factor,
                    crop_x0=crop_x0,
                    crop_y0=crop_y0,
                    dpi=validated.dpi,
                    min_length=validated.min_length,
                )
                if curves:
                    result_curves = curves
                    target_idx = d_idx
                    target_dwg = dwg
                    break

            if not result_curves or target_dwg is None:
                return {
                    "status": "error",
                    "message": (
                        "No curve or line segments extracted from items."
                    ),
                }

            d_rect = target_dwg.get("rect")
            rect_list = (
                [d_rect.x0, d_rect.y0, d_rect.x1, d_rect.y1]
                if d_rect is not None
                else None
            )

            return {
                "status": "success",
                "drawing_index": target_idx,
                "drawing_rect": rect_list,
                "total_curves": len(result_curves),
                "curves": result_curves,
            }

    except Exception as exc:
        return {
            "status": "error",
            "message": f"Failed to extract vector curve points: {exc}",
        }
