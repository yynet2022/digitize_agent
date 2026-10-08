"""OpenAI Function Calling 互換のツールスキーマ定義モジュール。

デジタイズエージェントで利用可能な全 12 ツールの JSON Schema 定義を提供します。
"""

import json
from pathlib import Path
from typing import Any

INSPECT_PDF_PRIMITIVES_SCHEMA: dict[str, Any] = {
    "name": "inspect_pdf_primitives",
    "description": (
        "Inspect a PDF page to detect if it contains native vector paths "
        "vs scanned raster images, and extract raw vector lines and text "
        "bounding boxes without raster degradation. Use this first to decide "
        "between vector extraction and raster color extraction pipelines."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "pdf_path": {
                "type": "string",
                "description": (
                    "Absolute or relative path to the local PDF file."
                ),
            },
            "page_number": {
                "type": "integer",
                "description": "0-indexed target page number (default: 0).",
                "default": 0,
            },
            "bbox_filter": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 4,
                "maxItems": 4,
                "description": (
                    "Optional bounding box [x0, y0, x1, y1] in points to "
                    "filter elements within target area."
                ),
            },
        },
        "required": ["pdf_path"],
    },
}

SEARCH_PDF_PRIMITIVES_SCHEMA: dict[str, Any] = {
    "name": "search_pdf_primitives",
    "description": (
        "Search text keywords (e.g. 'Figure 5', 'Fig. 4', 'Table 1') across "
        "all PDF pages and return matched pages, bounding boxes, and snippet "
        "context. Pass the matched caption bbox to crop_and_transform_region "
        "to automatically estimate and crop the figure area."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "pdf_path": {
                "type": "string",
                "description": "Path to the local PDF file.",
            },
            "query": {
                "type": "string",
                "description": (
                    "Search text keyword or phrase (e.g. 'Figure 5', 'Fig.')."
                ),
            },
            "case_sensitive": {
                "type": "boolean",
                "description": (
                    "Whether the search is case-sensitive (default: False)."
                ),
                "default": False,
            },
            "max_matches": {
                "type": "integer",
                "description": "Maximum number of matches to return.",
                "default": 50,
            },
        },
        "required": ["pdf_path", "query"],
    },
}

CROP_AND_TRANSFORM_REGION_SCHEMA: dict[str, Any] = {
    "name": "crop_and_transform_region",
    "description": (
        "Crop a specific region of an image or render directly from a PDF "
        "page at high DPI (default 300). Pass caption_bbox to auto-estimate "
        "the plot area above it. Supports deskewing and contrast enhancement. "
        "Outputs a cropped image ready for axis and curve extraction."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": (
                    "Path to source raster image (alternative to pdf_path)."
                ),
            },
            "pdf_path": {
                "type": "string",
                "description": (
                    "Path to PDF file to render and crop at custom DPI."
                ),
            },
            "page_number": {
                "type": "integer",
                "description": "0-indexed page number if extracting from PDF.",
                "default": 0,
            },
            "dpi": {
                "type": "number",
                "description": (
                    "Rendering resolution (DPI) for PDF (default: 300.0)."
                ),
                "default": 300.0,
            },
            "bbox": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 4,
                "description": (
                    "Explicit bounding box [x_min, y_min, x_max, y_max]."
                ),
            },
            "caption_bbox": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 4,
                "maxItems": 4,
                "description": (
                    "Bounding box [x0, y0, x1, y1] in points of caption "
                    "(from search_pdf_primitives) to estimate plot area."
                ),
            },
            "bbox_mode": {
                "type": "string",
                "enum": ["pixel", "point"],
                "description": "Unit of bbox: 'pixel' or 'point' (72 DPI pt).",
                "default": "pixel",
            },
            "deskew": {
                "type": "boolean",
                "description": "Apply Hough transform deskewing if true.",
                "default": False,
            },
            "enhance_contrast": {
                "type": "boolean",
                "description": "Apply CLAHE contrast enhancement if true.",
                "default": False,
            },
            "output_path": {
                "type": "string",
                "description": (
                    "Destination file path for the cropped image (PNG)."
                ),
            },
        },
        "required": ["output_path"],
    },
}

DETECT_AXES_AND_TICKS_SCHEMA: dict[str, Any] = {
    "name": "detect_axes_and_ticks",
    "description": (
        "Detect horizontal and vertical coordinate axes, tick mark pixel "
        "positions, and enclosing box frame from a plot image. When "
        "detect_box_frame=True (default), extracts inner_bbox of the plot "
        "frame, enabling calibration for qualitative or arbitrary-unit "
        "(a.u.) plots without numeric ticks."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": "Path to the cropped plot image file.",
            },
            "min_line_length_ratio": {
                "type": "number",
                "description": (
                    "Minimum line length as a ratio of image dimensions "
                    "(default: 0.3)."
                ),
                "default": 0.3,
            },
            "detect_box_frame": {
                "type": "boolean",
                "description": (
                    "Whether to detect rectangular box frame enclosing plot "
                    "area. Returns inner_bbox for normalized calibration."
                ),
                "default": True,
            },
        },
        "required": ["image_path"],
    },
}

DETECT_LEGEND_REGION_SCHEMA: dict[str, Any] = {
    "name": "detect_legend_region",
    "description": (
        "Detect legend bounding boxes and legend color items (RGB, Hex, "
        "and recommended HSV lower/upper bounds). Pass legend_bboxes to "
        "extract_plot_pixels_by_color's exclude_bboxes to avoid extracting "
        "legend text, and use suggested HSV bounds directly for curve "
        "extraction."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": "Path to the cropped plot image file.",
            },
            "plot_bbox": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 4,
                "maxItems": 4,
                "description": (
                    "Optional inner plot bounding box [x_min, y_min, "
                    "x_max, y_max]. If omitted, automatically estimated."
                ),
            },
        },
        "required": ["image_path"],
    },
}

OCR_REGION_TEXT_SCHEMA: dict[str, Any] = {
    "name": "ocr_region_text",
    "description": (
        "Extract text from an image snippet using Tesseract OCR, or "
        "directly read native embedded vector text from a digital PDF with "
        "zero OCR errors. Use for reading axis labels, units, and numeric "
        "tick values."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": "Path to cropped image snippet for OCR.",
            },
            "pdf_path": {
                "type": "string",
                "description": (
                    "Optional PDF path for native vector text reading."
                ),
            },
            "page_number": {
                "type": "integer",
                "description": "0-indexed PDF page number (default: 0).",
                "default": 0,
            },
            "bbox": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 4,
                "maxItems": 4,
                "description": "Bounding box [x0, y0, x1, y1] in points.",
            },
            "psm": {
                "type": "integer",
                "description": (
                    "Tesseract PSM: 6 for uniform block, 7 for single line."
                ),
                "default": 6,
            },
            "whitelist": {
                "type": "string",
                "description": (
                    "Optional character whitelist (e.g. '0123456789.-')."
                ),
            },
        },
    },
}

DETECT_PLOT_COLORS_SCHEMA: dict[str, Any] = {
    "name": "detect_plot_colors",
    "description": (
        "Automatically detect dominant colored plot curves in a plot image, "
        "filtering out white background and black/gray axes. Returns color "
        "names, RGB/Hex, and recommended HSV bounds (suggested_hsv_lower, "
        "suggested_hsv_upper) to pass directly to "
        "extract_plot_pixels_by_color."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": "Path to the cropped plot image file.",
            },
            "max_colors": {
                "type": "integer",
                "description": (
                    "Maximum number of dominant colors to detect (1-10)."
                ),
                "default": 5,
            },
            "min_pixel_ratio": {
                "type": "number",
                "description": (
                    "Minimum pixel ratio for plot lines (default: 0.002)."
                ),
                "default": 0.002,
            },
        },
        "required": ["image_path"],
    },
}

EXTRACT_PLOT_PIXELS_BY_COLOR_SCHEMA: dict[str, Any] = {
    "name": "extract_plot_pixels_by_color",
    "description": (
        "Extract pixel coordinates of curves matching target color via HSV "
        "bounds, Hex, RGB, or presets. Supports exclude_bboxes for legends, "
        "x_range for segmenting peak intervals, and smooth_filter (local "
        "median filter) to eliminate raster noise and jump artifacts."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": "Path to the cropped plot image file.",
            },
            "color_preset": {
                "type": "string",
                "enum": [
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
                ],
                "description": "Preset color name for quick extraction.",
            },
            "target_rgb": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 3,
                "maxItems": 3,
                "description": "Target color in RGB [R, G, B] (0-255 each).",
            },
            "target_hex": {
                "type": "string",
                "description": (
                    "Target color in hex format (e.g. '#0072BD', '#D95319')."
                ),
            },
            "color_tolerance": {
                "type": "number",
                "description": (
                    "Matching tolerance (0.01-1.0 normalized or 1-255 px "
                    "distance, default: 35.0)."
                ),
                "default": 35.0,
            },
            "hsv_lower": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 3,
                "maxItems": 3,
                "description": (
                    "Explicit HSV lower bound [H (0-179), S, V (0-255)]."
                ),
            },
            "hsv_upper": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 3,
                "maxItems": 3,
                "description": (
                    "Explicit HSV upper bound [H (0-179), S, V (0-255)]."
                ),
            },
            "bbox": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 4,
                "maxItems": 4,
                "description": "Extraction bbox [x_min, y_min, x_max, y_max].",
            },
            "exclude_bboxes": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 4,
                    "maxItems": 4,
                },
                "description": (
                    "List of bounding boxes to mask out (e.g. legend boxes "
                    "from detect_legend_region)."
                ),
            },
            "extract_mode": {
                "type": "string",
                "enum": ["continuous_line", "scatter_centroids"],
                "description": (
                    "Extraction mode: 'continuous_line' for lines or "
                    "'scatter_centroids' for dots."
                ),
                "default": "continuous_line",
            },
            "x_range": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 2,
                "maxItems": 2,
                "description": (
                    "Optional [x_min, x_max] pixel column range to restrict "
                    "extraction. Crucial for segmenting peak intervals."
                ),
            },
            "smooth_filter": {
                "type": "boolean",
                "description": (
                    "Enable local median outlier filter for continuous lines. "
                    "Strongly recommended for noisy raster curves."
                ),
                "default": False,
            },
            "max_jump": {
                "type": "number",
                "description": (
                    "Maximum allowed jump in pixels from local median "
                    "(default: 15.0 px)."
                ),
                "default": 15.0,
            },
        },
        "required": ["image_path"],
    },
}

EXTRACT_VECTOR_CURVE_POINTS_SCHEMA: dict[str, Any] = {
    "name": "extract_vector_curve_points",
    "description": (
        "Sample dense, high-precision coordinates directly from PDF vector "
        "drawing commands (Bézier curves 'c' or lines 'l') without raster "
        "degradation. Use drawing_indices and group_by_color=True to merge "
        "matching stroke curves automatically."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "pdf_path": {
                "type": "string",
                "description": "Path to the source PDF file.",
            },
            "page_number": {
                "type": "integer",
                "description": "0-indexed PDF page number (default: 0).",
                "default": 0,
            },
            "bbox_filter": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 4,
                "maxItems": 4,
                "description": (
                    "Optional [x0, y0, x1, y1] in points to filter drawings."
                ),
            },
            "drawing_index": {
                "type": "integer",
                "description": "Optional specific single drawing index.",
            },
            "drawing_indices": {
                "type": "array",
                "items": {"type": "integer"},
                "description": (
                    "Optional list of drawing indices to extract together."
                ),
            },
            "group_by_color": {
                "type": "boolean",
                "description": (
                    "Group drawings by stroke color and merge into unified "
                    "curves. Strongly recommended."
                ),
                "default": False,
            },
            "curve_types": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Element types to extract: 'c' (Bézier) or 'l' (lines)."
                ),
                "default": ["c"],
            },
            "item_indices": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Optional specific item indices in drawing.",
            },
            "curve_segments": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "integer"},
                },
                "description": "Optional explicit item grouping for curves.",
            },
            "num_samples_per_segment": {
                "type": "integer",
                "description": (
                    "Sample points per Bézier curve segment (default: 50)."
                ),
                "default": 50,
            },
            "sort_x_ascending": {
                "type": "boolean",
                "description": "Sort points so X increases monotonically.",
                "default": True,
            },
            "dpi": {
                "type": "number",
                "description": (
                    "DPI to scale coordinates to pixels (match crop DPI)."
                ),
            },
            "crop_bbox_pixels": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 4,
                "maxItems": 4,
                "description": "Optional [x0, y0, x1, y1] crop offset in px.",
            },
            "min_length": {
                "type": "number",
                "description": (
                    "Minimum path length in points to filter tick marks/noise."
                ),
                "default": 0.0,
            },
            "stroke_color": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 3,
                "maxItems": 4,
                "description": (
                    "Target stroke color [R, G, B] (0.0-1.0 or 0-255)."
                ),
            },
            "color_tolerance": {
                "type": "number",
                "description": "Tolerance factor for matching stroke_color.",
                "default": 0.15,
            },
            "extract_all_matching": {
                "type": "boolean",
                "description": "Return curves from all matching drawings.",
                "default": False,
            },
        },
        "required": ["pdf_path"],
    },
}

AUTO_CALIBRATE_AXES_SCHEMA: dict[str, Any] = {
    "name": "auto_calibrate_axes",
    "description": (
        "One-stop axis calibration deriving x_calibration and y_calibration "
        "for calibrate_and_convert_coordinates. Supports mode='tick_matched' "
        "(pairing ticks with PDF/OCR text) and mode='normalized' (mapping "
        "plot inner frame to [0,1] or specified domain for qualitative/a.u. "
        "plots). Automatically falls back to normalized mode if tick "
        "matching fails."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": "Path to the cropped plot image file.",
            },
            "pdf_path": {
                "type": "string",
                "description": (
                    "Optional path to original PDF for precise text reading."
                ),
            },
            "page_number": {
                "type": "integer",
                "description": "0-indexed PDF page number (default: 0).",
                "default": 0,
            },
            "crop_bbox_points": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 4,
                "maxItems": 4,
                "description": (
                    "Crop bounding box [x0, y0, x1, y1] in PDF pt (from "
                    "crop_and_transform_region estimated_bbox)."
                ),
            },
            "dpi": {
                "type": "number",
                "description": (
                    "Resolution of cropped image (default: 300.0)."
                ),
                "default": 300.0,
            },
            "x_tick_candidates": {
                "type": "array",
                "items": {"type": "integer"},
                "description": (
                    "Optional pre-detected X tick pixel positions."
                ),
            },
            "y_tick_candidates": {
                "type": "array",
                "items": {"type": "integer"},
                "description": (
                    "Optional pre-detected Y tick pixel positions."
                ),
            },
            "mode": {
                "type": "string",
                "enum": ["tick_matched", "normalized"],
                "description": (
                    "Calibration mode: 'tick_matched' for standard plots with "
                    "numeric labels, or 'normalized' for qualitative / a.u. "
                    "plots without numeric labels."
                ),
                "default": "tick_matched",
            },
            "normalized_domain_x": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 2,
                "maxItems": 2,
                "description": (
                    "Target domain [min, max] for X in normalized mode "
                    "(default: [0.0, 1.0])."
                ),
                "default": [0.0, 1.0],
            },
            "normalized_domain_y": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 2,
                "maxItems": 2,
                "description": (
                    "Target domain [min, max] for Y in normalized mode "
                    "(default: [0.0, 1.0])."
                ),
                "default": [0.0, 1.0],
            },
            "box_frame_bbox": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 4,
                "maxItems": 4,
                "description": (
                    "Optional plot inner frame [x0, y0, x1, y1] pixels (from "
                    "detect_axes_and_ticks box_frame inner_bbox)."
                ),
            },
            "fallback_to_normalized": {
                "type": "boolean",
                "description": (
                    "Automatically fall back to normalized mode if tick "
                    "label matching finds insufficient points."
                ),
                "default": True,
            },
        },
        "required": ["image_path"],
    },
}

CALIBRATE_AND_CONVERT_COORDINATES_SCHEMA: dict[str, Any] = {
    "name": "calibrate_and_convert_coordinates",
    "description": (
        "Convert pixel coordinates into physical values using linear/log "
        "axis calibration references. Supports single curve (pixel_points) "
        "or multi-curves (curves), long layout (X, Y, curve) or wide layout "
        "(X, curve1, curve2...) via common X grid interpolation. Set "
        "extrapolate=False to output NaN outside each curve's domain."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "pixel_points": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                },
                "description": (
                    "Pixel points [[x, y], ...] for single curve (from "
                    "extract_plot_pixels_by_color)."
                ),
            },
            "curves": {
                "type": "array",
                "items": {"type": "object"},
                "description": (
                    "List of curve dicts each having 'points' list and "
                    "optional 'label' (e.g. from color_grouped_curves)."
                ),
            },
            "column_names": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "maxItems": 2,
                "description": (
                    "Physical quantity names for X and Y in CSV (e.g. "
                    "['V_DS_V', 'I_DS_mA'])."
                ),
                "default": ["x", "y"],
            },
            "curve_label": {
                "type": "string",
                "description": (
                    "Optional label for single curve (default: 'curve_1')."
                ),
            },
            "x_calibration": {
                "type": "object",
                "description": (
                    "X axis calibration dictionary from auto_calibrate_axes."
                ),
            },
            "y_calibration": {
                "type": "object",
                "description": (
                    "Y axis calibration dictionary from auto_calibrate_axes."
                ),
            },
            "output_csv_path": {
                "type": "string",
                "description": "Destination file path for the CSV output.",
            },
            "output_format": {
                "type": "string",
                "enum": ["long", "wide"],
                "description": (
                    "Output CSV layout: 'long' (tidy [X, Y, curve]) or "
                    "'wide' (matrix [X, curve1, curve2...])."
                ),
                "default": "long",
            },
            "resample_x_grid": {
                "type": "array",
                "items": {"type": "number"},
                "description": (
                    "Optional explicit common X grid values for resampling."
                ),
            },
            "num_grid_points": {
                "type": "integer",
                "description": (
                    "Number of interpolation points for shared X grid "
                    "(recommended for wide format, e.g. 100 or 200)."
                ),
                "default": 100,
            },
            "extrapolate": {
                "type": "boolean",
                "description": (
                    "Whether to extrapolate beyond individual curve domain "
                    "during resampling. Set False to put NaN outside X range."
                ),
                "default": True,
            },
        },
        "required": [
            "x_calibration",
            "y_calibration",
            "output_csv_path",
        ],
    },
}

RENDER_VERIFICATION_OVERLAY_SCHEMA: dict[str, Any] = {
    "name": "render_verification_overlay",
    "description": (
        "Re-project digitized CSV data back to original image pixel "
        "coordinates and render a translucent multi-color verification "
        "overlay. Computes alignment_metric (0.0-1.0) against edge map "
        "for visual validation and self-reflection."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "original_image_path": {
                "type": "string",
                "description": "Path to the cropped original plot image.",
            },
            "csv_path": {
                "type": "string",
                "description": "Path to the digitized CSV file to verify.",
            },
            "x_calibration": {
                "type": "object",
                "description": (
                    "X axis calibration dictionary from auto_calibrate_axes."
                ),
            },
            "y_calibration": {
                "type": "object",
                "description": (
                    "Y axis calibration dictionary from auto_calibrate_axes."
                ),
            },
            "output_overlay_path": {
                "type": "string",
                "description": "Destination file path for the overlay image.",
            },
            "curve_column": {
                "type": "string",
                "description": (
                    "CSV column name for grouping curves in long format "
                    "(default: 'curve')."
                ),
                "default": "curve",
            },
        },
        "required": [
            "original_image_path",
            "csv_path",
            "x_calibration",
            "y_calibration",
            "output_overlay_path",
        ],
    },
}

GET_WORKFLOW_INSTRUCTIONS_SCHEMA: dict[str, Any] = {
    "name": "get_workflow_instructions",
    "description": (
        "Retrieve best-practice workflows, recommended parameter guidelines, "
        "and troubleshooting instructions for digitize-agent. Call this tool "
        "first to determine optimal tool sequence (vector vs raster numeric "
        "vs qualitative/normalized) and self-healing strategies."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "enum": [
                    "all",
                    "vector",
                    "raster_numeric",
                    "raster_qualitative",
                    "best_practices",
                    "troubleshooting",
                ],
                "description": (
                    "Topic to retrieve: 'all' (complete instructions), "
                    "'vector' (Pattern A: native digital vector plots), "
                    "'raster_numeric' (Pattern B: raster with tick labels), "
                    "'raster_qualitative' (Pattern C: arbitrary units / "
                    "normalized / Raman / XRD / spectra plots), "
                    "'best_practices' (parameter reference table), or "
                    "'troubleshooting' (self-healing tips)."
                ),
                "default": "all",
            },
        },
        "required": [],
    },
}

ALL_SCHEMAS: list[dict[str, Any]] = [
    INSPECT_PDF_PRIMITIVES_SCHEMA,
    SEARCH_PDF_PRIMITIVES_SCHEMA,
    CROP_AND_TRANSFORM_REGION_SCHEMA,
    DETECT_AXES_AND_TICKS_SCHEMA,
    DETECT_LEGEND_REGION_SCHEMA,
    AUTO_CALIBRATE_AXES_SCHEMA,
    OCR_REGION_TEXT_SCHEMA,
    DETECT_PLOT_COLORS_SCHEMA,
    EXTRACT_PLOT_PIXELS_BY_COLOR_SCHEMA,
    EXTRACT_VECTOR_CURVE_POINTS_SCHEMA,
    CALIBRATE_AND_CONVERT_COORDINATES_SCHEMA,
    RENDER_VERIFICATION_OVERLAY_SCHEMA,
    GET_WORKFLOW_INSTRUCTIONS_SCHEMA,
]

SCHEMAS_BY_NAME: dict[str, dict[str, Any]] = {
    schema["name"]: schema for schema in ALL_SCHEMAS
}


def export_schemas_to_directory(target_dir: str | Path) -> list[Path]:
    """登録されている全ツールスキーマを個別の JSON ファイルとして書き出す。

    Args:
        target_dir: 出力先ディレクトリパス。

    Returns:
        list[Path]: 生成された JSON ファイルパスのリスト。
    """
    out_dir = Path(target_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    exported: list[Path] = []
    for schema in ALL_SCHEMAS:
        tool_name = schema["name"]
        file_path = out_dir / f"{tool_name}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2, ensure_ascii=False)
        exported.append(file_path)
    return exported
