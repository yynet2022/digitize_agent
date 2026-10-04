"""OpenAI Function Calling 互換のツールスキーマ定義モジュール。

デジタイズエージェントで利用可能な全 9 ツールの JSON Schema 定義を提供します。
"""

from typing import Any

INSPECT_PDF_PRIMITIVES_SCHEMA: dict[str, Any] = {
    "name": "inspect_pdf_primitives",
    "description": (
        "Extract raw vector lines, rects, and text bounding boxes directly "
        "from a PDF page without raster degradation."
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
                "description": "0-indexed target page number.",
                "default": 0,
            },
            "bbox_filter": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 4,
                "maxItems": 4,
                "description": (
                    "Optional bounding box [x0, y0, x1, y1] in points."
                ),
            },
        },
        "required": ["pdf_path"],
    },
}

CROP_AND_TRANSFORM_REGION_SCHEMA: dict[str, Any] = {
    "name": "crop_and_transform_region",
    "description": (
        "Crop a specific region of an image or directly from a PDF page at "
        "custom DPI, apply deskewing and contrast enhancement."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": "Path to the source image file.",
            },
            "pdf_path": {
                "type": "string",
                "description": "Path to PDF file (alternative to image).",
            },
            "page_number": {
                "type": "integer",
                "description": "0-indexed page number if extracting from PDF.",
                "default": 0,
            },
            "dpi": {
                "type": "number",
                "description": "Rendering DPI for PDF (default 300).",
                "default": 300.0,
            },
            "bbox": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 4,
                "maxItems": 4,
                "description": "Bounding box [x_min, y_min, x_max, y_max].",
            },
            "bbox_mode": {
                "type": "string",
                "enum": ["pixel", "point"],
                "description": "Unit of bbox: 'pixel' or 'point' (72 DPI pt).",
                "default": "pixel",
            },
            "deskew": {
                "type": "boolean",
                "description": "Apply deskewing if true.",
                "default": False,
            },
            "enhance_contrast": {
                "type": "boolean",
                "description": "Apply CLAHE contrast enhancement if true.",
                "default": False,
            },
            "output_path": {
                "type": "string",
                "description": "Destination file path for the cropped image.",
            },
        },
        "required": ["bbox", "output_path"],
    },
}

DETECT_AXES_AND_TICKS_SCHEMA: dict[str, Any] = {
    "name": "detect_axes_and_ticks",
    "description": (
        "Detect horizontal and vertical coordinate axes, tick candidate "
        "positions, and bounding frame lines in a plot image."
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
                    "Minimum line length as a ratio of image dimensions."
                ),
                "default": 0.3,
            },
        },
        "required": ["image_path"],
    },
}

OCR_REGION_TEXT_SCHEMA: dict[str, Any] = {
    "name": "ocr_region_text",
    "description": (
        "Execute OCR on an image snippet to recognize axis labels, "
        "tick numbers, or table cell contents."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": "Path to the cropped snippet image.",
            },
            "psm": {
                "type": "integer",
                "description": "Tesseract Page Segmentation Mode (PSM).",
                "default": 6,
            },
            "whitelist": {
                "type": "string",
                "description": "Optional character whitelist for OCR.",
            },
        },
        "required": ["image_path"],
    },
}

DETECT_PLOT_COLORS_SCHEMA: dict[str, Any] = {
    "name": "detect_plot_colors",
    "description": (
        "Automatically detect dominant foreground plot colors, their HSV "
        "ranges, and suggested preset names from a plot image."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": "Path to the plot image file.",
            },
            "max_colors": {
                "type": "integer",
                "description": "Maximum number of dominant colors to detect.",
                "default": 5,
            },
            "min_pixel_ratio": {
                "type": "number",
                "description": "Minimum pixel ratio for plot lines.",
                "default": 0.002,
            },
        },
        "required": ["image_path"],
    },
}

EXTRACT_PLOT_PIXELS_BY_COLOR_SCHEMA: dict[str, Any] = {
    "name": "extract_plot_pixels_by_color",
    "description": (
        "Filter and extract pixel coordinates for data lines or markers of a "
        "specified color using presets or HSV thresholds."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": "Path to the plot image file.",
            },
            "color_preset": {
                "type": "string",
                "enum": ["blue", "red", "green", "orange", "black"],
                "description": "Preset color name for quick extraction.",
            },
            "hsv_lower": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 3,
                "maxItems": 3,
                "description": "HSV lower bound [H, S, V].",
            },
            "hsv_upper": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 3,
                "maxItems": 3,
                "description": "HSV upper bound [H, S, V].",
            },
            "extract_mode": {
                "type": "string",
                "enum": ["continuous_line", "scatter_centroids"],
                "description": "Extraction mode.",
                "default": "continuous_line",
            },
        },
        "required": ["image_path"],
    },
}

EXTRACT_VECTOR_CURVE_POINTS_SCHEMA: dict[str, Any] = {
    "name": "extract_vector_curve_points",
    "description": (
        "Sample dense coordinate points from vector Bézier curves or lines "
        "inside a PDF drawing, with auto-chaining and crop transformation."
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
                "description": "0-indexed PDF page number.",
                "default": 0,
            },
            "bbox_filter": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 4,
                "maxItems": 4,
                "description": "Optional bounding box in points to filter.",
            },
            "drawing_index": {
                "type": "integer",
                "description": "Optional specific drawing index.",
            },
            "curve_types": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Element types to extract: 'c' or 'l'.",
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
                "description": "Number of sample points per Bézier curve.",
                "default": 50,
            },
            "sort_x_ascending": {
                "type": "boolean",
                "description": "Sort points so X increases monotonically.",
                "default": True,
            },
            "dpi": {
                "type": "number",
                "description": "DPI to scale coordinates to pixels.",
            },
            "crop_bbox_pixels": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 4,
                "maxItems": 4,
                "description": "Optional [x0, y0, x1, y1] crop offset in px.",
            },
        },
        "required": ["pdf_path"],
    },
}

CALIBRATE_AND_CONVERT_COORDINATES_SCHEMA: dict[str, Any] = {
    "name": "calibrate_and_convert_coordinates",
    "description": (
        "Map extracted pixel coordinates to real-world domain values using "
        "linear or log scales, with column naming and multi-curve support."
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
                "description": "Pixel coordinate points [[x, y], ...].",
            },
            "curves": {
                "type": "array",
                "items": {"type": "object"},
                "description": (
                    "Optional list of curve dicts each having 'points' list."
                ),
            },
            "column_names": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "maxItems": 2,
                "description": "Custom X/Y column names in CSV.",
                "default": ["x", "y"],
            },
            "curve_label": {
                "type": "string",
                "description": "Optional label for single curve.",
            },
            "x_calibration": {
                "type": "object",
                "description": "X axis calibration parameters.",
            },
            "y_calibration": {
                "type": "object",
                "description": "Y axis calibration parameters.",
            },
            "output_csv_path": {
                "type": "string",
                "description": "Destination file path for the CSV output.",
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
        "Re-plot digitized numerical data onto the original cropped image "
        "as a multi-color semi-transparent overlay to verify alignment."
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
                "description": "Path to the digitized CSV file.",
            },
            "x_calibration": {
                "type": "object",
                "description": "X axis calibration parameters.",
            },
            "y_calibration": {
                "type": "object",
                "description": "Y axis calibration parameters.",
            },
            "output_overlay_path": {
                "type": "string",
                "description": "Destination file path for the overlay image.",
            },
            "curve_column": {
                "type": "string",
                "description": "CSV column name for grouping curves.",
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

ALL_SCHEMAS: list[dict[str, Any]] = [
    INSPECT_PDF_PRIMITIVES_SCHEMA,
    CROP_AND_TRANSFORM_REGION_SCHEMA,
    DETECT_AXES_AND_TICKS_SCHEMA,
    OCR_REGION_TEXT_SCHEMA,
    DETECT_PLOT_COLORS_SCHEMA,
    EXTRACT_PLOT_PIXELS_BY_COLOR_SCHEMA,
    EXTRACT_VECTOR_CURVE_POINTS_SCHEMA,
    CALIBRATE_AND_CONVERT_COORDINATES_SCHEMA,
    RENDER_VERIFICATION_OVERLAY_SCHEMA,
]

SCHEMAS_BY_NAME: dict[str, dict[str, Any]] = {
    schema["name"]: schema for schema in ALL_SCHEMAS
}
