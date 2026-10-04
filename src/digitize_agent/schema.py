"""OpenAI Function Calling 互換のツールスキーマ定義モジュール。

デジタイズエージェントで利用可能な各ツールの JSON Schema 定義を提供します。
"""

from typing import Any

INSPECT_PDF_PRIMITIVES_SCHEMA: dict[str, Any] = {
    "name": "inspect_pdf_primitives",
    "description": (
        "Extract raw vector lines, rects, and text bounding boxes directly "
        "from a PDF page without raster degradation. Use this prior to "
        "image processing if the source is a digital PDF."
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
                    "Optional bounding box [x0, y0, x1, y1] in points to "
                    "limit extraction."
                ),
            },
        },
        "required": ["pdf_path"],
    },
}

CROP_AND_TRANSFORM_REGION_SCHEMA: dict[str, Any] = {
    "name": "crop_and_transform_region",
    "description": (
        "Crop a specific region of an image, optionally apply contrast "
        "enhancement and deskewing, and save as a high-resolution sub-image."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": "Path to the source image file.",
            },
            "bbox": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 4,
                "maxItems": 4,
                "description": (
                    "Bounding box coordinates [x_min, y_min, x_max, y_max] "
                    "in pixels."
                ),
            },
            "deskew": {
                "type": "boolean",
                "description": (
                    "Whether to detect orientation and correct slight skew."
                ),
                "default": False,
            },
            "enhance_contrast": {
                "type": "boolean",
                "description": (
                    "Apply CLAHE (Contrast Limited Adaptive Histogram "
                    "Equalization)."
                ),
                "default": False,
            },
            "output_path": {
                "type": "string",
                "description": "Destination file path for the cropped image.",
            },
        },
        "required": ["image_path", "bbox", "output_path"],
    },
}

DETECT_AXES_AND_TICKS_SCHEMA: dict[str, Any] = {
    "name": "detect_axes_and_ticks",
    "description": (
        "Detect X and Y axis line pixel positions and potential tick mark "
        "coordinates within a plot image."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": "Path to the cropped plot area image.",
            },
            "min_line_length_ratio": {
                "type": "number",
                "description": (
                    "Minimum length ratio relative to image dimensions "
                    "to qualify as an axis line (default: 0.3)."
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
        "Perform precise OCR on a local image snippet to read numbers, "
        "tick labels, or table cell texts."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": "Path to the image snippet to recognize.",
            },
            "psm": {
                "type": "integer",
                "description": (
                    "Tesseract Page Segmentation Mode (PSM). "
                    "Use 6 (single block), 7 (single text line), "
                    "or 8 (single word/number)."
                ),
                "default": 6,
            },
            "whitelist": {
                "type": "string",
                "description": (
                    "Optional character whitelist, e.g., '0123456789.-+eE' "
                    "for numeric scale reads."
                ),
            },
        },
        "required": ["image_path"],
    },
}

EXTRACT_PLOT_PIXELS_BY_COLOR_SCHEMA: dict[str, Any] = {
    "name": "extract_plot_pixels_by_color",
    "description": (
        "Extract pixel coordinates of plot curves or scatter points "
        "based on HSV color thresholding."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": "Path to the plot image.",
            },
            "hsv_lower": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 3,
                "maxItems": 3,
                "description": (
                    "Lower bound for HSV threshold "
                    "[H (0-179), S (0-255), V (0-255)]."
                ),
            },
            "hsv_upper": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 3,
                "maxItems": 3,
                "description": (
                    "Upper bound for HSV threshold "
                    "[H (0-179), S (0-255), V (0-255)]."
                ),
            },
            "extract_mode": {
                "type": "string",
                "enum": ["continuous_line", "scatter_centroids"],
                "description": (
                    "Extract continuous line (sorted by X pixel) "
                    "or discrete marker centroids."
                ),
                "default": "continuous_line",
            },
        },
        "required": ["image_path", "hsv_lower", "hsv_upper"],
    },
}

CALIBRATE_AND_CONVERT_COORDINATES_SCHEMA: dict[str, Any] = {
    "name": "calibrate_and_convert_coordinates",
    "description": (
        "Map pixel coordinates to real-world domain values using calibrated "
        "reference points on X and Y axes, and export as CSV."
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
                    "maxItems": 2,
                },
                "description": "List of pixel coordinates [[x, y], ...].",
            },
            "x_calibration": {
                "type": "object",
                "properties": {
                    "pixel_refs": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "val_refs": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "scale_type": {
                        "type": "string",
                        "enum": ["linear", "log"],
                        "default": "linear",
                    },
                },
                "required": ["pixel_refs", "val_refs"],
            },
            "y_calibration": {
                "type": "object",
                "properties": {
                    "pixel_refs": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "val_refs": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "scale_type": {
                        "type": "string",
                        "enum": ["linear", "log"],
                        "default": "linear",
                    },
                },
                "required": ["pixel_refs", "val_refs"],
            },
            "output_csv_path": {
                "type": "string",
                "description": "Destination file path for the digitized CSV.",
            },
        },
        "required": [
            "pixel_points",
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
        "as an overlay to visually verify alignment and calibration accuracy."
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
                "description": (
                    "Same calibration parameter used in "
                    "calibrate_and_convert_coordinates to map back to pixels."
                ),
            },
            "y_calibration": {
                "type": "object",
                "description": (
                    "Same calibration parameter used in "
                    "calibrate_and_convert_coordinates to map back to pixels."
                ),
            },
            "output_overlay_path": {
                "type": "string",
                "description": (
                    "Destination file path for the verification image."
                ),
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
    EXTRACT_PLOT_PIXELS_BY_COLOR_SCHEMA,
    CALIBRATE_AND_CONVERT_COORDINATES_SCHEMA,
    RENDER_VERIFICATION_OVERLAY_SCHEMA,
]

SCHEMAS_BY_NAME: dict[str, dict[str, Any]] = {
    schema["name"]: schema for schema in ALL_SCHEMAS
}
