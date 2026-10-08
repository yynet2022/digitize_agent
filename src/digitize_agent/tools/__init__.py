"""デジタイズツール群パッケージの初期化モジュール。

OpenAI Function Calling 互換の各種デジタイズ関数を提供します。
"""

from digitize_agent.tools.calibration import (
    calibrate_and_convert_coordinates,
)
from digitize_agent.tools.color_extractor import (
    detect_plot_colors,
    extract_plot_pixels_by_color,
)
from digitize_agent.tools.geometry_detect import (
    auto_calibrate_axes,
    detect_axes_and_ticks,
    detect_legend_region,
)
from digitize_agent.tools.image_transforms import (
    crop_and_transform_region,
)
from digitize_agent.tools.instruction_tools import (
    get_workflow_instructions,
)
from digitize_agent.tools.ocr_tools import (
    ocr_region_text,
)
from digitize_agent.tools.pdf_tools import (
    inspect_pdf_primitives,
    search_pdf_primitives,
)
from digitize_agent.tools.vector_curves import (
    extract_vector_curve_points,
)
from digitize_agent.tools.visual_verifier import (
    render_verification_overlay,
)

__all__ = [
    "inspect_pdf_primitives",
    "search_pdf_primitives",
    "crop_and_transform_region",
    "detect_axes_and_ticks",
    "detect_legend_region",
    "auto_calibrate_axes",
    "ocr_region_text",
    "detect_plot_colors",
    "extract_plot_pixels_by_color",
    "extract_vector_curve_points",
    "calibrate_and_convert_coordinates",
    "render_verification_overlay",
    "get_workflow_instructions",
]
