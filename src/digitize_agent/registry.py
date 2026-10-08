"""ツール関数の登録およびディスパッチ管理モジュール。

OpenAI Function Calling からの呼び出し名と実関数をマッピングし、
安全な引数検証および実行ディスパッチを提供します。
"""

from collections.abc import Callable
from typing import Any

from digitize_agent.tools import (
    auto_calibrate_axes,
    calibrate_and_convert_coordinates,
    crop_and_transform_region,
    detect_axes_and_ticks,
    detect_legend_region,
    detect_plot_colors,
    extract_plot_pixels_by_color,
    extract_vector_curve_points,
    get_workflow_instructions,
    inspect_pdf_primitives,
    ocr_region_text,
    render_verification_overlay,
    search_pdf_primitives,
)

ToolFunction = Callable[..., dict[str, Any]]

TOOL_REGISTRY: dict[str, ToolFunction] = {
    "inspect_pdf_primitives": inspect_pdf_primitives,
    "search_pdf_primitives": search_pdf_primitives,
    "crop_and_transform_region": crop_and_transform_region,
    "detect_axes_and_ticks": detect_axes_and_ticks,
    "detect_legend_region": detect_legend_region,
    "auto_calibrate_axes": auto_calibrate_axes,
    "ocr_region_text": ocr_region_text,
    "detect_plot_colors": detect_plot_colors,
    "extract_plot_pixels_by_color": extract_plot_pixels_by_color,
    "extract_vector_curve_points": extract_vector_curve_points,
    "calibrate_and_convert_coordinates": calibrate_and_convert_coordinates,
    "render_verification_overlay": render_verification_overlay,
    "get_workflow_instructions": get_workflow_instructions,
}


def get_tool(tool_name: str) -> ToolFunction | None:
    """指定された名前のツール関数を取得する。

    Args:
        tool_name: 登録されているツール名。

    Returns:
        ToolFunction | None: 見つかったツール関数、存在しない場合は None。
    """
    return TOOL_REGISTRY.get(tool_name)


def list_tools() -> list[str]:
    """登録されている全ツール名の一覧を取得する。

    Returns:
        list[str]: ツール名のリスト。
    """
    return sorted(TOOL_REGISTRY.keys())


def dispatch_tool_call(
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """ツール名と引数辞書を受け取り、安全に関数を実行する。

    未知のツール名や実行時例外が発生した場合でもプロセスを停止させず、
    エラー情報辞書を返却します。

    Args:
        tool_name: 実行するツール関数の名称。
        arguments: ツール関数に引き渡す引数辞書。

    Returns:
        dict[str, Any]: 関数の実行結果辞書、またはエラー情報辞書。
    """
    tool_func = get_tool(tool_name)
    if tool_func is None:
        return {
            "status": "error",
            "message": (
                f"Unknown tool '{tool_name}'. Available tools: {list_tools()}"
            ),
        }

    try:
        return tool_func(**arguments)
    except Exception as exc:
        return {
            "status": "error",
            "message": f"Execution error in tool '{tool_name}': {exc}",
        }
