"""Digitize Agent MCP (Model Context Protocol) サーバーモジュール。

学術論文や技術文書内のグラフおよび表を高精度にデジタイズするための
13のツール関数を MCP ツールとして公開し、AI エージェントからの
呼び出しを可能にします。
"""

import sys

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

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

# 読み取り専用ツール向け共通アノテーション設定
READ_ONLY_ANNOTATIONS = ToolAnnotations(
    read_only_hint=True,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
)

# ファイル書き出しを伴うツール向け共通アノテーション設定
FILE_WRITE_ANNOTATIONS = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
)


def create_server() -> MCPServer:
    """Digitize Agent の全ツールを登録した MCP サーバーを構築する。

    Returns:
        MCPServer: ツール登録済みの MCP サーバーインスタンス。
    """
    server = MCPServer("digitize-agent")

    # デジタイズツール群の登録 (全13ツール)
    server.add_tool(
        inspect_pdf_primitives,
        title="Inspect PDF Primitives",
        annotations=READ_ONLY_ANNOTATIONS,
    )
    server.add_tool(
        search_pdf_primitives,
        title="Search PDF Primitives",
        annotations=READ_ONLY_ANNOTATIONS,
    )
    server.add_tool(
        crop_and_transform_region,
        title="Crop and Transform Region",
        annotations=FILE_WRITE_ANNOTATIONS,
    )
    server.add_tool(
        detect_axes_and_ticks,
        title="Detect Axes and Ticks",
        annotations=READ_ONLY_ANNOTATIONS,
    )
    server.add_tool(
        detect_legend_region,
        title="Detect Legend Region",
        annotations=READ_ONLY_ANNOTATIONS,
    )
    server.add_tool(
        auto_calibrate_axes,
        title="Auto Calibrate Axes",
        annotations=READ_ONLY_ANNOTATIONS,
    )
    server.add_tool(
        ocr_region_text,
        title="OCR Region Text",
        annotations=READ_ONLY_ANNOTATIONS,
    )
    server.add_tool(
        detect_plot_colors,
        title="Detect Plot Colors",
        annotations=READ_ONLY_ANNOTATIONS,
    )
    server.add_tool(
        extract_plot_pixels_by_color,
        title="Extract Plot Pixels by Color",
        annotations=READ_ONLY_ANNOTATIONS,
    )
    server.add_tool(
        extract_vector_curve_points,
        title="Extract Vector Curve Points",
        annotations=READ_ONLY_ANNOTATIONS,
    )
    server.add_tool(
        calibrate_and_convert_coordinates,
        title="Calibrate and Convert Coordinates",
        annotations=FILE_WRITE_ANNOTATIONS,
    )
    server.add_tool(
        render_verification_overlay,
        title="Render Verification Overlay",
        annotations=FILE_WRITE_ANNOTATIONS,
    )
    server.add_tool(
        get_workflow_instructions,
        title="Get Workflow Instructions",
        annotations=READ_ONLY_ANNOTATIONS,
    )

    return server


def main() -> None:
    """MCP サーバーを stdio トランスポートで起動する。"""
    server = create_server()
    try:
        server.run(transport="stdio")
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
