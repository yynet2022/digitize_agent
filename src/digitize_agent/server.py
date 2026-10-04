"""Digitize Agent MCP (Model Context Protocol) サーバーモジュール。

学術論文や技術文書内のグラフおよび表を高精度にデジタイズするための
8つのツール関数を MCP ツールとして公開し、Claude Desktop や各種 AI
エージェントからの呼び出しを可能にします。
"""

import sys

from mcp.server.mcpserver import MCPServer

from digitize_agent.tools import (
    calibrate_and_convert_coordinates,
    crop_and_transform_region,
    detect_axes_and_ticks,
    extract_plot_pixels_by_color,
    extract_vector_curve_points,
    inspect_pdf_primitives,
    ocr_region_text,
    render_verification_overlay,
)


def create_server() -> MCPServer:
    """Digitize Agent の全ツールを登録した MCP サーバーを構築する。

    Returns:
        MCPServer: ツール登録済みの MCP サーバーインスタンス。
    """
    server = MCPServer("digitize-agent")

    # デジタイズツール群の登録
    server.add_tool(inspect_pdf_primitives)
    server.add_tool(crop_and_transform_region)
    server.add_tool(detect_axes_and_ticks)
    server.add_tool(ocr_region_text)
    server.add_tool(extract_plot_pixels_by_color)
    server.add_tool(extract_vector_curve_points)
    server.add_tool(calibrate_and_convert_coordinates)
    server.add_tool(render_verification_overlay)

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
