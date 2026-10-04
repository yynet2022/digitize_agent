"""MCP サーバー機能の単体テストモジュール。

create_server によるツール登録、一覧取得、
およびツール呼び出し (call_tool) を検証します。
"""

import asyncio
from pathlib import Path

from digitize_agent.server import create_server

EXPECTED_MCP_TOOLS = [
    "calibrate_and_convert_coordinates",
    "crop_and_transform_region",
    "detect_axes_and_ticks",
    "detect_plot_colors",
    "extract_plot_pixels_by_color",
    "extract_vector_curve_points",
    "inspect_pdf_primitives",
    "ocr_region_text",
    "render_verification_overlay",
]


def test_server_list_tools() -> None:
    """MCP サーバーに仕様の全 9 ツールが登録されていることを検証する。"""

    async def _test() -> None:
        server = create_server()
        tools = await server.list_tools()
        tool_names = [t.name for t in tools]
        for expected in EXPECTED_MCP_TOOLS:
            assert expected in tool_names
        assert len(tool_names) == 9

    asyncio.run(_test())


def test_server_call_tool_success(temp_dir: Path) -> None:
    """MCP サーバー経由でツール関数が正常に呼び出せることを検証する。"""

    async def _test() -> None:
        server = create_server()
        out_csv = str(temp_dir / "mcp_out.csv")

        args = {
            "pixel_points": [[0.0, 10.0], [50.0, 20.0]],
            "x_calibration": {
                "pixel_refs": [0.0, 100.0],
                "val_refs": [0.0, 10.0],
                "scale_type": "linear",
            },
            "y_calibration": {
                "pixel_refs": [0.0, 100.0],
                "val_refs": [0.0, 10.0],
                "scale_type": "linear",
            },
            "output_csv_path": out_csv,
        }

        res = await server.call_tool("calibrate_and_convert_coordinates", args)
        assert res.is_error is False
        assert len(res.content) > 0
        assert "success" in res.content[0].text

    asyncio.run(_test())


def test_server_call_tool_error_handling() -> None:
    """存在しないファイルを指定した際に安全にエラーが返ることを検証する。"""

    async def _test() -> None:
        server = create_server()
        args = {"pdf_path": "non_existent_file.pdf"}

        res = await server.call_tool("inspect_pdf_primitives", args)
        assert len(res.content) > 0
        assert "error" in res.content[0].text

    asyncio.run(_test())
