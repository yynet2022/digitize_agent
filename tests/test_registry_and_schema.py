"""ツールスキーマおよびディスパッチャの単体テストモジュール。

schema.py の OpenAI JSON Schema 定義妥当性、
registry.py のディスパッチ処理およびエラーハンドリングを検証します。
"""

from pathlib import Path

from digitize_agent.registry import dispatch_tool_call, get_tool, list_tools
from digitize_agent.schema import ALL_SCHEMAS, SCHEMAS_BY_NAME

EXPECTED_TOOLS = [
    "auto_calibrate_axes",
    "calibrate_and_convert_coordinates",
    "crop_and_transform_region",
    "detect_axes_and_ticks",
    "detect_legend_region",
    "detect_plot_colors",
    "extract_plot_pixels_by_color",
    "extract_vector_curve_points",
    "get_workflow_instructions",
    "inspect_pdf_primitives",
    "ocr_region_text",
    "render_verification_overlay",
    "search_pdf_primitives",
]


def test_schema_definitions() -> None:
    """定義スキーマが全13ツールと一致し構造が妥当かを検証する。"""
    assert len(ALL_SCHEMAS) == 13
    for name in EXPECTED_TOOLS:
        assert name in SCHEMAS_BY_NAME
        schema = SCHEMAS_BY_NAME[name]
        assert "name" in schema
        assert "description" in schema
        assert "parameters" in schema
        assert schema["parameters"]["type"] == "object"
        assert "properties" in schema["parameters"]


def test_registry_tool_listing() -> None:
    """レジストリが仕様で定義された全13ツールを保持していることを検証する。"""
    tools = list_tools()
    assert sorted(tools) == sorted(EXPECTED_TOOLS)

    for name in EXPECTED_TOOLS:
        func = get_tool(name)
        assert callable(func)


def test_dispatch_tool_call_success(temp_dir: Path) -> None:
    """dispatch_tool_call 経由でツール関数が正常に実行されることを検証する。"""
    out_csv = str(temp_dir / "dispatch_out.csv")
    args = {
        "pixel_points": [[0.0, 10.0], [50.0, 20.0]],
        "x_calibration": {"pixel_refs": [0.0, 100.0], "val_refs": [0.0, 10.0]},
        "y_calibration": {"pixel_refs": [0.0, 100.0], "val_refs": [0.0, 10.0]},
        "output_csv_path": out_csv,
    }

    res = dispatch_tool_call("calibrate_and_convert_coordinates", args)
    assert res.get("status") == "success"
    assert res["row_count"] == 2


def test_dispatch_unknown_tool() -> None:
    """未知のツール名をディスパッチした際にエラー辞書が返ることを検証する。"""
    res = dispatch_tool_call("non_existent_tool", {})
    assert res["status"] == "error"
    assert "Unknown tool" in res["message"]
