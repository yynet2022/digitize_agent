"""instruction_tools モジュールの単体テスト.

get_workflow_instructions の各トピック抽出、スキーマ、
および MCP サーバー・レジストリ連携を検証します。
"""

from digitize_agent.registry import dispatch_tool_call, get_tool
from digitize_agent.server import create_server
from digitize_agent.tools.instruction_tools import (
    get_workflow_instructions,
)


def test_get_workflow_instructions_default() -> None:
    """デフォルト (topic='all') で全文が正しく返却されることを検証する."""
    res = get_workflow_instructions()
    assert isinstance(res, dict)
    assert res["topic"] == "all"
    assert "Digitize Agent" in res["content"]
    assert "パターン A" in res["content"]
    assert "パターン B" in res["content"]
    assert "パターン C" in res["content"]
    assert "トラブルシューティング" in res["content"]
    assert isinstance(res["available_topics"], list)
    assert "vector" in res["available_topics"]


def test_get_workflow_instructions_vector() -> None:
    """topic='vector' でパターン A のセクションが抽出されることを検証する."""
    res = get_workflow_instructions(topic="vector")
    assert res["topic"] == "vector"
    assert "パターン A" in res["content"]
    assert "extract_vector_curve_points" in res["content"]


def test_get_workflow_instructions_raster_numeric() -> None:
    """topic='raster_numeric' でパターン B が抽出されることを検証する."""
    res = get_workflow_instructions(topic="raster_numeric")
    assert res["topic"] == "raster_numeric"
    assert "パターン B" in res["content"]
    assert "detect_axes_and_ticks" in res["content"]


def test_get_workflow_instructions_raster_qualitative() -> None:
    """topic='raster_qualitative' でパターン C が抽出されることを検証する."""
    res = get_workflow_instructions(topic="raster_qualitative")
    assert res["topic"] == "raster_qualitative"
    assert "パターン C" in res["content"]
    assert "detect_box_frame=True" in res["content"]


def test_get_workflow_instructions_troubleshooting() -> None:
    """topic='troubleshooting' で
    トラブルシューティングが抽出されることを検証する。
    """
    res = get_workflow_instructions(topic="troubleshooting")
    assert res["topic"] == "troubleshooting"
    assert "トラブルシューティング" in res["content"]


def test_get_workflow_instructions_best_practices() -> None:
    """topic='best_practices' で勘所表が抽出されることを検証する."""
    res = get_workflow_instructions(topic="best_practices")
    assert res["topic"] == "best_practices"
    assert "パラメータ指定の勘所" in res["content"]


def test_registry_dispatch_instruction_tool() -> None:
    """registry 経由で get_workflow_instructions が
    ディスパッチできることを検証する。
    """
    tool_fn = get_tool("get_workflow_instructions")
    assert tool_fn is not None

    res = dispatch_tool_call(
        "get_workflow_instructions", {"topic": "troubleshooting"}
    )
    assert "content" in res
    assert res["topic"] == "troubleshooting"


def test_server_registration() -> None:
    """MCP サーバーインスタンスに 13 ツール全て登録されていることを検証する."""
    server = create_server()
    # FastMCP の内部ツール一覧を確認
    tool_names = [t.name for t in server._tool_manager.list_tools()]
    assert "get_workflow_instructions" in tool_names
    assert len(tool_names) == 13
