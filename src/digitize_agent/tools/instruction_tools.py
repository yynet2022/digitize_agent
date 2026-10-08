"""エージェント向け推奨ワークフロー指示書の取得ツール.

このモジュールは、digitize-agent を使用する AI エージェント向けに、
典型的なグラフ形式ごとの推奨ワークフロー、パラメータ指定指針、
およびトラブルシューティングの指示書 (instructions.md) をオンデマンドで
提供します。
"""

import importlib.resources
import pathlib
from typing import Any, Literal

from pydantic import BaseModel, Field

TopicType = Literal[
    "all",
    "vector",
    "raster_numeric",
    "raster_qualitative",
    "troubleshooting",
    "best_practices",
]


class GetWorkflowInstructionsInput(BaseModel):
    """ワークフロー指示書取得ツールの入力スキーマ."""

    topic: TopicType = Field(
        default="all",
        description=(
            "取得したい指示書トピック。"
            "'all': 全文、"
            "'vector': パターンA(電子PDFベクター)、"
            "'raster_numeric': パターンB(ラスタ数値目盛り)、"
            "'raster_qualitative': パターンC(定性/規格化/a.u.)、"
            "'troubleshooting': トラブル対処法、"
            "'best_practices': パラメータ勘所表。"
        ),
    )


def _load_raw_instructions() -> str:
    """パッケージリソースまたはローカルファイルから指示書テキストを読み込む.

    Returns:
        instructions.md の全テキスト。

    Raises:
        FileNotFoundError: instructions.md が見つからない場合。
    """
    try:
        data_res = importlib.resources.files("digitize_agent.data")
        file_path = data_res.joinpath("instructions.md")
        return file_path.read_text(encoding="utf-8")
    except Exception:
        # パッケージ未インストール時や開発環境用のフォールバック
        fallback_candidates = [
            pathlib.Path(__file__).parent.parent / "data" / "instructions.md",
            pathlib.Path(__file__).parent.parent.parent.parent
            / "INSTRUCTIONS.md",
            pathlib.Path("INSTRUCTIONS.md"),
        ]
        for candidate in fallback_candidates:
            if candidate.is_file():
                return candidate.read_text(encoding="utf-8")
        raise FileNotFoundError(
            "Workflow instructions file 'instructions.md' not found."
        )


def _extract_section(content: str, topic: TopicType) -> str:
    """指定されたトピックに合致するセクションを抽出する.

    Args:
        content: instructions.md の全文。
        topic: 抽出対象のトピック名。

    Returns:
        抽出されたマークダウンテキスト。
    """
    if topic == "all":
        return content

    lines = content.splitlines()

    section_markers: dict[str, tuple[str, list[str]]] = {
        "vector": (
            "### パターン A",
            ["### パターン B", "---", "## 2."],
        ),
        "raster_numeric": (
            "### パターン B",
            ["### パターン C", "---", "## 2."],
        ),
        "raster_qualitative": (
            "### パターン C",
            ["---", "## 2."],
        ),
        "best_practices": (
            "## 2. パラメータ指定の勘所",
            ["---", "## 3."],
        ),
        "troubleshooting": (
            "## 3. トラブルシューティング",
            [],
        ),
    }

    if topic not in section_markers:
        return content

    start_marker, end_markers = section_markers[topic]
    capturing = False
    extracted_lines: list[str] = []

    for line in lines:
        if not capturing:
            if start_marker in line:
                capturing = True
                extracted_lines.append(line)
        else:
            # 終了マーカーのいずれかに一致したら終了
            if any(end in line for end in end_markers):
                break
            extracted_lines.append(line)

    result = "\n".join(extracted_lines).strip()
    return result if result else content


def get_workflow_instructions(
    topic: TopicType = "all",
) -> dict[str, Any]:
    """digitize-agent の推奨ワークフローとベストプラクティス指示書を取得する.

    各グラフ形式（ベクター、ラスタ数値目盛り、定性・正規化グラフ）に応じた
    標準的な実行手順や、パラメータ設定の勘所、トラブルシューティング手法を
    返却します。エージェントがデジタイズ作業を開始する前に参照してください。

    Args:
        topic: 取得したい指示書トピック。
            'all': 全文
            'vector': パターン A（ベクターグラフ）
            'raster_numeric': パターン B（ラスタ数値目盛りグラフ）
            'raster_qualitative': パターン C（定性・規格化プロット）
            'troubleshooting': トラブルシューティングと自己修復
            'best_practices': パラメータ指定の勘所

    Returns:
        トピック名、抽出された指示書テキスト、利用可能なトピック一覧を含む辞書。
    """
    full_content = _load_raw_instructions()
    extracted = _extract_section(full_content, topic)

    available_topics = [
        "all",
        "vector",
        "raster_numeric",
        "raster_qualitative",
        "best_practices",
        "troubleshooting",
    ]

    return {
        "topic": topic,
        "content": extracted,
        "available_topics": available_topics,
    }
