# Digitize Agent

学術論文や技術文書内のグラフ（プロット）および表を、ローカル環境で自律的（Agentic）に高精度デジタイズするためのツール群および **MCP (Model Context Protocol)** サーバーです。

外部 API へのデータ送信や機械学習モデル重みの自動ダウンロードを一切排除し、完全ローカルかつ決定論的に動作するライブラリ（OpenCV, NumPy, SciPy, pandas, PyMuPDF, pdfplumber, pytesseract, matplotlib）のみで構成されています。

Claude Desktop や各種 AI コーディングアシスタント（Antigravity など）から MCP 経由で呼び出すことができるほか、OpenAI Function Calling 互換ツールや Python ライブラリとしても直接利用できます。

---

## 主な特徴

* **完全ローカル & オフライン動作**: 外部通信なしで機密文書や未発表論文も安全に処理可能。
* **MCP (Model Context Protocol) 標準対応**: FastMCP による stdio トランスポート対応。Claude Desktop 等に設定するだけで即座に連携。
* **OpenAI Function Calling 互換**: 全 7 ツールの JSON Schema 定義（`schema.py`）および安全なディスパッチャ（`registry.py`）を完備。
* **高精度な画像処理パイプライン**:
  * デジタル PDF からのベクター線・埋め込みテキスト直接抽出（解像度劣化ゼロ）
  * ハフ変換による傾き検出・自動補正 (Deskew) および CLAHE コントラスト強調
  * モルフォロジー演算とプロジェクションプロファイルによる直交座標軸・目盛りの自動特定
  * HSV 色空間閾値処理による連続線プロット・散布図マーカー（重心）抽出
  * 線形および対数（Log）スケール対応の座標キャリブレーションと CSV 出力
  * 元画像と再構築プロットの透過合成オーバーレイによる適合度検証 (Visual Feedback)
* **自己修復・エラー耐性**: パラメータ不正やファイル欠損時にもプロセスを落とさず、エージェントが再試行できる構造化エラーを返却。
* **高いコード品質**: 全コードで PEP 8・最大行長 79 文字制限・型ヒント・docstring を遵守。全 36 件の単体テストをパス（テストカバレッジ 87%）。

---

## 収録ツール一覧 (全 7 ツール)

| ツール関数名 | 役割・機能概要 |
| :--- | :--- |
| `inspect_pdf_primitives` | PDF からラスター変換を経由せず、直接埋め込まれたテキスト要素（座標・フォントサイズ）およびベクター罫線を抽出 |
| `crop_and_transform_region` | 指定バウンディングボックスの領域を切り出し、傾き自動補正 (Deskew) やコントラスト強調 (CLAHE) を適用 |
| `detect_axes_and_ticks` | グラフ画像内の主軸（水平 X 軸・垂直 Y 軸）および目盛り線（Tick marks）のピクセル座標を検出 |
| `ocr_region_text` | 切り出し画像スニペットに対して Tesseract OCR を実行し、目盛り数値や軸ラベル、表セルの文字列と認識信頼度を返却 |
| `extract_plot_pixels_by_color` | HSV 色空間の閾値に基づき、指定色プロットのピクセル座標群（連続線の中央値、または散布図マーカーの重心）を抽出 |
| `calibrate_and_convert_coordinates` | 軸の既知基準点（ピクセルと実数値）に基づき、線形または対数スケールで実数値へ変換し CSV ファイルとして保存 |
| `render_verification_overlay` | デジタイズされた CSV データを元画像の座標系へ逆変換して蛍光半透明オーバーレイ画像を生成し、適合度指標を算出 |

---

## システム要件

* **OS**: Windows / Linux / macOS
* **Python**: 3.10 以上
* **システム依存パッケージ**: `tesseract-ocr`（OCR 機能を利用する場合、ローカルにインストール済みであること）

---

## インストール手順

本リポジトリをクローンし、仮想環境を作成してインストールします。

```bash
# リポジトリのクローン
git clone https://github.com/your-username/digitize_agent.git
cd digitize_agent

# 仮想環境の作成と有効化
python -m venv .venv

# Windows (PowerShell) の場合:
.venv\Scripts\Activate.ps1
# Linux / macOS の場合:
source .venv/bin/activate

# パッケージと依存関係のインストール (編集可能モード)
pip install -e .
```

---

## 使い方

### 1. MCP サーバーとして利用する (Claude Desktop 等)

本パッケージをインストールすると、CLI コマンド `digitize-agent` が利用可能になります。

#### サーバーの手動起動確認

```bash
digitize-agent
# または
python -m digitize_agent.server
```

#### Claude Desktop の設定 (`claude_desktop_config.json`)

Claude Desktop の設定ファイルに以下を追加します。

**Windows の場合 (`%APPDATA%\Claude\claude_desktop_config.json`):**

```json
{
  "mcpServers": {
    "digitize-agent": {
      "command": "C:\\path\\to\\digitize_agent\\.venv\\Scripts\\python.exe",
      "args": ["-m", "digitize_agent.server"]
    }
  }
}
```

**macOS / Linux の場合 (`~/Library/Application Support/Claude/claude_desktop_config.json` 等):**

```json
{
  "mcpServers": {
    "digitize-agent": {
      "command": "/path/to/digitize_agent/.venv/bin/python",
      "args": ["-m", "digitize_agent.server"]
    }
  }
}
```

設定後、Claude Desktop を再起動すると、Claude がグラフや文書のデジタイズツールを自律的に呼び出せるようになります。

---

### 2. Python ライブラリとして直接利用する

Python スクリプト内から各ツール関数を直接呼び出すことも可能です。

```python
from digitize_agent.tools import (
    crop_and_transform_region,
    detect_axes_and_ticks,
    extract_plot_pixels_by_color,
    calibrate_and_convert_coordinates,
    render_verification_overlay,
)

# 1. グラフ領域の切り出し
crop_res = crop_and_transform_region(
    image_path="sample_paper.png",
    bbox=[100, 200, 600, 700],
    deskew=True,
    output_path="output/cropped_plot.png",
)

# 2. 座標軸と目盛りの検出
axes_res = detect_axes_and_ticks(
    image_path="output/cropped_plot.png",
    min_line_length_ratio=0.3,
)

# 3. 青色プロット線のピクセル抽出 (HSV 範囲指定)
pixels_res = extract_plot_pixels_by_color(
    image_path="output/cropped_plot.png",
    hsv_lower=[100, 100, 100],
    hsv_upper=[140, 255, 255],
    extract_mode="continuous_line",
)

# 4. 実数値へのキャリブレーション変換と CSV 出力
calib_res = calibrate_and_convert_coordinates(
    pixel_points=pixels_res["pixel_points"],
    x_calibration={"pixel_refs": [50.0, 450.0], "val_refs": [0.0, 100.0], "scale_type": "linear"},
    y_calibration={"pixel_refs": [450.0, 50.0], "val_refs": [0.0, 50.0], "scale_type": "linear"},
    output_csv_path="output/digitized_data.csv",
)

# 5. オーバーレイ検証画像の生成
verify_res = render_verification_overlay(
    original_image_path="output/cropped_plot.png",
    csv_path="output/digitized_data.csv",
    x_calibration={"pixel_refs": [50.0, 450.0], "val_refs": [0.0, 100.0], "scale_type": "linear"},
    y_calibration={"pixel_refs": [450.0, 50.0], "val_refs": [0.0, 50.0], "scale_type": "linear"},
    output_overlay_path="output/verification_overlay.png",
)

print(f"デジタイズ完了: {calib_res['row_count']} 件")
print(f"検証適合度指標: {verify_res['alignment_metric']}")
```

---

### 3. OpenAI Function Calling ディスパッチャの利用

LLM から返された関数名と引数辞書を安全に実行するためのディスパッチャも提供されています。

```python
from digitize_agent.registry import dispatch_tool_call
from digitize_agent.schema import ALL_SCHEMAS

# OpenAI API に ALL_SCHEMAS を tools として渡す
# ...
# LLM からツール呼び出し要求を受け取った場合:
tool_name = "detect_axes_and_ticks"
arguments = {"image_path": "output/cropped_plot.png"}

result = dispatch_tool_call(tool_name, arguments)
print(result)
```

---

## ディレクトリ構成

```text
digitize_agent/
├─ pyproject.toml              # プロジェクト設定および依存関係定義
├─ README.md                   # プロジェクト説明書
├─ SPECIFICATION.md            # 詳細設計仕様書
├─ src/
│    └─ digitize_agent/
│          ├─ __init__.py
│          ├─ server.py        # MCP (Model Context Protocol) サーバー実装
│          ├─ schema.py        # OpenAI Function Calling 互換 JSON Schema 定義
│          ├─ registry.py      # 関数ディスパッチャー (名前と実関数の安全な実行管理)
│          └─ tools/           # 各デジタイズツールの実装
│                ├─ __init__.py
│                ├─ pdf_tools.py          # Tool 1: inspect_pdf_primitives
│                ├─ image_transforms.py   # Tool 2: crop_and_transform_region
│                ├─ geometry_detect.py    # Tool 3: detect_axes_and_ticks
│                ├─ ocr_tools.py          # Tool 4: ocr_region_text
│                ├─ color_extractor.py    # Tool 5: extract_plot_pixels_by_color
│                ├─ calibration.py        # Tool 6: calibrate_and_convert_coordinates
│                └─ visual_verifier.py    # Tool 7: render_verification_overlay
└─ tests/                      # 単体テストスイート (pytest)
      ├─ conftest.py           # 合成データ・テストフィクスチャ
      ├─ test_pdf_tools.py
      ├─ test_image_transforms.py
      ├─ test_geometry_detect.py
      ├─ test_ocr_tools.py
      ├─ test_color_extractor.py
      ├─ test_calibration.py
      ├─ test_visual_verifier.py
      ├─ test_registry_and_schema.py
      └─ test_server.py
```

---

## テストとコード品質

単体テストの実行および静的解析には `pytest` と `ruff` を使用しています。

```bash
# 全テストの実行
pytest -v

# テストカバレッジの測定
pytest --cov=src/digitize_agent --cov-report=term-missing

# Ruff によるリントチェック (PEP 8・行長79文字制限等)
ruff check src tests

# Ruff によるフォーマットチェック
ruff format --check src tests
```

---

## ライセンス

本プロジェクトは MIT ライセンスの下で公開されています。
