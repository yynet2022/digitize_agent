# Digitize Agent

学術論文や技術文書内のグラフ（プロット）および表を、ローカル環境で自律的（Agentic）に高精度デジタイズするためのツール群および **MCP (Model Context Protocol)** サーバーです。

外部 API へのデータ送信や機械学習モデル重みの自動ダウンロードを一切排除し、完全ローカルかつ決定論的に動作するライブラリ（OpenCV, NumPy, SciPy, pandas, PyMuPDF, pdfplumber, pytesseract, matplotlib）のみで構成されています。

Claude Desktop や各種 AI コーディングアシスタント（Antigravity など）から MCP 経由で呼び出すことができるほか、OpenAI Function Calling 互換ツールや Python ライブラリとしても直接利用できます。

---

## 主な特徴

* **完全ローカル & オフライン動作**: 外部通信なしで機密文書や未発表論文も安全に処理可能。
* **MCP (Model Context Protocol) 標準対応**: FastMCP による stdio トランスポート対応。Claude Desktop や Antigravity 等に設定するだけで即座に連携。
* **OpenAI Function Calling 互換**: 全 12 ツールの JSON Schema 定義（`schema.py`）および安全なディスパッチャ（`registry.py`）を完備。
* **高精度な画像・ベクター処理パイプライン**:
  * PDF 内キーワード高速検索（`search_pdf_primitives`）による図表・キャプションの特定
  * キャプション bbox からの直上図表領域自動推定（`crop_and_transform_region` のスマートオートクロップ）
  * デジタル PDF からのベクター線・埋め込みテキスト直接抽出（解像度劣化ゼロ）
  * PDF ページからの直接レンダリング切り出し＆任意 DPI 指定
  * ハフ変換による傾き検出・自動補正 (Deskew) および CLAHE コントラスト強調
  * モルフォロジー演算とプロジェクションプロファイルによる直交座標軸・目盛りの自動特定
  * 目盛り線と数値ラベルの幾何学的自動ペアリング校正（`auto_calibrate_axes`）によるワンストップ校正値推定
  * プロット画像内の凡例枠・密集テキスト検出および項目代表色取得（`detect_legend_region`）
  * 電子 PDF の埋め込みベクターテキスト直接抽出を優先する OCR フォールバック（`ocr_region_text`）
  * 画像内の代表プロット色自動検知（`detect_plot_colors`）
  * 12 色プリセット、RGB 配列、Hex 値、正規化/絶対距離の双方に対応した柔軟な許容誤差による高精度色抽出（`extract_plot_pixels_by_color`）
  * PDF 内部のベクター描画命令からの座標抽出、複数描画の一括抽出（`drawing_indices`）、同一ストローク色描画の自動統合（`group_by_color`）
  * 線形および対数（Log）スケール対応の座標キャリブレーション、複数曲線の共通 X 格子線形補間（`resample_x_grid`, `num_grid_points`）、横持ち（Wide: `x, c1, c2`）および縦持ち（Long: `x, y, curve`）CSV 出力
  * 複数曲線の自動色分けパレット描画と透過合成による適合度検証 (Visual Feedback)
* **自己修復・エラー耐性**: パラメータ不正やファイル欠損時にもプロセスを落とさず、エージェントが再試行できる構造化エラーを返却。
* **高いコード品質**: 全コードで PEP 8・最大行長 79 文字制限・型ヒント・docstring を遵守。全 71 件の単体テストをパス（テストカバレッジ 90% 以上）。

---

## 収録ツール一覧 (全 12 ツール)

| ツール関数名 | 役割・機能概要 |
| :--- | :--- |
| `inspect_pdf_primitives` | PDF からラスター変換を経由せず、直接埋め込まれたテキスト要素（座標・フォントサイズ）およびベクター罫線を抽出 |
| `search_pdf_primitives` | PDF 全体または特定ページから指定キーワードを検索し、出現ページ、bbox、文脈スニペットを返却 |
| `crop_and_transform_region` | 画像または PDF から指定 DPI で領域を切り出し（キャプション bbox からの図表領域自動推定に対応）、傾き自動補正 (Deskew) やコントラスト強調 (CLAHE) を適用 |
| `detect_axes_and_ticks` | グラフ画像内の主軸（水平 X 軸・垂直 Y 軸）および目盛り線（Tick marks）のピクセル座標を検出 |
| `detect_legend_region` | プロット画像内の凡例（Legend）枠や密集テキスト領域を検出し、プロット抽出時の除外領域 (exclude_bboxes) および各凡例項目の代表色 (legend_items) を特定 |
| `auto_calibrate_axes` | 検出された目盛り線と近傍の数値テキスト（PDF埋め込みテキストまたはOCR）を幾何学的に自動照合し、X軸・Y軸のキャリブレーションパラメータ（pixel_refs, val_refs, scale_multiplier）をワンストップで自動推定 |
| `ocr_region_text` | 切り出し画像スニペットに対して Tesseract OCR を実行（PDF指定時は電子埋め込みテキストの直接抽出を優先フォールバック） |
| `detect_plot_colors` | 画像内の主要プロット色（色名、代表 HSV 値、画素占有率）を自動検出し、色抽出のための推奨設定を返却 |
| `extract_plot_pixels_by_color` | 色プリセット（12色）、RGB配列、Hex値、正規化/絶対距離の双方に対応した許容誤差、除外領域指定に基づき指定色プロットのピクセル座標群を抽出 |
| `extract_vector_curve_points` | PDF 内部のベクター描画命令から等間隔座標列をサンプリング。複数描画一括抽出 (drawing_indices)、同色描画自動統合 (group_by_color)、目盛り線除外、ストローク色指定に対応 |
| `calibrate_and_convert_coordinates` | 軸基準点に基づき線形/対数スケールで実数値へ変換。共通 X 格子への線形リサンプル、横持ち (wide) / 縦持ち (long) 形式の CSV 出力に対応 |
| `render_verification_overlay` | デジタイズされた CSV データを元画像の座標系へ逆変換して複数曲線を自動色分けした半透明オーバーレイ画像を生成し、適合度指標を算出 |

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

### 1. MCP サーバーとして利用する (Claude Desktop / Antigravity 等)

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

設定後、クライアントを再起動すると、全 12 種類のデジタイズツール群が自律的に呼び出せるようになります。

---

### 2. Python ライブラリとして直接利用する

Python スクリプト内から各ツール関数を直接呼び出すことも可能です。

#### 例 A: PDF ベクターグラフのデジタイズ（自動校正・色別統合・横持ちCSV）

```python
from digitize_agent.tools import (
    auto_calibrate_axes,
    calibrate_and_convert_coordinates,
    crop_and_transform_region,
    extract_vector_curve_points,
    render_verification_overlay,
)

# 1. キャプション bbox から直上の図表領域を自動推定して 300 DPI でクロップ
crop_res = crop_and_transform_region(
    pdf_path="paper.pdf",
    page_number=1,
    dpi=300.0,
    caption_bbox=[240, 210, 380, 225],  # キャプションの pt 座標
    output_path="output/fig_cropped.png",
)

# 2. 目盛り線と数値ラベルを幾何学照合し、X軸・Y軸の校正値を全自動推定
calib_auto = auto_calibrate_axes(
    image_path="output/fig_cropped.png",
    pdf_path="paper.pdf",
    page_number=1,
    crop_bbox_points=crop_res["estimated_bbox"],
    dpi=300.0,
)
x_calib = calib_auto["x_calibration"]
y_calib = calib_auto["y_calibration"]

# 3. 複数描画を一括抽出し、同一ストローク色ごとに自動統合
vec_res = extract_vector_curve_points(
    pdf_path="paper.pdf",
    page_number=1,
    drawing_indices=[31, 32, 33],
    group_by_color=True,
    dpi=300.0,
)

# 4. 複数曲線を共通 X 格子に補間し、横持ち (Wide) 形式で CSV 保存
calib_res = calibrate_and_convert_coordinates(
    curves=vec_res["color_grouped_curves"],
    column_names=["V_DS_V", "I_DS_mA"],
    x_calibration=x_calib,
    y_calibration=y_calib,
    output_csv_path="output/curves_wide.csv",
    output_format="wide",
    num_grid_points=100,
)

# 5. 複数曲線を色分けした透過オーバーレイ検証画像を生成
verify_res = render_verification_overlay(
    original_image_path="output/fig_cropped.png",
    csv_path="output/curves_wide.csv",
    x_calibration=x_calib,
    y_calibration=y_calib,
    output_overlay_path="output/verification_overlay.png",
)

print(
    f"変換行数: {calib_res['row_count']}, 一致率: {verify_res['alignment_metric']}"
)
```

#### 例 B: ラスター画像からの色プロット抽出（凡例除外・Hex/RGB指定）

```python
from digitize_agent.tools import (
    detect_legend_region,
    detect_plot_colors,
    extract_plot_pixels_by_color,
)

# 1. 凡例ボックスを自動検出し、除外矩形リストを取得
legend_res = detect_legend_region(image_path="plot.png")
exclude_boxes = legend_res["legend_bboxes"]

# 2. 画像内のプロット色を自動検出
color_info = detect_plot_colors(image_path="plot.png", max_colors=3)
print("検出色:", [c["color_name"] for c in color_info["dominant_colors"]])

# 3. 凡例領域を除外しながら指定色プロット線を抽出
pixel_res = extract_plot_pixels_by_color(
    image_path="plot.png",
    target_hex="#1F77B4",
    color_tolerance=35.0,
    exclude_bboxes=exclude_boxes,
    extract_mode="continuous_line",
)
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
│                ├─ pdf_tools.py          # Tool 1: inspect, Tool 2: search_pdf_primitives
│                ├─ image_transforms.py   # Tool 3: crop_and_transform_region
│                ├─ geometry_detect.py    # Tool 4: detect_axes, Tool 5: detect_legend_region
│                ├─ ocr_tools.py          # Tool 6: ocr_region_text (PDFフォールバック)
│                ├─ color_extractor.py    # Tool 7: detect_colors, Tool 8: extract_plot_pixels
│                ├─ vector_curves.py      # Tool 9: extract_vector_curve_points
│                ├─ calibration.py        # Tool 10: calibrate_and_convert_coordinates
│                └─ visual_verifier.py    # Tool 11: render_verification_overlay
└─ tests/                      # 単体テストスイート (pytest)
      ├─ conftest.py           # 合成データ・テストフィクスチャ
      ├─ test_pdf_tools.py
      ├─ test_image_transforms.py
      ├─ test_geometry_detect.py
      ├─ test_ocr_tools.py
      ├─ test_color_extractor.py
      ├─ test_vector_curves.py
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
