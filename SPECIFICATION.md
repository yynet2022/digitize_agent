# Agentic Plot & Document Digitization ツール仕様書

本仕様書は、学術論文や技術文書内のグラフ（プロット）および表を、ローカル環境で自律的（Agentic）に高精度デジタイズするためのVLM Tool（OpenAI Function Calling互換）および MCP (Model Context Protocol) サーバー対応ツール群の設計仕様です。

外部API送信およびモデル重みの自動ダウンロード（Hugging Face Hub等へのアクセス）を伴うモジュールを一切排除し、完全ローカルかつ決定論的に動作するライブラリ（`OpenCV`, `NumPy`, `SciPy`, `pandas`, `PyMuPDF`, `pdfplumber`, `pytesseract`, `matplotlib`, `mcp`）のみで構成しています。

---

## 1. システム要件および依存関係

### 実行環境

* OS: Windows / Linux / macOS
* Python: 3.10以上
* 外部ネットワーク通信: なし（完全オフライン動作可能）
* システム依存パッケージ: `tesseract-ocr`（ローカルインストール済みであること）

### 依存パッケージ (`pyproject.toml`)

```toml
dependencies = [
    "numpy>=1.24.0",
    "scipy>=1.10.0",
    "pandas>=2.0.0",
    "opencv-python>=4.8.0",
    "pillow>=10.0.0",
    "PyMuPDF>=1.23.0",
    "pdfplumber>=0.10.0",
    "pytesseract>=0.3.10",
    "matplotlib>=3.7.0",
    "pydantic>=2.0.0",
    "mcp>=1.0.0",
]
```

---

## 2. ツール一覧とパイプライン構成

```text
[入力ドキュメント (PDF / 画像)]
   │
   ├─ (PDF の場合: 構造解析・検索)
   │     ├─> search_pdf_primitives (キーワード検索による図表・キャプションのページ・bbox高速特定)
   │     └─> inspect_pdf_primitives (埋め込みテキスト・ベクター直接抽出)
   │
   ├─ パターン A: PDF ベクター描画グラフ (ベジェ曲線)
   │     │
   │     ├──> crop_and_transform_region (PDF直接レンダリング・DPI指定クロップ・キャプション自動推定)
   │     │       ├──> detect_axes_and_ticks (軸線および目盛りピクセル検出)
   │     │       ├──> detect_legend_region (凡例枠・テキスト領域の検出・除外範囲特定・項目色判定)
   │     │       └──> auto_calibrate_axes (目盛り線と数値ラベルの幾何学的自動ペアリング校正)
   │     │
   │     ├──> extract_vector_curve_points (PDFベクターパスからサンプリング・色別一括統合抽出)
   │     │
   │     ├──> calibrate_and_convert_coordinates (物理量変換・共通X格子線形補間・Wide/Long CSV 出力)
   │     │
   │     └──> render_verification_overlay (複数曲線色分け透過オーバーレイ検証)
   │
   └─ パターン B: ラスター画像グラフ (スキャン文書・写真)
         │
         ├──> crop_and_transform_region (領域クロップ・傾き自動補正・コントラスト強調)
         │       │
         │       ├──> detect_axes_and_ticks (座標軸・目盛りピクセル検出)
         │       │
         │       ├──> detect_legend_region (凡例ボックスの検出、除外矩形・項目色取得)
         │       │
         │       ├──> auto_calibrate_axes (目盛り線とOCRテキストの自動ペアリング校正)
         │       │
         │       ├──> ocr_region_text (軸ラベル・目盛り数値の認識・PDFテキストフォールバック)
         │       │
         │       ├──> detect_plot_colors (画像内の主要プロット色を自動検出)
         │       │
         │       └──> extract_plot_pixels_by_color (RGB/Hex/柔軟な許容誤差/除外領域指定による点抽出)
         │
         ├──> calibrate_and_convert_coordinates (実数値変換・共通X格子リサンプル・列名指定 CSV 出力)
         │
         └──> render_verification_overlay (再描画と元画像の重ね合わせ検証)
```

### 全 12 ツール一覧表

| ツール関数名 | 役割・機能概要 |
| :--- | :--- |
| `inspect_pdf_primitives` | PDF からラスター変換を経由せず、直接埋め込まれたテキスト要素およびベクター罫線を抽出 |
| `search_pdf_primitives` | PDF 全体または特定ページから指定キーワードを検索し、出現ページ、bbox、文脈スニペットを返却 |
| `crop_and_transform_region` | 画像または PDF から直接指定 DPI で領域を切り出し（キャプション bbox からの図表領域自動推定にも対応）、傾き補正 (Deskew) や強調 (CLAHE) を適用 |
| `detect_axes_and_ticks` | グラフ画像内の主軸（水平 X 軸・垂直 Y 軸）および目盛り線（Tick marks）のピクセル座標を幾何学的に検出 |
| `detect_legend_region` | プロット画像内の凡例（Legend）矩形枠やテキストブロックを検出し、プロット抽出時の除外領域 (exclude_bboxes) および凡例項目ごとの代表色 (legend_items) を特定 |
| `auto_calibrate_axes` | 検出された目盛り線と近傍の数値テキスト（PDF埋め込みテキストまたはOCR）を幾何学的に自動照合し、X軸・Y軸のキャリブレーションパラメータ（pixel_refs, val_refs, scale_multiplier）をワンストップで自動推定 |
| `ocr_region_text` | 切り出し画像スニペットに対して Tesseract OCR を実行（PDF指定時は電子埋め込みテキストの直接抽出を優先フォールバック） |
| `detect_plot_colors` | 画像内の主要プロット色（色名、代表 HSV 値、画素占有率）を自動検出し、色抽出のための推奨設定を提示 |
| `extract_plot_pixels_by_color` | 色名プリセット、RGB配列、Hex値、正規化/絶対距離の双方に対応した柔軟な許容誤差 (color_tolerance)、抽出領域 (bbox)、除外領域 (exclude_bboxes) に基づきプロット点を高精度抽出 |
| `extract_vector_curve_points` | PDF 内部のベクター描画命令から等間隔座標列をサンプリング。複数描画の一括抽出 (drawing_indices)、同色描画の自動統合 (group_by_color)、目盛り線除外、特定色線指定に対応 |
| `calibrate_and_convert_coordinates` | 軸基準点に基づき線形/対数スケールで実数値へ変換。共通 X 格子への線形リサンプリング (resample_x_grid, num_grid_points) および横持ち (wide) / 縦持ち (long) CSV 出力に対応 |
| `render_verification_overlay` | デジタイズされた CSV データを元画像の座標系へ逆変換して複数曲線を自動色分けした半透明オーバーレイ画像を生成し、適合度指標を算出 |

---

## 3. 各ツール関数の詳細仕様

### Tool 1: `inspect_pdf_primitives`

* **ツール関数名**: `inspect_pdf_primitives`
* **内部使用モジュール**: `pymupdf` (`fitz`), `pdfplumber`
* **役割・機能**:
PDFファイルからラスター変換を経由せず、直接埋め込まれているテキスト要素（フォント、バウンディングボックス座標）およびベクターパス（罫線、四角形）を抽出します。スキャン画像ではない真正PDFにおいて、解像度低下やOCR誤認識を回避して100%の再現性を確保します。
* **OpenAI Function Calling 定義 (JSON Schema)**:

```json
{
  "name": "inspect_pdf_primitives",
  "description": "Extract raw vector lines, rects, and text bounding boxes directly from a PDF page without raster degradation.",
  "parameters": {
    "type": "object",
    "properties": {
      "pdf_path": {
        "type": "string",
        "description": "Absolute or relative path to the local PDF file."
      },
      "page_number": {
        "type": "integer",
        "description": "0-indexed target page number.",
        "default": 0
      },
      "bbox_filter": {
        "type": "array",
        "items": {"type": "number"},
        "minItems": 4,
        "maxItems": 4,
        "description": "Optional bounding box [x0, y0, x1, y1] in points to limit extraction."
      }
    },
    "required": ["pdf_path"]
  }
}
```

* **入出力仕様**:
  * **引数**:
    * `pdf_path` (str): 対象PDFファイルのローカルパス。
    * `page_number` (int, 任意, 初期値: 0): 対象ページ。
    * `bbox_filter` (list[float], 任意): `[x0, y0, x1, y1]` で指定する抽出範囲（pt単位）。
  * **戻り値 (dict)**:
    * `is_scanned` (bool): テキストが一切埋め込まれていない場合はTrue。
    * `text_elements` (list[dict]): `{"text": str, "bbox": [x0, y0, x1, y1], "font_size": float}` の配列。
    * `vector_lines` (list[dict]): `{"coords": [x0, y0, x1, y1], "width": float}` の配列。
* **実装要件・アルゴリズム**:
  * `pymupdf.open(pdf_path)` でドキュメントを読み込み。
  * `page.get_text("dict")` により座標・サイズ付きテキストスパンを取得。
  * `pdfplumber` の `page.lines` を用いてベクター罫線を抽出。

---

### Tool 2: `search_pdf_primitives`

* **ツール関数名**: `search_pdf_primitives`
* **内部使用モジュール**: `pymupdf` (`fitz`)
* **役割・機能**:
PDF ドキュメント内から指定されたキーワード（大文字小文字無視）を高速にテキスト検索し、マッチした箇所の出現ページ番号、バウンディングボックス座標 `[x0, y0, x1, y1]`、および前後の文脈スニペットを返却します。論文中の "Figure 5" や "Fig. 3" などのキャプション位置をピンポイントで特定し、その近傍のプロット領域を正確に切り出すための前処理として機能します。
* **OpenAI Function Calling 定義 (JSON Schema)**:

```json
{
  "name": "search_pdf_primitives",
  "description": "Search for keywords across a PDF document or a specific page to locate target figures, tables, or sections, returning page numbers, bounding boxes, and surrounding text snippets.",
  "parameters": {
    "type": "object",
    "properties": {
      "pdf_path": {
        "type": "string",
        "description": "Absolute or relative path to the local PDF file."
      },
      "query": {
        "type": "string",
        "description": "Keyword or text snippet to search for (case-insensitive)."
      },
      "page_number": {
        "type": "integer",
        "description": "Optional 0-indexed page number to restrict search. If omitted, searches all pages."
      },
      "max_results": {
        "type": "integer",
        "description": "Maximum number of search results to return.",
        "default": 10
      }
    },
    "required": ["pdf_path", "query"]
  }
}
```

* **入出力仕様**:
  * **引数**:
    * `pdf_path` (str): 対象 PDF ファイルのローカルパス。
    * `query` (str): 検索語句（大文字小文字は区別されません）。
    * `page_number` (int, 任意): 検索対象ページ番号（0始まり）。省略時は全ページ対象。
    * `max_results` (int, 任意, 初期値: 10): 最大取得件数。
  * **戻り値 (dict)**:
    * `status` (str): `"success"`。
    * `query` (str): 検索された語句。
    * `total_matches` (int): ヒットした件数。
    * `matches` (list[dict]): 各ヒット情報のリスト。
      * `page`: ページ番号（0始まり）。
      * `bbox`: マッチ箇所のバウンディングボックス `[x0, y0, x1, y1]`（pt単位）。
      * `snippet`: マッチ箇所を含む行や文脈の抜粋テキスト。
* **実装要件・アルゴリズム**:
  * `pymupdf.open(pdf_path)` でドキュメントを走査。
  * `page.search_for(query)` により矩形座標リストを取得。
  * 各マッチ矩形を含むテキストブロックまたは行から文脈スニペットを抽出。

---

### Tool 3: `crop_and_transform_region`

* **ツール関数名**: `crop_and_transform_region`
* **内部使用モジュール**: `cv2` (OpenCV), `numpy`, `pymupdf` (`fitz`)
* **役割・機能**:
画像ファイル、または PDF ファイルから直接指定 DPI でレンダリングして指定領域を切り出します。バウンディングボックス (`bbox`) の直接指定に加え、キャプションのバウンディングボックス (`caption_bbox`) を渡すことで直上の図表描画領域を自動推定してクロップするスマートオートクロップに対応しています。傾き（Deskew）の自動検出と補正、CLAHE コントラスト強調を行い、解析用サブ画像を生成します。
* **OpenAI Function Calling 定義 (JSON Schema)**:

```json
{
  "name": "crop_and_transform_region",
  "description": "Crop a specific region of an image or directly from a PDF page at custom DPI, apply deskewing and contrast enhancement.",
  "parameters": {
    "type": "object",
    "properties": {
      "image_path": {
        "type": "string",
        "description": "Path to the source image file."
      },
      "pdf_path": {
        "type": "string",
        "description": "Path to source PDF file (alternative to image)."
      },
      "page_number": {
        "type": "integer",
        "description": "0-indexed page number if extracting from PDF.",
        "default": 0
      },
      "dpi": {
        "type": "number",
        "description": "Rendering resolution DPI for PDF (default 300).",
        "default": 300.0
      },
      "bbox": {
        "type": "array",
        "items": {"type": "number"},
        "minItems": 4,
        "maxItems": 4,
        "description": "Bounding box coordinates [x_min, y_min, x_max, y_max]."
      },
      "caption_bbox": {
        "type": "array",
        "items": {"type": "number"},
        "minItems": 4,
        "maxItems": 4,
        "description": "Bounding box [x0, y0, x1, y1] in points of caption to auto-estimate figure region."
      },
      "bbox_mode": {
        "type": "string",
        "enum": ["pixel", "point"],
        "description": "Unit of bbox: 'pixel' or 'point' (72 DPI pt).",
        "default": "pixel"
      },
      "deskew": {
        "type": "boolean",
        "description": "Whether to detect orientation and correct slight skew.",
        "default": false
      },
      "enhance_contrast": {
        "type": "boolean",
        "description": "Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).",
        "default": false
      },
      "output_path": {
        "type": "string",
        "description": "Destination file path for the cropped image."
      }
    },
    "required": ["output_path"]
  }
}
```

* **入出力仕様**:
  * **引数**:
    * `output_path` (str): 保存先ファイルパス。
    * `bbox` (list[float], 任意): `[x_min, y_min, x_max, y_max]`。省略時は `caption_bbox` が必須。
    * `caption_bbox` (list[float], 任意): `[x0, y0, x1, y1]`（pt単位）。指定時はキャプション直上のベクター描画領域を自動探索して切り出し範囲を推定。
    * `image_path` (str, 任意): 元画像パス。
    * `pdf_path` (str, 任意): 元PDFパス（指定時はPDFから直接レンダリング）。
    * `page_number` (int, 任意, 初期値: 0): PDFの対象ページ番号。
    * `dpi` (float, 任意, 初期値: 300.0): PDFレンダリング解像度。
    * `bbox_mode` (str, 任意, 初期値: "pixel"): `"pixel"` または `"point"` (72pt基準)。
    * `deskew` (bool, 任意, 初期値: False): 傾き補正フラグ。
    * `enhance_contrast` (bool, 任意, 初期値: False): コントラスト強調フラグ。
  * **戻り値 (dict)**:
    * `status` (str): `"success"` または `"error"`。
    * `output_path` (str): 保存先パス。
    * `dimensions` (dict): `{"width": int, "height": int}`。
    * `skew_angle_detected` (float): 検出された傾き角度（度数法）。
    * `estimated_bbox` (list[float], 任意): `caption_bbox` から自動推定された領域の `[x0, y0, x1, y1]`（pt単位）。
* **実装要件・アルゴリズム**:
  * `caption_bbox` 指定時は、PDF ページのベクター描画矩形群（`page.get_drawings()`）を走査し、キャプション直上（search_top〜c_y0）に存在する描画要素の外接矩形に適切なパディング（幅±35pt、高さ±15pt）を加えて図表領域を自動推定。
  * `pdf_path` 指定時は `page.get_pixmap(dpi=int(round(dpi)))` で高解像度レンダリング。
  * `bbox_mode == "point"`（または自動推定時）の場合は `dpi / 72.0` を乗じてピクセル座標へ換算。
  * 傾き検出: Canny エッジ検出と確率的ハフ変換（`cv2.HoughLinesP`）による傾斜中央値の算出およびアフィン回転補正。
  * コントラスト強調: LAB色空間のLチャンネルに対するCLAHE適用。

---

### Tool 4: `detect_axes_and_ticks`

* **ツール関数名**: `detect_axes_and_ticks`
* **内部使用モジュール**: `cv2` (OpenCV), `numpy`, `scipy.signal`
* **役割・機能**:
グラフ画像内の主軸（X軸・Y軸）となる直線と、それに付随する目盛り線（Tick marks）のピクセル位置を幾何学的に特定します。
* **OpenAI Function Calling 定義 (JSON Schema)**:

```json
{
  "name": "detect_axes_and_ticks",
  "description": "Detect horizontal and vertical coordinate axes, tick candidate positions, and bounding frame lines in a plot image.",
  "parameters": {
    "type": "object",
    "properties": {
      "image_path": {
        "type": "string",
        "description": "Path to the cropped plot area image."
      },
      "min_line_length_ratio": {
        "type": "number",
        "description": "Minimum length ratio relative to image dimensions (default: 0.3).",
        "default": 0.3
      }
    },
    "required": ["image_path"]
  }
}
```

* **入出力仕様**:
  * **引数**:
    * `image_path` (str): グラフ領域の画像パス。
    * `min_line_length_ratio` (float, 初期値: 0.3): 軸線と見なす最小長さ比。
  * **戻り値 (dict)**:
    * `x_axis`: `{"y_pixel": int, "x_range": [int, int]}` （X軸のYピクセル位置とX範囲）
    * `y_axis`: `{"x_pixel": int, "y_range": [int, int]}` （Y軸のXピクセル位置とY範囲）
    * `x_tick_candidates`: list[int] （検出された目盛りのXピクセル座標群）
    * `y_tick_candidates`: list[int] （検出された目盛りのYピクセル座標群）

---

### Tool 5: `detect_legend_region`

* **ツール関数名**: `detect_legend_region`
* **内部使用モジュール**: `cv2` (OpenCV), `numpy`
* **役割・機能**:
切り出したプロット画像の中から、凡例（Legend）ボックスの矩形枠線、または複数行のテキストやマーカーが密集した凡例候補領域を輪郭検出および連結成分解析により自動検出します。検出された矩形バウンディングボックス群は、色抽出ツール `extract_plot_pixels_by_color` の `exclude_bboxes` に直接渡すことで、凡例マーカーや説明テキストがデータ曲線として誤抽出されるのを防ぎます。
* **OpenAI Function Calling 定義 (JSON Schema)**:

```json
{
  "name": "detect_legend_region",
  "description": "Detect legend bounding boxes or text-dense annotation regions within a plot image to exclude them during curve pixel extraction.",
  "parameters": {
    "type": "object",
    "properties": {
      "image_path": {
        "type": "string",
        "description": "Path to the cropped plot image."
      },
      "min_area_ratio": {
        "type": "number",
        "description": "Minimum area ratio relative to image size to qualify as a legend box.",
        "default": 0.01
      },
      "max_area_ratio": {
        "type": "number",
        "description": "Maximum area ratio relative to image size.",
        "default": 0.35
      }
    },
    "required": ["image_path"]
  }
}
```

* **入出力仕様**:
  * **引数**:
    * `image_path` (str): グラフ領域の画像パス。
    * `min_area_ratio` (float, 初期値: 0.01): 画像全体に対する凡例枠の最小面積比率。
    * `max_area_ratio` (float, 初期値: 0.35): 画像全体に対する凡例枠の最大面積比率。
  * **戻り値 (dict)**:
    * `status` (str): `"success"`。
    * `legends_detected` (int): 検出された凡例領域の総数。
    * `legend_bboxes` (list[list[int]]): 検出された凡例の矩形座標 `[x0, y0, x1, y1]` のリスト（`exclude_bboxes` 互換形式）。
    * `candidates` (list[dict]): 各候補領域のバウンディングボックス、面積比等の詳細情報。
    * `legend_items` (list[dict]): 各凡例項目の詳細情報。テキスト近傍のプロット線マーカー色サンプリング情報（`color_rgb`, `color_hex`）を含み、凡例ラベルと曲線色の自動紐付けに利用可能。

---

### Tool 6: `auto_calibrate_axes`

* **ツール関数名**: `auto_calibrate_axes`
* **内部使用モジュール**: `cv2` (OpenCV), `numpy`, `pymupdf` (`fitz`)
* **役割・機能**:
切り出したプロット画像内の軸・目盛り線と、近傍の数値テキスト（PDF埋め込みテキストまたはOCRテキスト）を幾何学的に自動照合し、X軸およびY軸のキャリブレーションパラメータ（`pixel_refs`, `val_refs`）とスケール乗数（$10^{32}$, $10^3$ 等）をワンストップで自動推定します。人間やエージェントが個別に目盛り座標と数値を読み取ってペアリングする作業を全自動化します。
* **OpenAI Function Calling 定義 (JSON Schema)**:

```json
{
  "name": "auto_calibrate_axes",
  "description": "Automatically detect coordinate axes, tick marks, and nearby numerical text in an image or PDF to derive calibration parameters (pixel_refs, val_refs, scale_multiplier).",
  "parameters": {
    "type": "object",
    "properties": {
      "image_path": {
        "type": "string",
        "description": "Path to the cropped plot image file."
      },
      "pdf_path": {
        "type": "string",
        "description": "Optional path to original PDF for precise text reading."
      },
      "page_number": {
        "type": "integer",
        "description": "0-indexed PDF page number.",
        "default": 0
      },
      "crop_bbox_points": {
        "type": "array",
        "items": {"type": "number"},
        "minItems": 4,
        "maxItems": 4,
        "description": "Optional crop bounding box [x0, y0, x1, y1] in pt."
      },
      "dpi": {
        "type": "number",
        "description": "Resolution of cropped image (default 300).",
        "default": 300.0
      },
      "x_tick_candidates": {
        "type": "array",
        "items": {"type": "integer"},
        "description": "Optional pre-detected X tick pixel positions."
      },
      "y_tick_candidates": {
        "type": "array",
        "items": {"type": "integer"},
        "description": "Optional pre-detected Y tick pixel positions."
      }
    },
    "required": ["image_path"]
  }
}
```

* **入出力仕様**:
  * **引数**:
    * `image_path` (str): クロップされたプロット画像パス。
    * `pdf_path` (str, 任意): 元 PDF パス（埋め込みテキストの直接取得用、推奨）。
    * `page_number` (int, 任意, 初期値: 0): PDF のページ番号 (0-indexed)。
    * `crop_bbox_points` (list[float], 任意): 切り出し時の PDF point バウンディングボックス。
    * `dpi` (float, 任意, 初期値: 300.0): クロップ画像の解像度 (DPI)。
    * `x_tick_candidates` (list[int], 任意): 事前検出された X 目盛り候補ピクセル列。
    * `y_tick_candidates` (list[int], 任意): 事前検出された Y 目盛り候補ピクセル列。
  * **戻り値 (dict)**:
    * `status` (str): `"success"` または `"error"`。
    * `x_calibration` (dict): `{"pixel_refs": [float, float], "val_refs": [float, float], "scale_type": "linear"}`。
    * `y_calibration` (dict): `{"pixel_refs": [float, float], "val_refs": [float, float], "scale_type": "linear"}`。
    * `scale_multiplier` (float | None): 軸ラベル等から検出された乗数（例: $10^{32} \rightarrow 1.0\times 10^{32}$）。
    * `matched_x_ticks` (list[dict]): ペアリングされた X 目盛りピクセルと数値。
    * `matched_y_ticks` (list[dict]): ペアリングされた Y 目盛りピクセルと数値。
* **実装要件・アルゴリズム**:
  * `detect_axes_and_ticks` により画像内の X 軸・Y 軸位置および目盛り線を検出。
  * `pdf_path` 指定時は `page.get_text("dict")` を用いて解像度劣化のない正確なテキスト座標を取得し、クロップオフセットと DPI スケールを適用して画像ピクセル座標系へ投影。
  * 正規表現による指数表記・乗数テキスト（`1eX`, `10^X`, `×10^X` 等）の解析。
  * 各目盛り線ピクセルから許容距離内（X軸: 横方向近傍かつ軸直下、Y軸: 縦方向近傍かつ軸左側）に位置する数値をユークリッド距離最小化で幾何学マッチング。
  * 得られた照合ペアから両端の代表 2 点を選定して `pixel_refs` と `val_refs` を構成。

---

### Tool 7: `ocr_region_text`

* **ツール関数名**: `ocr_region_text`
* **内部使用モジュール**: `pytesseract`, `PIL` (Pillow), `cv2`, `pymupdf` (`fitz`)
* **役割・機能**:
切り出した領域（目盛りの数値部分、軸ラベル、凡例など）に対してテキストを認識します。`pdf_path` および `pdf_bbox` が指定された場合は、解像度劣化やOCR誤認識を防ぐため、電子PDF内部の埋め込みベクターテキストの直接抽出を優先試行（確信度 100.0）し、該当テキストが存在しない場合に Tesseract OCR へ自動フォールバックします。
* **OpenAI Function Calling 定義 (JSON Schema)**:

```json
{
  "name": "ocr_region_text",
  "description": "Execute OCR on an image snippet to recognize axis labels, tick numbers, or table cell contents. If pdf_path and pdf_bbox are provided, attempts direct text extraction first.",
  "parameters": {
    "type": "object",
    "properties": {
      "image_path": {
        "type": "string",
        "description": "Path to the cropped snippet image."
      },
      "psm": {
        "type": "integer",
        "description": "Tesseract PSM mode (6: single block, 7: single line, 8: single word).",
        "default": 6
      },
      "whitelist": {
        "type": "string",
        "description": "Optional character whitelist, e.g., '0123456789.-+eE'."
      },
      "pdf_path": {
        "type": "string",
        "description": "Optional path to the source PDF file for direct text extraction fallback."
      },
      "page_number": {
        "type": "integer",
        "description": "0-indexed PDF page number.",
        "default": 0
      },
      "pdf_bbox": {
        "type": "array",
        "items": {"type": "number"},
        "minItems": 4,
        "maxItems": 4,
        "description": "Optional PDF point bounding box [x0, y0, x1, y1] for direct text extraction."
      }
    },
    "required": ["image_path"]
  }
}
```

* **入出力仕様**:
  * **引数**:
    * `image_path` (str): 入力画像パス。
    * `psm` (int, 初期値: 6): Tesseract PSMモード。
    * `whitelist` (str, 任意): 許可する文字一覧。
    * `pdf_path` (str, 任意): 元 PDF のローカルパス。
    * `page_number` (int, 任意, 初期値: 0): 対象ページ。
    * `pdf_bbox` (list[float], 任意): 抽出対象の PDF ポイント座標矩形 `[x0, y0, x1, y1]`。
  * **戻り値 (dict)**:
    * `status` (str): `"success"`。
    * `text` (str): 生テキスト。
    * `clean_text` (str): 空白・改行整形テキスト。
    * `confidence` (float): テキスト確信度スコア（0.0〜100.0、PDF直接抽出時は 100.0）。
    * `source` (str): テキスト取得元（`"pdf_vector"` または `"tesseract_ocr"`）。

---

### Tool 8: `detect_plot_colors`

* **ツール関数名**: `detect_plot_colors`
* **内部使用モジュール**: `cv2` (OpenCV), `numpy`
* **役割・機能**:
画像内の白背景や無彩色ノイズを除外し、プロット曲線として存在しそうな主要色候補をK-Meansクラスタリングにより自動検出します。各色の代表HSV値、推奨HSV閾値範囲、直感的な色名ラベル（'blue', 'red', 'green' 等）および画素占有率を返します。
* **OpenAI Function Calling 定義 (JSON Schema)**:

```json
{
  "name": "detect_plot_colors",
  "description": "Automatically detect dominant foreground plot colors, their HSV ranges, and suggested preset names from a plot image.",
  "parameters": {
    "type": "object",
    "properties": {
      "image_path": {
        "type": "string",
        "description": "Path to the plot image file."
      },
      "max_colors": {
        "type": "integer",
        "description": "Maximum number of dominant colors to detect.",
        "default": 5
      },
      "min_pixel_ratio": {
        "type": "number",
        "description": "Minimum pixel ratio for plot lines.",
        "default": 0.002
      }
    },
    "required": ["image_path"]
  }
}
```

* **入出力仕様**:
  * **引数**:
    * `image_path` (str): 対象画像のファイルパス。
    * `max_colors` (int, 初期値: 5): 検出する最大色数。
    * `min_pixel_ratio` (float, 初期値: 0.002): プロット線とみなす最小画素占有比率。
  * **戻り値 (dict)**:
    * `status` (str): `"success"`。
    * `detected_colors_count` (int): 検出された色数。
    * `dominant_colors` (list[dict]): 各色情報辞書のリスト。
      * `color_name` (str): 判定色名（'blue', 'red', 'green', 'black' 等）。
      * `pixel_count` (int): 該当画素数。
      * `pixel_ratio` (float): 画像全体に対する占有率。
      * `representative_hsv` (list[float]): 代表 HSV 値。
      * `suggested_hsv_lower` (list[int]): 推奨 HSV 下限。
      * `suggested_hsv_upper` (list[int]): 推奨 HSV 上限。

---

### Tool 9: `extract_plot_pixels_by_color`

* **ツール関数名**: `extract_plot_pixels_by_color`
* **内部使用モジュール**: `cv2` (OpenCV), `numpy`
* **役割・機能**:
代表色名プリセット、RGB配列 `[R, G, B]`、Hexカラーコード `"#RRGGBB"`、または手動の HSV 色閾値に基づき、指定色プロットのピクセル座標群を高精度に抽出します。抽出対象領域の限定 (`bbox`) や凡例等の除外矩形リスト (`exclude_bboxes`)、色差の許容誤差 (`color_tolerance`) を指定可能です。`color_tolerance` は `0.001〜1.0` の正規化値（最大ユークリッド距離に対する割合）および `1.0〜255.0` の RGB 絶対距離の両方を自動判別して柔軟に受理します。ノイズ除去（開閉演算）を行い、連続線または散布図マーカーの重心座標を配列化して返します。
* **OpenAI Function Calling 定義 (JSON Schema)**:

```json
{
  "name": "extract_plot_pixels_by_color",
  "description": "Filter and extract pixel coordinates for data lines or markers of a specified color using presets, target RGB/Hex, or HSV thresholds, with bbox restrictions and legend exclusion.",
  "parameters": {
    "type": "object",
    "properties": {
      "image_path": {
        "type": "string",
        "description": "Path to the plot image file."
      },
      "color_preset": {
        "type": "string",
        "enum": [
          "blue", "red", "green", "orange", "black",
          "cyan", "magenta", "yellow", "purple",
          "brown", "pink", "gray"
        ],
        "description": "Preset color name for quick extraction."
      },
      "target_rgb": {
        "type": "array",
        "items": {"type": "integer"},
        "minItems": 3,
        "maxItems": 3,
        "description": "Target color in RGB [R, G, B] (0-255)."
      },
      "target_hex": {
        "type": "string",
        "description": "Target color in Hex format, e.g. '#1F77B4' or 'FF5733'."
      },
      "color_tolerance": {
        "type": "number",
        "description": "Color tolerance factor (0.01-1.0 normalized or 1.0-255.0 Euclidean distance).",
        "default": 35.0
      },
      "hsv_lower": {
        "type": "array",
        "items": {"type": "integer"},
        "minItems": 3,
        "maxItems": 3,
        "description": "HSV lower bound [H, S, V]."
      },
      "hsv_upper": {
        "type": "array",
        "items": {"type": "integer"},
        "minItems": 3,
        "maxItems": 3,
        "description": "HSV upper bound [H, S, V]."
      },
      "bbox": {
        "type": "array",
        "items": {"type": "number"},
        "minItems": 4,
        "maxItems": 4,
        "description": "Optional bounding box [x0, y0, x1, y1] to restrict pixel extraction."
      },
      "exclude_bboxes": {
        "type": "array",
        "items": {
          "type": "array",
          "items": {"type": "number"},
          "minItems": 4,
          "maxItems": 4
        },
        "description": "List of bounding boxes [[x0, y0, x1, y1], ...] to exclude from extraction (e.g. legend boxes)."
      },
      "extract_mode": {
        "type": "string",
        "enum": ["continuous_line", "scatter_centroids"],
        "description": "Extraction mode.",
        "default": "continuous_line"
      }
    },
    "required": ["image_path"]
  }
}
```

* **入出力仕様**:
  * **引数**:
    * `image_path` (str): 対象画像パス。
    * `color_preset` (str, 任意): 代表色プリセット名（`blue`, `red`, `green`, `orange`, `black`, `cyan`, `magenta`, `yellow`, `purple`, `brown`, `pink`, `gray`）。
    * `target_rgb` (list[int], 任意): 抽出対象の RGB 値 `[R, G, B]` (0〜255)。
    * `target_hex` (str, 任意): 抽出対象の 16 進数カラーコード（例: `"#1F77B4"`）。
    * `color_tolerance` (float, 初期値: 35.0): 色の許容誤差。0.001〜1.0（正規化値）または 1.0〜255.0（RGBユークリッド距離）に対応。
    * `hsv_lower` (list[int], 任意): HSV 下限値 `[H, S, V]`。
    * `hsv_upper` (list[int], 任意): HSV 上限値 `[H, S, V]`。
    * `bbox` (list[float], 任意): 抽出領域を制限するバウンディングボックス `[x0, y0, x1, y1]`。
    * `exclude_bboxes` (list[list[float]], 任意): 除外する矩形領域リスト（凡例ボックス等）。
    * `extract_mode` (str, 初期値: "continuous_line"): `"continuous_line"` または `"scatter_centroids"`。
  * **戻り値 (dict)**:
    * `status` (str): `"success"`。
    * `point_count` (int): 抽出されたデータ点数。
    * `pixel_points` (list[list[int]]): `[[x1, y1], [x2, y2], ...]`（ピクセル座標配列）。

---

### Tool 10: `extract_vector_curve_points`

* **ツール関数名**: `extract_vector_curve_points`
* **内部使用モジュール**: `pymupdf` (`fitz`), `numpy`, `math`
* **役割・機能**:
PDF内部のベクター描画命令（3次ベジェ曲線 `'c'` や線分 `'l'`）を解析し、等間隔にサンプリングした座標列を抽出します。
単一の描画番号指定（`drawing_index`）に加え、複数の描画オブジェクトを一括指定（`drawing_indices`）して同時に処理できます。さらに `group_by_color=True` を指定することで、同一の線色（stroke color）を持つ複数描画セグメントを1本の統一された曲線として自動連結・重複排除・昇順ソートして抽出可能です。
微小な目盛り線やノイズ線分を自動除外する `min_length` フィルタ、特定ストローク色のみを抽出する `stroke_color` フィルタ、ページ内の全描画オブジェクトから一括抽出する `extract_all_matching` を備えています。また、描画番号未指定時は有効な曲線を含む描画オブジェクトを自動探索・選択します。
DPI スケーリングおよびクロップ原点オフセットの自動変換に対応し、画像解析ツール側のピクセル座標系と完全に一致する点列を出力できます。
* **OpenAI Function Calling 定義 (JSON Schema)**:

```json
{
  "name": "extract_vector_curve_points",
  "description": "Sample dense coordinate points from vector Bézier curves or lines inside a PDF drawing, with auto-chaining, min_length filtering, color filtering, drawing_indices batch extraction, group_by_color merging, and crop transformation.",
  "parameters": {
    "type": "object",
    "properties": {
      "pdf_path": {
        "type": "string",
        "description": "Path to the source PDF file."
      },
      "page_number": {
        "type": "integer",
        "description": "0-indexed PDF page number.",
        "default": 0
      },
      "bbox_filter": {
        "type": "array",
        "items": {"type": "number"},
        "minItems": 4,
        "maxItems": 4,
        "description": "Optional bounding box in points to filter."
      },
      "drawing_index": {
        "type": "integer",
        "description": "Optional specific drawing index."
      },
      "drawing_indices": {
        "type": "array",
        "items": {"type": "integer"},
        "description": "Optional list of drawing indices to extract in batch."
      },
      "group_by_color": {
        "type": "boolean",
        "description": "Group extracted curves by stroke color, merging into unified curves.",
        "default": false
      },
      "curve_types": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Element types to extract: 'c' or 'l'.",
        "default": ["c"]
      },
      "item_indices": {
        "type": "array",
        "items": {"type": "integer"},
        "description": "Optional specific item indices in drawing."
      },
      "curve_segments": {
        "type": "array",
        "items": {
          "type": "array",
          "items": {"type": "integer"}
        },
        "description": "Optional explicit item grouping for curves."
      },
      "min_length": {
        "type": "number",
        "description": "Minimum path length in points to filter out small tick marks or noisy segments.",
        "default": 0.0
      },
      "stroke_color": {
        "type": "array",
        "items": {"type": "number"},
        "minItems": 3,
        "maxItems": 3,
        "description": "Optional stroke color filter [r, g, b] with values normalized in [0.0, 1.0]."
      },
      "extract_all_matching": {
        "type": "boolean",
        "description": "If true, extracts and returns curves from all matching drawing objects on the page.",
        "default": false
      },
      "num_samples_per_segment": {
        "type": "integer",
        "description": "Number of sample points per Bézier curve.",
        "default": 50
      },
      "sort_x_ascending": {
        "type": "boolean",
        "description": "Sort points so X increases monotonically.",
        "default": true
      },
      "dpi": {
        "type": "number",
        "description": "DPI to scale coordinates to pixels."
      },
      "crop_bbox_pixels": {
        "type": "array",
        "items": {"type": "number"},
        "minItems": 4,
        "maxItems": 4,
        "description": "Optional [x0, y0, x1, y1] crop offset in px."
      }
    },
    "required": ["pdf_path"]
  }
}
```

* **入出力仕様**:
  * **引数**:
    * `pdf_path` (str): 対象 PDF パス。
    * `page_number` (int, 初期値: 0): 0始まりのページ番号。
    * `bbox_filter` (list[float], 任意): pt 単位の描画検索枠。
    * `drawing_index` (int, 任意): 特定の描画オブジェクト番号。
    * `drawing_indices` (list[int], 任意): 複数の特定描画オブジェクト番号リスト。
    * `group_by_color` (bool, 初期値: False): 色別に描画曲線を自動グループ化・統合するか。
    * `curve_types` (list[str], 初期値: `["c"]`): 対象要素種別（`'c'` はベジェ曲線、`'l'` は線分）。
    * `item_indices` (list[int], 任意): 描画内の特定アイテム番号リスト。
    * `curve_segments` (list[list[int]], 任意): 曲線ごとにアイテムをグループ化したリスト。
    * `min_length` (float, 初期値: 0.0): 微小目盛り線やノイズを除外する最小長さ（pt）。
    * `stroke_color` (list[float], 任意): `[r, g, b]` (0.0〜1.0) のストローク色フィルタ。
    * `extract_all_matching` (bool, 初期値: False): True の場合、全マッチ描画の曲線を返却。
    * `num_samples_per_segment` (int, 初期値: 50): 1区間のサンプリング点数。
    * `sort_x_ascending` (bool, 初期値: True): X座標を昇順に整列するか。
    * `dpi` (float, 任意): ピクセル変換用 DPI。
    * `crop_bbox_pixels` (list[float], 任意): クロップ画像の [x0, y0, x1, y1] (px)。
  * **戻り値 (dict)**:
    * `status` (str): `"success"`。
    * `total_curves` (int): 抽出・連結された曲線本数（単一描画モード時）。
    * `curves` (list[dict]): 各曲線データ（単一描画モード時）。
    * `total_groups` (int): 抽出された色グループ総数（`group_by_color=True` 時）。
    * `color_grouped_curves` (list[dict]): 各色グループ内の統合曲線データ（`group_by_color=True` 時）。
      * `stroke_color` (list[float]): 元ストローク色。
      * `color_rgb` (list[int]): RGB (0-255)。
      * `color_hex` (str): 16進数カラーコード。
      * `drawing_indices` (list[int]): 統合された描画インデックス。
      * `total_points` (int): 統合点数。
      * `points` (list[list[float]]): サンプリングされた [[x, y], ...] 座標列。
    * `total_drawings` (int): マッチした描画オブジェクト総数（`extract_all_matching=True` 時）。
    * `drawings` (list[dict]): 各描画オブジェクト内の曲線群（`extract_all_matching=True` 時）。

---

### Tool 11: `calibrate_and_convert_coordinates`

* **ツール関数名**: `calibrate_and_convert_coordinates`
* **内部使用モジュール**: `numpy`, `pandas`, `math`
* **役割・機能**:
既知の基準点（ピクセル座標と実数値の対応点）をもとに線形または対数スケール変換を行い、実世界ドメインの数値データテーブルを CSV 形式で出力します。
出力 CSV の列名指定（`column_names`）、単一曲線のラベル付与（`curve_label`）、および複数曲線（`curves`）の一括統合変換に対応します。各曲線辞書は `label` に加え、`name` または `curve_name` も曲線識別子として柔軟に受理します。
さらに、複数曲線を単一の共通 X 格子に線形補間（`np.interp`）して横持ちマトリクス形式（`output_format="wide"`: 列 `x, curve1, curve2, ...`）で出力する機能を備えており、共通格子点の直接指定（`resample_x_grid`）または等間隔点数指定（`num_grid_points`）に対応します。
* **OpenAI Function Calling 定義 (JSON Schema)**:

```json
{
  "name": "calibrate_and_convert_coordinates",
  "description": "Map extracted pixel coordinates to real-world domain values using linear or log scales, with column naming, multi-curve support, common X grid resampling, and wide/long CSV formats.",
  "parameters": {
    "type": "object",
    "properties": {
      "pixel_points": {
        "type": "array",
        "items": {
          "type": "array",
          "items": {"type": "number"},
          "minItems": 2
        },
        "description": "Pixel coordinate points [[x, y], ...]."
      },
      "curves": {
        "type": "array",
        "items": {"type": "object"},
        "description": "Optional list of curve dicts each having 'points' list and label/name/curve_name."
      },
      "column_names": {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 2,
        "maxItems": 2,
        "description": "Custom X/Y column names in CSV.",
        "default": ["x", "y"]
      },
      "curve_label": {
        "type": "string",
        "description": "Optional label for single curve."
      },
      "x_calibration": {
        "type": "object",
        "properties": {
          "pixel_refs": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
          "val_refs": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
          "scale_type": {"type": "string", "enum": ["linear", "log"], "default": "linear"}
        },
        "required": ["pixel_refs", "val_refs"]
      },
      "y_calibration": {
        "type": "object",
        "properties": {
          "pixel_refs": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
          "val_refs": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
          "scale_type": {"type": "string", "enum": ["linear", "log"], "default": "linear"}
        },
        "required": ["pixel_refs", "val_refs"]
      },
      "output_csv_path": {
        "type": "string",
        "description": "Destination file path for the CSV output."
      },
      "output_format": {
        "type": "string",
        "enum": ["long", "wide"],
        "description": "Output CSV layout: 'long' (tidy) or 'wide' (matrix).",
        "default": "long"
      },
      "resample_x_grid": {
        "type": "array",
        "items": {"type": "number"},
        "description": "Optional explicit common X grid values for resampling."
      },
      "num_grid_points": {
        "type": "integer",
        "description": "Number of interpolation points for shared X grid.",
        "default": 100
      }
    },
    "required": ["x_calibration", "y_calibration", "output_csv_path"]
  }
}
```

* **入出力仕様**:
  * **引数**:
    * `x_calibration`, `y_calibration`: 座標マッピング定義（pixel_refs, val_refs, scale_type）。
    * `output_csv_path` (str): 出力 CSV パス。
    * `pixel_points` (list[list[float]], 任意): 単一曲線のピクセル座標列。
    * `curves` (list[dict], 任意): 複数曲線の辞書リスト（`label`, `name`, `curve_name` のいずれかでラベル指定可能）。
    * `column_names` (list[str], 初期値: `["x", "y"]`): CSV の X/Y 列名。
    * `curve_label` (str, 任意): 単一曲線時のラベル名。
    * `output_format` (str, 初期値: `"long"`): 出力レイアウト。`"long"`（縦持ち形式: 列 `x, y, curve`）または `"wide"`（横持ちマトリクス形式: 列 `x, curve1, curve2, ...`）。
    * `resample_x_grid` (list[float], 任意): リサンプリング用の明示的な共通 X 格子配列。
    * `num_grid_points` (int, 任意): 共通 X 格子の等間隔補間点数（省略時は 100）。
  * **戻り値 (dict)**:
    * `status` (str): `"success"`。
    * `csv_path` (str): 保存された CSV パス。
    * `row_count` (int): 出力行数。
    * `data_preview` (list[dict]): 先頭5行のプレビュー。

---

### Tool 12: `render_verification_overlay`

* **ツール関数名**: `render_verification_overlay`
* **内部使用モジュール**: `cv2` (OpenCV), `numpy`, `pandas`
* **役割・機能**:
デジタイズされた CSV データを元画像の座標系へ逆変換し、元画像の上に半透明オーバーレイ画像を生成して保存します。
CSV 内に複数曲線が含まれる場合（`curve` カラム等）は、曲線間で余計なジャンプ線が引かれないよう自動グループ化し、それぞれ異なる鮮やかな蛍光色（シアン、マゼンタ、イエロー、グリーン等）で色分け描画します。
エッジ重なり度合いに基づく客観的な適合度スコア（`alignment_metric`: 0.0〜1.0）を算出します。
* **OpenAI Function Calling 定義 (JSON Schema)**:

```json
{
  "name": "render_verification_overlay",
  "description": "Re-plot digitized numerical data onto the original cropped image as a multi-color semi-transparent overlay to verify alignment.",
  "parameters": {
    "type": "object",
    "properties": {
      "original_image_path": {
        "type": "string",
        "description": "Path to the cropped original plot image."
      },
      "csv_path": {
        "type": "string",
        "description": "Path to the digitized CSV file."
      },
      "x_calibration": {
        "type": "object",
        "description": "X axis calibration parameters."
      },
      "y_calibration": {
        "type": "object",
        "description": "Y axis calibration parameters."
      },
      "output_overlay_path": {
        "type": "string",
        "description": "Destination file path for the overlay image."
      },
      "curve_column": {
        "type": "string",
        "description": "CSV column name for grouping curves.",
        "default": "curve"
      }
    },
    "required": [
      "original_image_path",
      "csv_path",
      "x_calibration",
      "y_calibration",
      "output_overlay_path"
    ]
  }
}
```

* **入出力仕様**:
  * **引数**:
    * `original_image_path` (str): 元画像パス。
    * `csv_path` (str): デジタイズ結果 CSV パス。
    * `x_calibration`, `y_calibration`: 座標マッピング定義。
    * `output_overlay_path` (str): オーバーレイ画像出力先。
    * `curve_column` (str, 初期値: "curve"): 曲線グループ分け列名。
  * **戻り値 (dict)**:
    * `status` (str): `"success"`。
    * `verification_image_path` (str): 生成画像パス。
    * `alignment_metric` (float): 一致率指標スコア (0.0〜1.0)。
    * `curves_rendered` (int): 描画された曲線本数。

---

## 4. コーディングエージェント（CLI）向け実装ガイドライン

本仕様書を `codex` や `agy` などのCLIに投入して実装させる際は、以下のファイル構成で生成させてください。

### ディレクトリ構成

```text
digitize_agent/
├─ pyproject.toml              # プロジェクト設定および依存関係定義
├─ README.md                   # プロジェクト説明書
├─ SPECIFICATION.md            # 詳細設計仕様書
├─ src/
│    └─ digitize_agent/
│          ├─ __init__.py
│          ├─ server.py        # MCP (Model Context Protocol) サーバー実装 (全11ツール)
│          ├─ schema.py        # OpenAI Function Calling 互換 JSON Schema 定義 (全11ツール)
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

### 実装上の必須ルール

1. **型ヒントとPydanticモデル**: すべての引数および戻り値辞書には厳密なType Annotationsを付与すること。
2. **エラー耐性**: ファイルが存在しない場合やパラメータ不正時には例外でプロセスを落とさず、`{"status": "error", "message": str(e)}` を返してVLMが再試行（リトライ）できるようにすること。
3. **完全オフライン保証**: コード内で `urllib`, `requests`, `transformers`, `huggingface_hub` 等の外部通信モジュールをインポート・実行しないこと。
4. **PEP 8 & 最大行長 79 文字遵守**: すべての Python コードで PEP 8 スタイルガイドを厳守し、最大行長を 79 文字に制限すること。

---

## 5. MCP (Model Context Protocol) サーバー仕様

本ツール群はローカルの MCP サーバーとして起動し、Claude Desktop や各種 AI コーディングエージェント（Antigravity 等）から stdio 経由で呼び出すことが可能です。全 11 種類のツール関数が標準公開されます。

### サーバー起動方法

```bash
# パッケージの CLI エントリーポイント経由
digitize-agent

# または Python モジュール経由
python -m digitize_agent.server
```

### クライアント設定例 (`claude_desktop_config.json` 等)

```json
{
  "mcpServers": {
    "digitize-agent": {
      "command": "python",
      "args": ["-m", "digitize_agent.server"]
    }
  }
}
```

※仮想環境を利用している場合は、`command` に仮想環境内の Python インタプリタの絶対パス（Windows 例: `C:\\path\\to\\digitize_agent\\.venv\\Scripts\\python.exe`、macOS/Linux 例: `/path/to/digitize_agent/.venv/bin/python`）を指定してください。
