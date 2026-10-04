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
   ├─ (PDF の場合: 構造解析) ──> inspect_pdf_primitives (埋め込みテキスト・ベクター直接抽出)
   │
   ├─ パターン A: PDF ベクター描画グラフ (ベジェ曲線)
   │     │
   │     ├──> crop_and_transform_region (PDF直接レンダリング・DPI指定クロップ)
   │     │       └──> detect_axes_and_ticks (軸線および目盛りピクセル検出)
   │     │
   │     ├──> extract_vector_curve_points (PDFベクターパスから解析的サンプリング)
   │     │
   │     ├──> calibrate_and_convert_coordinates (物理量変換・複数曲線一括 CSV 出力)
   │     │
   │     └──> render_verification_overlay (複数曲線色分け透過オーバーレイ検証)
   │
   └─ パターン B: ラスター画像グラフ (スキャン文書・写真)
         │
         ├──> crop_and_transform_region (領域クロップ・傾き自動補正・コントラスト強調)
         │       │
         │       ├──> detect_axes_and_ticks (座標軸・目盛りピクセル検出)
         │       │
         │       ├──> ocr_region_text (軸ラベル・目盛り数値の文字認識)
         │       │
         │       ├──> detect_plot_colors (画像内の主要プロット色を自動検出)
         │       │
         │       └──> extract_plot_pixels_by_color (色プリセット/HSV閾値による点抽出)
         │
         ├──> calibrate_and_convert_coordinates (実数値変換・列名指定 CSV 出力)
         │
         └──> render_verification_overlay (再描画と元画像の重ね合わせ検証)
```

### 全 9 ツール一覧表

| ツール関数名 | 役割・機能概要 |
| :--- | :--- |
| `inspect_pdf_primitives` | PDF からラスター変換を経由せず、直接埋め込まれたテキスト要素およびベクター罫線を抽出 |
| `crop_and_transform_region` | 画像または PDF から直接指定 DPI で領域を切り出し、傾き補正 (Deskew) や強調 (CLAHE) を適用 |
| `detect_axes_and_ticks` | グラフ画像内の主軸（水平 X 軸・垂直 Y 軸）および目盛り線（Tick marks）のピクセル座標を幾何学的に検出 |
| `ocr_region_text` | 切り出し画像スニペットに対して Tesseract OCR を実行し、目盛り数値や軸ラベル等のテキストと信頼度を返却 |
| `detect_plot_colors` | 画像内の主要プロット色（色名、代表 HSV 値、画素占有率）を自動検出し、色抽出のための推奨設定を提示 |
| `extract_plot_pixels_by_color` | 色プリセット（'blue', 'red' 等）または HSV 閾値に基づき、指定色プロットのピクセル座標群を抽出 |
| `extract_vector_curve_points` | PDF 内部のベクター描画命令（3次ベジェ曲線・折れ線）から等間隔座標列をサンプリングし、クロップ座標系へ自動変換 |
| `calibrate_and_convert_coordinates` | 軸基準点に基づき線形/対数スケールで実数値へ変換し、列名指定や複数曲線を統合した CSV ファイルを出力 |
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

### Tool 2: `crop_and_transform_region`

* **ツール関数名**: `crop_and_transform_region`
* **内部使用モジュール**: `cv2` (OpenCV), `numpy`, `pymupdf` (`fitz`)
* **役割・機能**:
画像ファイル、または PDF ファイルから直接指定 DPI でレンダリングして指定領域を切り出します。傾き（Deskew）の自動検出と補正、CLAHE コントラスト強調を行い、解析用サブ画像を生成します。
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
    "required": ["bbox", "output_path"]
  }
}
```

* **入出力仕様**:
  * **引数**:
    * `bbox` (list[float]): `[x_min, y_min, x_max, y_max]`。
    * `output_path` (str): 保存先ファイルパス。
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
* **実装要件・アルゴリズム**:
  * `pdf_path` 指定時は `page.get_pixmap(dpi=int(round(dpi)))` で画像化。
  * `bbox_mode == "point"` の場合は `dpi / 72.0` を乗じてピクセル座標へ自動換算。
  * 傾き検出: Canny エッジ検出と確率的ハフ変換（`cv2.HoughLinesP`）による傾斜中央値の算出およびアフィン回転補正。
  * コントラスト強調: LAB色空間のLチャンネルに対するCLAHE適用。

---

### Tool 3: `detect_axes_and_ticks`

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

### Tool 4: `ocr_region_text`

* **ツール関数名**: `ocr_region_text`
* **内部使用モジュール**: `pytesseract`, `PIL` (Pillow), `cv2`
* **役割・機能**:
切り出した領域（目盛りの数値部分、軸ラベル、凡例など）に対して、ローカルのTesseract OCRを実行し、認識テキストと確信度スコアを返します。
* **OpenAI Function Calling 定義 (JSON Schema)**:

```json
{
  "name": "ocr_region_text",
  "description": "Execute OCR on an image snippet to recognize axis labels, tick numbers, or table cell contents.",
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
  * **戻り値 (dict)**:
    * `text` (str): 生テキスト。
    * `clean_text` (str): 空白・改行整形テキスト。
    * `confidence` (float): OCR全体の平均確信度スコア（0.0〜100.0）。

---

### Tool 5: `detect_plot_colors`

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

### Tool 6: `extract_plot_pixels_by_color`

* **ツール関数名**: `extract_plot_pixels_by_color`
* **内部使用モジュール**: `cv2` (OpenCV), `numpy`
* **役割・機能**:
代表色プリセット（'blue', 'red', 'green' 等）または手動の HSV 色閾値に基づき、指定色プロットのピクセル座標群を抽出します。ノイズ除去（開閉演算）を行い、連続線または散布図マーカーの重心座標を配列化して返します。
* **OpenAI Function Calling 定義 (JSON Schema)**:

```json
{
  "name": "extract_plot_pixels_by_color",
  "description": "Filter and extract pixel coordinates for data lines or markers of a specified color using presets or HSV thresholds.",
  "parameters": {
    "type": "object",
    "properties": {
      "image_path": {
        "type": "string",
        "description": "Path to the plot image file."
      },
      "color_preset": {
        "type": "string",
        "enum": ["blue", "red", "green", "orange", "black"],
        "description": "Preset color name for quick extraction."
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
    * `color_preset` (str, 任意): 代表色プリセット名。
    * `hsv_lower` (list[int], 任意): HSV下限値。
    * `hsv_upper` (list[int], 任意): HSV上限値。
    * `extract_mode` (str, 初期値: "continuous_line"): `"continuous_line"` または `"scatter_centroids"`。
  * **戻り値 (dict)**:
    * `status` (str): `"success"`。
    * `point_count` (int): 抽出されたデータ点数。
    * `pixel_points` (list[list[int]]): `[[x1, y1], [x2, y2], ...]`（ピクセル座標配列）。

---

### Tool 7: `extract_vector_curve_points`

* **ツール関数名**: `extract_vector_curve_points`
* **内部使用モジュール**: `pymupdf` (`fitz`), `numpy`, `math`
* **役割・機能**:
PDF内部のベクター描画命令（3次ベジェ曲線 `'c'` や線分 `'l'`）を解析し、等間隔にサンプリングした座標列を抽出します。
原点を共有する複数曲線が誤って折り返し結合されないよう、ベクトル進行方向の内積を判定する幾何学的連結処理（`_chain_segments`）を備えています。
DPI スケーリングおよびクロップ原点オフセットの自動変換に対応し、画像解析ツール側のピクセル座標系と完全に一致する点列を出力できます。
* **OpenAI Function Calling 定義 (JSON Schema)**:

```json
{
  "name": "extract_vector_curve_points",
  "description": "Sample dense coordinate points from vector Bézier curves or lines inside a PDF drawing, with auto-chaining and crop transformation.",
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
    * `curve_types` (list[str], 初期値: `["c"]`): 対象要素種別（`'c'` はベジェ曲線、`'l'` は線分）。
    * `item_indices` (list[int], 任意): 描画内の特定アイテム番号リスト。
    * `curve_segments` (list[list[int]], 任意): 曲線ごとにアイテムをグループ化したリスト。
    * `num_samples_per_segment` (int, 初期値: 50): 1区間のサンプリング点数。
    * `sort_x_ascending` (bool, 初期値: True): X座標を昇順に整列するか。
    * `dpi` (float, 任意): ピクセル変換用 DPI。
    * `crop_bbox_pixels` (list[float], 任意): クロップ画像の [x0, y0, x1, y1] (px)。
  * **戻り値 (dict)**:
    * `status` (str): `"success"`。
    * `total_curves` (int): 抽出・連結された曲線本数。
    * `curves` (list[dict]): 各曲線データ。
      * `curve_index` (int): 0から始まる曲線番号。
      * `item_indices` (list[int]): 構成アイテム番号リスト。
      * `num_points` (int): サンプリング点数。
      * `bounds` (list[float]): [min_x, min_y, max_x, max_y]。
      * `points` (list[list[float]]): サンプリングされた [[x, y], ...] 座標列。

---

### Tool 8: `calibrate_and_convert_coordinates`

* **ツール関数名**: `calibrate_and_convert_coordinates`
* **内部使用モジュール**: `numpy`, `pandas`, `math`
* **役割・機能**:
既知の基準点（ピクセル座標と実数値の対応点）をもとに線形または対数スケール変換を行い、実世界ドメインの数値データテーブルを CSV 形式で出力します。
出力 CSV の列名指定（`column_names`）、単一曲線のラベル付与（`curve_label`）、および複数曲線（`curves`）の一括統合変換に対応します。
* **OpenAI Function Calling 定義 (JSON Schema)**:

```json
{
  "name": "calibrate_and_convert_coordinates",
  "description": "Map extracted pixel coordinates to real-world domain values using linear or log scales, with column naming and multi-curve support.",
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
        "description": "Optional list of curve dicts each having 'points' list."
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
    * `curves` (list[dict], 任意): 複数曲線の辞書リスト。
    * `column_names` (list[str], 初期値: `["x", "y"]`): CSV の X/Y 列名。
    * `curve_label` (str, 任意): 単一曲線時のラベル名。
  * **戻り値 (dict)**:
    * `status` (str): `"success"`。
    * `csv_path` (str): 保存された CSV パス。
    * `row_count` (int): 出力行数。
    * `data_preview` (list[dict]): 先頭5行のプレビュー。

---

### Tool 9: `render_verification_overlay`

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
│          ├─ server.py        # MCP (Model Context Protocol) サーバー実装 (全9ツール)
│          ├─ schema.py        # OpenAI Function Calling 互換 JSON Schema 定義 (全9ツール)
│          ├─ registry.py      # 関数ディスパッチャー (名前と実関数の安全な実行管理)
│          └─ tools/           # 各デジタイズツールの実装
│                ├─ __init__.py
│                ├─ pdf_tools.py          # Tool 1: inspect_pdf_primitives
│                ├─ image_transforms.py   # Tool 2: crop_and_transform_region
│                ├─ geometry_detect.py    # Tool 3: detect_axes_and_ticks
│                ├─ ocr_tools.py          # Tool 4: ocr_region_text
│                ├─ color_extractor.py    # Tool 5: detect_plot_colors, Tool 6: extract_plot_pixels_by_color
│                ├─ vector_curves.py      # Tool 7: extract_vector_curve_points
│                ├─ calibration.py        # Tool 8: calibrate_and_convert_coordinates
│                └─ visual_verifier.py    # Tool 9: render_verification_overlay
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

本ツール群はローカルの MCP サーバーとして起動し、Claude Desktop や各種 AI コーディングエージェント（Antigravity 等）から stdio 経由で呼び出すことが可能です。全 9 種類のツール関数が標準公開されます。

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
