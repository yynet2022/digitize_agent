# Agentic Plot & Document Digitization ツール仕様書

本仕様書は、学術論文や技術文書内のグラフ（プロット）および表を、ローカル環境で自律的（Agentic）に高精度デジタイズするためのVLM Tool（OpenAI Function Calling互換）および MCP (Model Context Protocol) サーバー対応ツール群の設計仕様です。

外部API送信およびモデル重みの自動ダウンロード（Hugging Face Hub等へのアクセス）を伴うモジュールを一切排除し、完全ローカルかつ決定論的に動作するライブラリ（`OpenCV`, `NumPy`, `SciPy`, `pandas`, `PyMuPDF`, `pytesseract`, `matplotlib`, `mcp`）のみで構成しています。

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

```
[入力ドキュメント (PDF / 画像)]
   │
   ├─ (PDFの場合) ──> inspect_pdf_primitives (埋め込みテキスト・ベクター直接抽出)
   │
   └─ (画像・共通)
         │
         ├──> crop_and_transform_region (領域クロップ・傾き補正)
         │       │
         │       ├──> ocr_region_text (軸ラベル・目盛り・表セルの文字認識)
         │       │
         │       ├──> detect_axes_and_ticks (直交座標軸・目盛りピクセル検出)
         │       │
         │       └──> extract_plot_pixels_by_color (特定色プロット点の抽出)
         │
         ├──> calibrate_and_convert_coordinates (ピクセル座標 ─> 実数値変換)
         │
         └──> render_verification_overlay (再描画と元画像の重ね合わせ検証)

```

---

## 3. 各ツール関数の詳細仕様

### Tool 1: `inspect_pdf_primitives`

* **ツール関数名**: `inspect_pdf_primitives`
* **内部使用モジュール**: `fitz` (`PyMuPDF`), `pdfplumber`
* **役割・機能**:
PDFファイルからラスター変換を経由せず、直接埋め込まれているテキスト要素（フォント、バウンディングボックス座標）およびベクターパス（罫線、四角形）を抽出します。スキャン画像ではない真正PDFにおいて、解像度低下やOCR誤認識を回避して100%の再現性を確保します。
* **OpenAI Function Calling 定義 (JSON Schema)**:

```json
{
  "name": "inspect_pdf_primitives",
  "description": "Extract raw vector lines, rects, and text bounding boxes directly from a PDF page without raster degradation. Use this prior to image processing if the source is a digital PDF.",
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
* `bbox_filter` (list[float], 任意): `[x0, y0, x1, y1]` で指定する抽出範囲。


* **戻り値 (dict)**:
* `is_scanned` (bool): テキストが一切埋め込まれていない場合はTrue。
* `text_elements` (list[dict]): `{"text": str, "bbox": [x0, y0, x1, y1], "font_size": float}` の配列。
* `vector_lines` (list[dict]): `{"coords": [x0, y0, x1, y1], "width": float}` の配列。




* **実装要件・アルゴリズム**:
* `fitz.open(pdf_path)` でドキュメントを読み込み。
* `page.get_text("blocks")` または `page.get_text("words")` により座標付きテキストを取得。
* `pdfplumber` の `page.lines` / `page.rects` を用いて表の罫線候補を抽出。



---

### Tool 2: `crop_and_transform_region`

* **ツール関数名**: `crop_and_transform_region`
* **内部使用モジュール**: `cv2` (OpenCV), `numpy`
* **役割・機能**:
指定されたバウンディングボックスの領域を切り出し、傾き（Deskew）の検出と補正、二値化、コントラスト調整を行ってローカルの一時ファイルに保存します。微小な文字や複雑なプロットを拡大・強調してVLMやOCRに渡す前処理として使用します。
* **OpenAI Function Calling 定義 (JSON Schema)**:

```json
{
  "name": "crop_and_transform_region",
  "description": "Crop a specific region of an image, optionally apply contrast enhancement and deskewing, and save as a high-resolution sub-image.",
  "parameters": {
    "type": "object",
    "properties": {
      "image_path": {
        "type": "string",
        "description": "Path to the source image file."
      },
      "bbox": {
        "type": "array",
        "items": {"type": "integer"},
        "minItems": 4,
        "maxItems": 4,
        "description": "Bounding box coordinates [x_min, y_min, x_max, y_max] in pixels."
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
    "required": ["image_path", "bbox", "output_path"]
  }
}

```

* **入出力仕様**:
* **引数**:
* `image_path` (str): 元画像パス。
* `bbox` (list[int]): `[x_min, y_min, x_max, y_max]`（ピクセル単位）。
* `deskew` (bool, 初期値: False): 傾き自動補正を行うか。
* `enhance_contrast` (bool, 初期値: False): コントラスト強調（CLAHE）を行うか。
* `output_path` (str): 切り出し画像の保存先パス。


* **戻り値 (dict)**:
* `status` (str): `"success"` または `"error"`。
* `output_path` (str): 保存された画像パス。
* `dimensions` (dict): `{"width": int, "height": int}`。
* `skew_angle_detected` (float): 検出された傾き角度（度数法）。




* **実装要件・アルゴリズム**:
* `cv2.imread` で画像を読み込み、スライス処理でクロップ。
* 傾き検出: グレースケール化後、エッジ検出（`cv2.Canny`）からハフ変換（`cv2.HoughLinesP`）で直線群の傾斜中央値を算出。アフィン変換行列（`cv2.getRotationMatrix2D`, `cv2.warpAffine`）で回転補正。
* コントラスト強調: `cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))` を適用。



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
  "description": "Detect X and Y axis line pixel positions and potential tick mark coordinates within a plot image.",
  "parameters": {
    "type": "object",
    "properties": {
      "image_path": {
        "type": "string",
        "description": "Path to the cropped plot area image."
      },
      "min_line_length_ratio": {
        "type": "number",
        "description": "Minimum length ratio relative to image dimensions to qualify as an axis line (default: 0.3).",
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
* `min_line_length_ratio` (float, 初期値: 0.3): 軸線と見なす最小長さの画像幅/高さ比。


* **戻り値 (dict)**:
* `x_axis`: `{"y_pixel": int, "x_range": [int, int]}` （X軸のYピクセル位置とX範囲）
* `y_axis`: `{"x_pixel": int, "y_range": [int, int]}` （Y軸のXピクセル位置とY範囲）
* `x_tick_candidates`: list[int] （検出された目盛りのXピクセル座標群）
* `y_tick_candidates`: list[int] （検出された目盛りのYピクセル座標群）




* **実装要件・アルゴリズム**:
* グレースケール化後、Sobelフィルタまたはモルフォロジー演算（水平カーネル・垂直カーネルによる膨張・収縮）を用いて水平線・垂直線を分離。
* 水平/垂直プロジェクション（ピクセル強度の行方向・列方向の総和プロファイル）を計算。
* `scipy.signal.find_peaks` を用いて、軸線近傍にある周期的な目盛り線の突出ピークを検出。



---

### Tool 4: `ocr_region_text`

* **ツール関数名**: `ocr_region_text`
* **内部使用モジュール**: `pytesseract`, `PIL` (Pillow), `cv2`
* **役割・機能**:
切り出した領域（目盛りの数値部分、軸ラベル、凡例、表のセルなど）に対して、ローカルのTesseract OCRを実行し、認識テキストと確信度スコアを返します。
* **OpenAI Function Calling 定義 (JSON Schema)**:

```json
{
  "name": "ocr_region_text",
  "description": "Perform precise OCR on a local image snippet to read numbers, tick labels, or table cell texts.",
  "parameters": {
    "type": "object",
    "properties": {
      "image_path": {
        "type": "string",
        "description": "Path to the image snippet to recognize."
      },
      "psm": {
        "type": "integer",
        "description": "Tesseract Page Segmentation Mode (PSM). Use 6 (single block), 7 (single text line), or 8 (single word/number).",
        "default": 6
      },
      "whitelist": {
        "type": "string",
        "description": "Optional character whitelist, e.g., '0123456789.-+eE' for numeric scale reads."
      }
    },
    "required": ["image_path"]
  }
}

```

* **入出力仕様**:
* **引数**:
* `image_path` (str): 入力画像パス。
* `psm` (int, 初期値: 6): Tesseract PSMモード。目盛り数値の場合は 7（1行）または 8（単一単語）を推奨。
* `whitelist` (str, 任意): 許可する文字一覧（数値のみ読み取らせる場合に誤認識を防ぐ）。


* **戻り値 (dict)**:
* `text` (str): 抽出された生テキスト。
* `clean_text` (str): 改行や空白を除去したテキスト。
* `confidence` (float): OCR全体の平均確信度スコア（0.0〜100.0）。




* **実装要件・アルゴリズム**:
* `pytesseract.image_to_data(..., output_type=Output.DICT)` を使用してテキストと各単語の信頼度（`conf`）をパース。
* `whitelist` 指定がある場合、`config=f'--psm {psm} -c tessedit_char_whitelist={whitelist}'` を設定。



---

### Tool 5: `extract_plot_pixels_by_color`

* **ツール関数名**: `extract_plot_pixels_by_color`
* **内部使用モジュール**: `cv2` (OpenCV), `numpy`
* **役割・機能**:
プロット線や散布図マーカーの色（HSV色空間の範囲）を指定し、合致するピクセル座標群を抽出します。ノイズ除去（開閉演算）を行い、連続線または個別データ点の中心座標（重心）を配列化して返します。
* **OpenAI Function Calling 定義 (JSON Schema)**:

```json
{
  "name": "extract_plot_pixels_by_color",
  "description": "Extract pixel coordinates of plot curves or scatter points based on HSV color thresholding.",
  "parameters": {
    "type": "object",
    "properties": {
      "image_path": {
        "type": "string",
        "description": "Path to the plot image."
      },
      "hsv_lower": {
        "type": "array",
        "items": {"type": "integer"},
        "minItems": 3,
        "maxItems": 3,
        "description": "Lower bound for HSV threshold [H (0-179), S (0-255), V (0-255)]."
      },
      "hsv_upper": {
        "type": "array",
        "items": {"type": "integer"},
        "minItems": 3,
        "maxItems": 3,
        "description": "Upper bound for HSV threshold [H (0-179), S (0-255), V (0-255)]."
      },
      "extract_mode": {
        "type": "string",
        "enum": ["continuous_line", "scatter_centroids"],
        "description": "Extract continuous line (sorted by X pixel) or discrete marker centroids.",
        "default": "continuous_line"
      }
    },
    "required": ["image_path", "hsv_lower", "hsv_upper"]
  }
}

```

* **入出力仕様**:
* **引数**:
* `image_path` (str): 対象画像パス。
* `hsv_lower` (list[int]): HSV下限値 `[H, S, V]`。
* `hsv_upper` (list[int]): HSV上限値 `[H, S, V]`。
* `extract_mode` (str): `"continuous_line"`（X座標ごとにY中央値を抽出）または `"scatter_centroids"`（輪郭ごとの重心を抽出）。


* **戻り値 (dict)**:
* `point_count` (int): 抽出されたデータ点数。
* `pixel_points` (list[list[int]]): `[[x1, y1], [x2, y2], ...]`（ピクセル座標配列）。




* **実装要件・アルゴリズム**:
* `cv2.cvtColor(img, cv2.COLOR_BGR2HSV)` で変換。
* `cv2.inRange` でマスクを作成し、`cv2.morphologyEx`（カーネルサイズ 2x2〜3x3）で微細ノイズを除去。
* `continuous_line` モード: マスク上で白となったピクセルについて、X座標ごとにグループ化し、Y座標の中央値（Median）を取得してソート。
* `scatter_centroids` モード: `cv2.findContours` で輪郭を検出し、モーメント（`cv2.moments`）から各輪郭の重心 `(cx, cy)` を計算。



---

### Tool 6: `calibrate_and_convert_coordinates`

* **ツール関数名**: `calibrate_and_convert_coordinates`
* **内部使用モジュール**: `numpy`, `pandas`
* **役割・機能**:
既知の基準点（ピクセル座標と実数値の対応点）をもとにキャリブレーション（線形補間または対数変換）を行い、ピクセル座標配列を実際の物理値・統計値データテーブルに変換してCSV形式で保存します。
* **OpenAI Function Calling 定義 (JSON Schema)**:

```json
{
  "name": "calibrate_and_convert_coordinates",
  "description": "Map pixel coordinates to real-world domain values using calibrated reference points on X and Y axes, and export as CSV.",
  "parameters": {
    "type": "object",
    "properties": {
      "pixel_points": {
        "type": "array",
        "items": {
          "type": "array",
          "items": {"type": "number"},
          "minItems": 2,
          "maxItems": 2
        },
        "description": "List of pixel coordinates [[x, y], ...]."
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
        "description": "Destination file path for the digitized CSV."
      }
    },
    "required": ["pixel_points", "x_calibration", "y_calibration", "output_csv_path"]
  }
}

```

* **入出力仕様**:
* **引数**:
* `pixel_points` (list[list[float]]): 変換対象のピクセル座標リスト。
* `x_calibration`: X軸の既知点（ピクセル2点、実数値2点、スケール種別: linear/log）。
* `y_calibration`: Y軸の既知点（ピクセル2点、実数値2点、スケール種別: linear/log）。
* `output_csv_path` (str): 出力CSVファイルのパス。


* **戻り値 (dict)**:
* `status` (str): `"success"`。
* `csv_path` (str): 保存されたCSVパス。
* `row_count` (int): 出力データ件数。
* `data_preview` (list[dict]): 先頭5件のプレビュー。




* **実装要件・アルゴリズム**:
* 線形スケール変換式:

$$V_x = V_{x1} + \frac{P_x - P_{x1}}{P_{x2} - P_{x1}} \times (V_{x2} - V_{x1})$$


* ※画像座標系のY軸は下向き正であるため、Y軸の計算における方向反転を正確に処理すること。
* 対数スケールの場合: 基準値の対数 $\log_{10}(V)$ で線形補間を行った後、指数変換 $10^{\text{interpolated}}$ を適用。



---

### Tool 7: `render_verification_overlay`

* **ツール関数名**: `render_verification_overlay`
* **内部使用モジュール**: `matplotlib`, `cv2`, `numpy`, `pandas`
* **役割・機能**:
抽出・数値変換されたCSVデータからグラフを再プロットし、元画像の上に半透明オーバーレイまたは差分比較画像を生成して保存します。VLMはこの画像を確認し、ズレや欠落がある場合に自己修正（Visual Feedback）を行います。
* **OpenAI Function Calling 定義 (JSON Schema)**:

```json
{
  "name": "render_verification_overlay",
  "description": "Re-plot digitized numerical data onto the original cropped image as an overlay to visually verify alignment and calibration accuracy.",
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
        "description": "Same calibration parameter used in calibrate_and_convert_coordinates to map back to pixels."
      },
      "y_calibration": {
        "type": "object",
        "description": "Same calibration parameter used in calibrate_and_convert_coordinates to map back to pixels."
      },
      "output_overlay_path": {
        "type": "string",
        "description": "Destination file path for the verification image."
      }
    },
    "required": ["original_image_path", "csv_path", "x_calibration", "y_calibration", "output_overlay_path"]
  }
}

```

* **入出力仕様**:
* **引数**:
* `original_image_path` (str): 元画像パス。
* `csv_path` (str): デジタイズ結果CSV。
* `x_calibration`, `y_calibration`: 座標マッピング定義。
* `output_overlay_path` (str): オーバーレイ画像出力先。


* **戻り値 (dict)**:
* `verification_image_path` (str): 生成された画像パス。
* `alignment_metric`: 推定適合度指標（オーバーレイ線と元画像エッジの平均重なりスコア等、0.0〜1.0）。




* **実装要件・アルゴリズム**:
* 元画像をロード。
* CSVデータを逆変換してピクセル座標を再計算。
* 元画像の上に、蛍光色（シアンまたはマゼンタなど元画像に存在しない色、太さ2px）でプロット点または線を `cv2.polylines` / `cv2.circle` を用いて描画（`cv2.addWeighted` で透過合成）。



---

## 4. コーディングエージェント（CLI）向け実装ガイドライン

本仕様書を `codex` や `agy` などのCLIに投入して実装させる際は、以下のファイル構成で生成させてください。

### ディレクトリ構成

```text
digitize_agent/
├─ pyproject.toml
├─ src/
│    └─ digitize_agent/
│          ├─ tools/
│          │   ├─ __init__.py
│          │   ├─ pdf_tools.py          # Tool 1: inspect_pdf_primitives
│          │   ├─ image_transforms.py   # Tool 2: crop_and_transform_region
│          │   ├─ geometry_detect.py    # Tool 3: detect_axes_and_ticks
│          │   ├─ ocr_tools.py          # Tool 4: ocr_region_text
│          │   ├─ color_extractor.py    # Tool 5: extract_plot_pixels_by_color
│          │   ├─ calibration.py        # Tool 6: calibrate_and_convert_coordinates
│          │   └─ visual_verifier.py    # Tool 7: render_verification_overlay
│          ├─ schema.py                 # OpenAI Function Calling スキーマ定義辞書
│          ├─ registry.py               # 関数ディスパッチャー (名前と実関数のマッピング)
│          └─ server.py                 # MCP (Model Context Protocol) サーバー
└─ tests/

```

### 実装上の必須ルール

1. **型ヒントとPydanticモデル**: すべての引数および戻り値辞書には厳密なType Annotationsを付与すること。
2. **エラー耐性**: ファイルが存在しない場合やパラメータ不正時には例外でプロセスを落とさず、`{"status": "error", "message": str(e)}` を返してVLMが再試行（リトライ）できるようにすること。
3. **完全オフライン保証**: コード内で `urllib`, `requests`, `transformers`, `huggingface_hub` 等の外部通信モジュールをインポート・実行しないこと。

---

## 5. MCP (Model Context Protocol) サーバー仕様

本ツール群はローカルの MCP サーバーとして起動し、Claude Desktop や各種 AI コーディングエージェント（Antigravity 等）から stdio 経由で呼び出すことが可能です。

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

※仮想環境を利用している場合は、`command` に仮想環境内の Python インタプリタの絶対パス（例: `C:\\Users\\yy9zz\\PyWorks\\digitize_agent\\.venv\\Scripts\\python.exe`）を指定してください。

