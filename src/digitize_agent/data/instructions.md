# Digitize Agent - MCP ツール利用ガイドと推奨ワークフロー (Best Practices)

`digitize-agent` は、論文や技術文書内のグラフ（プロット）および図表を高精度かつ自律的にデジタイズするための MCP (Model Context Protocol) ツールスイートです。
外部 API や機械学習重みの自動ダウンロードを一切行わず、完全ローカルかつ決定論的な幾何・画像処理（PyMuPDF, OpenCV, NumPy, SciPy, pandas）によって高速・高精度に動作します。

AI エージェント（Claude Desktop, Antigravity, OpenAI Function Calling 等）が本ツール群を利用する際は、以下の 3 つの王道ワークフローおよびパラメータ指針に従うことで、最高の精度を発揮できます。

---

## 1. 典型的な 3 大デジタイズワークフロー (Standard Workflows)

グラフの形式に応じて、以下の 3 つのワークフローを選択してください。

### パターン A: PDF ネイティブベクターグラフ (最高精度・解像度劣化ゼロ)
電子ジャーナル（デジタル PDF）内のプロット線がベクターパスとして描画されている場合のパイプラインです。

1. **`search_pdf_primitives`**:
   * 検索クエリ（例: `"Figure 5"`, `"Fig. 4"`）で図表の掲載ページとキャプションのバウンディングボックス (`bbox` in PDF points) を特定します。
2. **`inspect_pdf_primitives`**:
   * 該当ページを検査し、`is_scanned: false` かつベクター線分・ベジェ曲線群が存在することを確認します。
3. **`crop_and_transform_region`**:
   * `caption_bbox` にキャプションの bbox を渡し、図表領域をスマートに自動推定して 300 DPI でクロップ画像を生成します（`estimated_bbox` を取得）。
4. **`auto_calibrate_axes`**:
   * `mode="tick_matched"`, `pdf_path`, `page_number`, `crop_bbox_points=estimated_bbox`, `dpi=300.0` を渡して実行。
   * PDF 埋め込みテキストと目盛り線を照合し、`x_calibration` と `y_calibration` を全自動算出します。
5. **`extract_vector_curve_points`**:
   * `drawing_indices=[...]`（抽出対象の曲線描画 ID 列）、`group_by_color=True`, `dpi=300.0` を指定して実行。
   * 同一ストローク色のパスが自動統合され、`color_grouped_curves` が返却されます。
6. **`calibrate_and_convert_coordinates`**:
   * `curves=color_grouped_curves`, `x_calibration`, `y_calibration`, `output_format="wide"`, `num_grid_points=100` を指定して CSV を出力します。
7. **`render_verification_overlay`**:
   * 元画像と CSV データを照合した半透明色分け画像を生成し、`alignment_metric` を確認します。

---

### パターン B: ラスタ・目盛り数値ありグラフ (標準的なスキャン・画像プロット)
画像として埋め込まれたグラフやスキャン PDF で、軸に数値目盛りが印字されている場合です。

1. **`crop_and_transform_region`**:
   * 図表領域を 300 DPI でクロップ（必要に応じて `deskew=True` で傾き補正）。
2. **`detect_axes_and_ticks`**:
   * X 軸・Y 軸の位置および目盛り線のピクセル候補列を特定します。
3. **`detect_legend_region`** (または **`detect_plot_colors`**):
   * 凡例枠（`legend_bboxes`）を取得してプロット抽出時の除外対象とします。
   * 各凡例項目の代表色と推奨 HSV 範囲（`suggested_hsv_lower`, `suggested_hsv_upper`）を取得します。
4. **`auto_calibrate_axes`**:
   * `mode="tick_matched"` で実行。OCR または埋め込みテキストから校正パラメータ（`x_calibration`, `y_calibration`）を算出。
5. **`extract_plot_pixels_by_color`**:
   * 各曲線について、`hsv_lower`, `hsv_upper`（または `target_hex`）、`exclude_bboxes=legend_bboxes`、`smooth_filter=True` を指定して抽出。
6. **`calibrate_and_convert_coordinates`**:
   * 抽出されたピクセル点列を物理量へ変換し CSV 出力。
7. **`render_verification_overlay`**:
   * 検証オーバーレイを生成して抽出精度を目視確認。

---

### パターン C: ラスタ・目盛り数値なし／定性グラフ／任意単位 (a.u.) (例: c.pdf Fig 4a 形式)
ラマン分光、XRD、吸収スペクトル、規格化曲線など、目盛りに具体的な数値がなく、外枠（Box Frame）や任意単位（arbitrary units, a.u.）で描画されたグラフの場合です。

1. **`crop_and_transform_region`**:
   * 300 DPI で対象プロット領域を高解像度クロップ。
2. **`detect_axes_and_ticks`** (`detect_box_frame=True`):
   * グラフを囲む外枠矩形（`box_frame["inner_bbox"]`）を取得します。
3. **`detect_legend_region`** (または **`detect_plot_colors`**):
   * 凡例除外枠および各曲線の代表色（推奨 HSV 範囲）を取得します。
4. **`auto_calibrate_axes`**:
   * `mode="normalized"` を指定し、`box_frame_bbox=box_frame["inner_bbox"]`、`normalized_domain_x=[x_min, x_max]`（例: キャプションや本文記載の波長範囲など）、`normalized_domain_y=[0.0, 1.0]` を渡して実行。
   * 外枠内寸を基準とした正規化校正パラメータが生成されます。
5. **`extract_plot_pixels_by_color`**:
   * 曲線ごとに以下を指定して抽出します：
     * `hsv_lower`, `hsv_upper`: 凡例から得られた推奨 HSV 範囲。
     * `smooth_filter=True`, `max_jump=15.0`: ラスタ特有のジャギや跳びを除去。
     * `x_range=[x0, x1]`: 曲線が特定の X 区間（ピーク付近）に集中している場合、その区間を指定してノイズ混入や重複を防止。
6. **`calibrate_and_convert_coordinates`**:
   * `curves=[{"label": "...", "points": ...}, ...]`, `output_format="wide"`, `num_grid_points=200`, `extrapolate=False` を指定。
   * `extrapolate=False` により、特定区間のみ存在する曲線は定義域外が正しく `NaN` となり、フラットな端点値で埋まるのを防ぎます。
7. **`render_verification_overlay`**:
   * オーバーレイ画像を確認し、全曲線が原図のピーク・ショルダーと合致していることを検証します。

---

## 2. パラメータ指定の勘所とベストプラクティス

| ツール / パラメータ | 推奨設定・使い分けの基準 |
| :--- | :--- |
| **`crop_and_transform_region`**<br>`caption_bbox` | `search_pdf_primitives` で得たキャプションの `bbox` を渡すと、直上の図表全体を自動推定 (`estimated_bbox`) して切り出してくれます。手動でピクセル座標を測る必要がありません。 |
| **`detect_axes_and_ticks`**<br>`detect_box_frame=True` | 四方を枠線で囲まれたグラフの枠内寸 (`inner_bbox`) を高精度に検出します。目盛り数値のない定性グラフで必須となります。 |
| **`auto_calibrate_axes`**<br>`mode="tick_matched"` vs `"normalized"` | 数値目盛りがあるグラフには `"tick_matched"`、数値のない任意単位 (a.u.) グラフには `"normalized"` を指定します。なお `fallback_to_normalized=True`（デフォルト）にしておくと、目盛り照合に失敗しても自動で枠ベース正規化へ移行するため処理が停止しません。 |
| **`extract_plot_pixels_by_color`**<br>`hsv_lower` / `hsv_upper` | RGB や Hex よりも、HSV 範囲指定が最も耐ノイズ性・分離能に優れています。`detect_legend_region` の `legend_items` や `detect_plot_colors` の戻り値に含まれる `suggested_hsv_lower` / `suggested_hsv_upper` をそのまま渡すのがベストプラクティスです。 |
| **`extract_plot_pixels_by_color`**<br>`smooth_filter=True` | ラスタ画像から連続曲線を抽出する際は常に `True` を推奨します。局所移動中央値から `max_jump`（標準 15 px）以上離れた外れ値を自動除去し、滑らかな連続曲線を生成します。 |
| **`extract_plot_pixels_by_color`**<br>`x_range=[x_min, x_max]` | 複数曲線が交差したり、ピーク部分のみ着目したい場合に指定します。指定範囲外の誤抽出を完全に防ぐことができます。 |
| **`calibrate_and_convert_coordinates`**<br>`output_format="wide"` & `num_grid_points` | 複数曲線をまとめて分析・CSV 出力する場合、`wide` を指定すると共通 X 格子上に全曲線を線形補間した 1 つの表形式（列: `x, curve1, curve2...`）で出力されます。 |
| **`calibrate_and_convert_coordinates`**<br>`extrapolate=False` | 各曲線で X の測定区間が異なる場合、必ず `extrapolate=False` を指定してください。定義域外が `NaN` になり、端点の平坦化外挿を防げます。 |
| **`render_verification_overlay`** | デジタイズ完了後、必ずこのツールで透過オーバーレイ画像を生成し、`alignment_metric`（通常 0.6〜0.9 以上）と目視で適合度を確認してください。 |

---

## 3. トラブルシューティングと自己修復ガイド (Self-Healing Tips)

1. **目盛りの数値テキストがうまく認識されない場合**:
   * 電子 PDF であれば、`pdf_path` と `crop_bbox_points` を `auto_calibrate_axes` に渡してください。OCR を介さず直接埋め込みテキストを読み取るため、誤認識が起きません。
   * 目盛りの数字が印刷されていないグラフの場合は、迷わず `mode="normalized"` と `box_frame_bbox`（または自動検出）に切り替えてください。
2. **色抽出で凡例や軸の文字が混入してしまう場合**:
   * `detect_legend_region` を実行して得られた `legend_bboxes` を、`extract_plot_pixels_by_color` の `exclude_bboxes` に渡してください。
3. **抽出した曲線にトゲのようなノイズや跳びがある場合**:
   * `smooth_filter=True` を指定し、必要に応じて `max_jump` を `10.0` 〜 `15.0` に設定してください。
4. **複数の曲線が一部で同色に見えて分離できない場合**:
   * `x_range=[x_min, x_max]` を使って曲線ごとに抽出する X ピクセル範囲を区切るか、`detect_plot_colors` でより細かい色クラスタを取得してください。
