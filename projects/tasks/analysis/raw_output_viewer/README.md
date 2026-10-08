# 生出力ビューアーの要件

詳細な合意事項は [research-hub Issue #1 / Contract 1](https://github.com/Applica-760/research-slide/issues/1#issuecomment-6062796301) を参照する。

## 保存済みの生出力を個別に見る

各時刻の最大出力ノードの多数決に限定せず、判定ロジックを検討するためのビューアー。採食・反芻の2ラベルで学習したモデルに、採食・反芻のデータを入力した結果を対象とする。正誤で分けず、複数データの平均・標準偏差にも集約しない。パラメータ条件ごとの出力も比較する。

追加の学習・推論は行わない。スパコンの実行済み結果をローカルへ取得し、ローカルでホストを起動する。

## 条件とIDを選んで描画する

- ID一覧とは別にgroup・foldの選択UIを設ける。
- 左側にスクロール可能なID一覧を置き、訓練／テストと真値（採食／反芻）のバッジ・フィルターを付ける。
- IDクリックで各ノードの生出力を描いた2画像を取得し、上段に採食ノード、下段に反芻ノードの画像を配置する。ズームは実装しない。
- 横軸は保存系列の先頭を0とし、データごとの系列長の違いを許容する。縦軸は−0.2〜1.2に固定する。
- 縦軸の範囲外の値があっても画像を描画し、画面上に警告を表示する。データの欠損はエラーとして表示する。

## Task層でデータを提供し、App層で表示する

Task層はFastAPIによる起動・結果読み込み・APIを担う。グラフはPythonのMatplotlibでノードごとに描画し、別々の2画像として返す。フロントエンドはHTML・JavaScriptで実装して `projects/app/` に置き、画像の配置を担う。

選択中の結果ファイルを保持し、毎クリックで読み直さない。サンプルはID・実験条件・訓練／テストの区分で識別する。

## バックエンドの起動とAPI

`cfg.yaml` の `pred_result_dir` に、ローカルへ取得した予測結果のルートを指定する。既存の `group/parameter/train_results.npz`・`test_results.npz` の配置を使い、同名のJSONにも対応する。両形式がある場合は既存ローダーと同じくJSONを優先する。ノード順は2ラベル変換処理に合わせて採食・反芻とする。

リポジトリルートで実行する。

```sh
uv run -m projects.tasks.cli analysis.raw_output_viewer
```

`127.0.0.1:8000` で起動する。ポートは `cfg.yaml` で変更できる。`output_dir` は既存Taskの設定保存先であり、画像は保存しない。

| GET API | 引数 | 応答 |
| --- | --- | --- |
| `/api/conditions` | なし／`group`／`group, parameter` | それぞれ `groups`／`parameters`／`folds` |
| `/api/samples` | `group, parameter, fold` | `samples`: ID・mode・true_label・lengthの一覧 |
| `/api/plots` | `group, parameter, fold, mode, id` | 条件・ID・真値・系列長、`images`、`warnings` |

`mode` は `train` または `test`。`true_label` は `foraging` または `rumination`。`images.foraging` と `images.rumination` は別々のPNGをBase64にした文字列で、応答の `mime_type` は `image/png`、`encoding` は `base64`。

範囲外でもHTTP 200で画像と `warnings` を返す。警告は `code: out_of_range`、対象の `node`、`message` を持つ。警告がなければ空配列となる。

表示できない場合は `error: {code, message}` を返す。HTTP 404は `not_found`、422は `invalid_request` または `invalid_data`、500は `render_failed`。欠損・非有限値・不正な形状・重複IDはデータ不備として扱う。

選択中のgroup・parameterのtrain/testをメモリに保持する。閲覧中に結果ファイルを差し替えた場合は再起動する。フロントエンドは未実装。
