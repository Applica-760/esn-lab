# 生出力ビューアーの要件

詳細な合意事項は [research-hub Issue #1 / Contract 1](https://github.com/Applica-760/research-slide/issues/1#issuecomment-6062796301) を参照する。

## 保存済みの生出力を個別に見る

各時刻の最大出力ノードの多数決に限定せず、判定ロジックを検討するためのビューアー。採食・反芻の2ラベルで学習したモデルに、採食・反芻のデータを入力した結果を対象とする。正誤で分けず、複数データの平均・標準偏差にも集約しない。パラメータ条件ごとの出力も比較する。

追加の学習・推論は行わない。スパコンの実行済み結果をローカルへ取得し、ローカルでホストを起動する。

## 条件とIDを選んで描画する

- ID一覧とは別にパラメータ・group・foldの順に選択UIを設ける。各欄の先頭候補を自動選択し、ID一覧まで読み込む。パラメータを変更した場合も、対応するgroup・foldの先頭を選ぶ。
- 左側にスクロール可能なID一覧を置き、訓練／テストと真値（採食／反芻）のバッジ・フィルターを付ける。
- ID選択後、プロット上部の「同じデータを別Foldで見る」にFold・訓練／テストのボタンを表示する。選択中を強調し、対象IDがないFoldは「データなし」として無効にする。両区分にあるFoldは別ボタンにする。
- FoldボタンではID・Group・パラメータを保持し、3画像だけを切り替える。上部の「データ一覧の条件」と左一覧は維持し、グラフ見出しに表示中のGroup・Fold・区分・真値を示す。左側のフィルターは一覧に対して適用する。
- IDクリックで3画像を取得し、採食ノード、反芻ノード、両ノードの重ね描きの順に配置する。重ね描きは採食が青、反芻がオレンジで、凡例を付ける。ズームは実装しない。
- 横軸は保存系列の先頭を0とし、データごとの系列長の違いを許容する。縦軸は−1.5〜2.5に固定する。
- Matplotlibの画像サイズは元の1000×300pxを使い、表示サイズはWeb側だけで調整する。1枚の表示上限は幅1080px・高さ300px。3枚を縦に配置し、収まらない場合はグラフ欄をスクロールする。画面が小さい場合は縦横比を保って縮小する。
- 縦軸の範囲外の値があっても画像を描画し、画面上に警告を表示する。データの欠損はエラーとして表示する。

## Task層でデータを提供し、App層で表示する

Task層はFastAPIによる起動・結果読み込み・APIを担う。グラフはPythonのMatplotlibで個別の2画像と重ね描き1画像を返す。フロントエンドはHTML・JavaScriptで実装して `projects/app/` に置き、画像の配置を担う。

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
| `/api/sample-folds` | `group, parameter, id` | `folds`: Fold番号と指定IDの訓練／テスト区分の一覧 |
| `/api/plots` | `group, parameter, fold, mode, id` | 条件・ID・真値・系列長、`images`、`warnings` |

`mode` は `train` または `test`。`true_label` は `foraging` または `rumination`。`images.foraging`・`images.rumination`・`images.overlay` はそれぞれPNGをBase64にした文字列で、応答の `mime_type` は `image/png`、`encoding` は `base64`。

範囲外でもHTTP 200で画像と `warnings` を返す。警告は `code: out_of_range`、対象の `node`、`message` を持つ。警告がなければ空配列となる。

表示できない場合は `error: {code, message}` を返す。HTTP 404は `not_found`、422は `invalid_request` または `invalid_data`、500は `render_failed`。欠損・非有限値・不正な形状・重複IDはデータ不備として扱う。

`/api/sample-folds` は同じgroup・parameterの結果からIDを完全一致で探し、例として `{"folds":[{"fold":0,"modes":["test"]},{"fold":1,"modes":["train"]},{"fold":2,"modes":[]}]}` を返す。サンプルを含むFoldを昇順に並べ、指定IDがないFoldは空配列、両区分にある場合は `["train","test"]` とする。保存結果にないFold番号は補わない。全FoldにIDがなければ404、引数不足やデータ不備は422。Foldボタンは取得したfold・modeを既存の `/api/plots` に渡して描画する。

選択中のgroup・parameterのtrain/testをメモリに保持する。Fold一覧の取得でもこのキャッシュを共用し、同じ条件を再読込しない。閲覧中に結果ファイルを差し替えた場合は再起動する。

起動後、ブラウザで `http://127.0.0.1:8000/` を開く。パラメータ・Group・Foldの先頭候補が選択されるので、左側のIDをクリックする。必要に応じて条件を変更し、訓練／テストと真値で一覧を絞り込める。画面・CSS・JavaScriptは `projects/app/raw_output_viewer/` に配置し、同じFastAPIサーバーから配信する。

HTMLはキャッシュせず、CSS・JavaScriptのURLには各ファイルの更新日時を付ける。画面変更は通常の再読み込みで反映する。Python側の変更にはサーバー再起動が必要。

Fold切替中は古い画像を消し、連続操作で遅れて届いた応答は破棄する。別IDや一覧の条件を選ぶとFoldボタンも更新する。Fold一覧の取得に失敗した場合はIDを再選択し、描画に失敗した場合はFoldボタンを再度押して取得を試せる。
