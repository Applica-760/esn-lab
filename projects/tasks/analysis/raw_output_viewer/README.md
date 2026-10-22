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
- IDクリックで元画像、両ノードの重ね描き、個別の採食・反芻ノードの順に表示する。元画像と重ね描きは横軸を揃え、採食が青、反芻がオレンジ。画像の列中心x＝保存出力の時刻インデックスx（先頭0）で、秒換算・ズームは行わない。元画像がない場合は重ね描きから表示する。
- 横軸は保存系列の先頭を0とし、データごとの系列長の違いを許容する。縦軸は−1.5〜2.5に固定する。
- 通常のグラフは1000×300px、元画像との比較図は1000×500px。画面では幅1080pxを上限に縦横比を保って表示し、収まらない場合はグラフ欄をスクロールする。
- 縦軸の範囲外の値があっても画像を描画し、画面上に警告を表示する。データの欠損はエラーとして表示する。

## Task層でデータを提供し、App層で表示する

Task層はFastAPIによる起動・結果読み込み・APIと、Matplotlibによるグラフ・比較図の生成を担う。フロントエンドはHTML・JavaScriptで実装して `projects/app/` に置き、画像の配置を担う。

選択中の結果ファイルを保持し、毎クリックで読み直さない。サンプルはID・実験条件・訓練／テストの区分で識別する。

## バックエンドの起動とAPI

`judgment_result_dir` は保存判定のルート。配下の `<parameter>/judgment_results_{train,test}.csv` を参照し、未指定・空なら判定表示を無効にする。

`source_image_dir` は元画像の参照先（現在は `data/all_s`）。`<ID>_ffmpeg.jpg`を読み、画像幅と選択系列長が一致した場合だけ比較図を生成する。未設定・画像なし・読込不正・幅不一致は画面に明示し、生出力グラフは表示する。元画像・outputsは変更せず、表示用データはメモリ上で扱う。対応の根拠は [#6の調査記録](https://github.com/Applica-760/research-slide/issues/6#issuecomment-6118593917)を参照。

`cfg.yaml` の `pred_result_dir` に、ローカルへ取得した予測結果のルートを指定する。既存の `group/parameter/train_results.npz`・`test_results.npz` の配置を使い、同名のJSONにも対応する。両形式がある場合は既存ローダーと同じくJSONを優先する。ノード順は2ラベル変換処理に合わせて採食・反芻とする。

リポジトリルートで実行する。

```sh
cd /Users/murakamitakumi/Developer/projects/esn-lab
uv run -m projects.tasks.cli analysis.raw_output_viewer
```

`127.0.0.1:8000` で起動する。ポートは `cfg.yaml` で変更できる。このTaskはファイルを出力せず、起動時にも設定を書き出さない。

| GET API | 引数 | 応答 |
| --- | --- | --- |
| `/api/conditions` | なし／`group`／`group, parameter` | それぞれ `groups`／`parameters`／`folds` |
| `/api/samples` | `group, parameter, fold` | `samples`: ID・mode・true_label・lengthの一覧 |
| `/api/sample-folds` | `group, parameter, id` | `folds`: Fold番号と指定IDの訓練／テスト区分の一覧 |
| `/api/plots` | `group, parameter, fold, mode, id` | 条件・ID・真値・系列長、`images`、`warnings` |

`/api/plots` は画像と独立した `judgment` も返す。`status` が `available` なら `pred_label`・`true_label`・`is_correct`、`disabled`・`not_found`・`invalid_data` なら `message` を含む。判定の欠落・不正でも画像は返す。判定CSVは現在のparameter・modeを保持するため、差し替え後は再起動する。WebUIのグラフ見出し直下に「［正答／誤判定］予測ラベル 得票率% ｜ 真値」を表示し、ID・Fold切替で更新する。%は予測ラベルの得票率を小数1桁で示す。保存判定と再計算が不一致なら得票率を結びつけず警告し、得票情報がない場合は確認不可と表示する。

`/api/plots` の `votes` は保存生出力の全時刻から算出した得票情報（`source: saved_predictions`、`total`、ラベル別の `counts`・`ratios`、`pred_label`）を返す。割合は0〜1で丸めず、同値・同票は採食を選ぶ。`matches_saved_judgment` は保存予測との一致／不一致を真偽値、照合不可を `null` で示す。入力ファイルは変更せず、集計結果も保存しない。

`mode` は `train` または `test`。`true_label` は `foraging` または `rumination`。`images.foraging`・`images.rumination`・`images.overlay` はそれぞれPNGをBase64にした文字列で、応答の `mime_type` は `image/png`、`encoding` は `base64`。

`source_image.status` は `available`・`disabled`・`not_found`・`invalid_data`・`width_mismatch`。対応可能な場合は寸法・対応規則と `images.comparison`（同じPNG/Base64形式）を返し、WebUIは重ね描き欄に比較図を表示する。

範囲外でもHTTP 200で画像と `warnings` を返す。警告は `code: out_of_range`、対象の `node`、`message` を持つ。警告がなければ空配列となる。

表示できない場合は `error: {code, message}` を返す。HTTP 404は `not_found`、422は `invalid_request` または `invalid_data`、500は `render_failed`。欠損・非有限値・不正な形状・重複IDはデータ不備として扱う。

`/api/sample-folds` は同じgroup・parameterの結果からIDを完全一致で探し、例として `{"folds":[{"fold":0,"modes":["test"]},{"fold":1,"modes":["train"]},{"fold":2,"modes":[]}]}` を返す。サンプルを含むFoldを昇順に並べ、指定IDがないFoldは空配列、両区分にある場合は `["train","test"]` とする。保存結果にないFold番号は補わない。全FoldにIDがなければ404、引数不足やデータ不備は422。Foldボタンは取得したfold・modeを既存の `/api/plots` に渡して描画する。

選択中のgroup・parameterのtrain/testをメモリに保持する。Fold一覧の取得でもこのキャッシュを共用し、同じ条件を再読込しない。閲覧中に結果ファイルを差し替えた場合は再起動する。

起動後、ブラウザで `http://127.0.0.1:8000/` を開く。パラメータ・Group・Foldの先頭候補が選択されるので、左側のIDをクリックする。必要に応じて条件を変更し、訓練／テストと真値で一覧を絞り込める。画面・CSS・JavaScriptは `projects/app/raw_output_viewer/` に配置し、同じFastAPIサーバーから配信する。

HTMLはキャッシュせず、CSS・JavaScriptのURLには各ファイルの更新日時を付ける。画面変更は通常の再読み込みで反映する。Python側の変更にはサーバー再起動が必要。

Fold切替中は古い画像を消し、連続操作で遅れて届いた応答は破棄する。別IDや一覧の条件を選ぶとFoldボタンも更新する。Fold一覧の取得に失敗した場合はIDを再選択し、描画に失敗した場合はFoldボタンを再度押して取得を試せる。
