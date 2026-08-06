# `projects/tasks` の構造と再編境界

この文書は、`esn-lab:006`（TASK-006）で `projects/tasks/` を再編するための基準を定める。research-hubに記録されていた2026-07-30時点のスナップショットを移管し、2026-08-06時点の `aa497d5` で現行実装と照合した。

## 実行契約

実行入口は `projects/tasks/cli.py` の `TASK_REGISTRY` である。

```console
uv run -m projects.tasks.cli <CLI名> [--config <設定ファイル名>]
```

| CLI名 | 実装モジュール | 役割 |
| --- | --- | --- |
| `train` | `projects.tasks.train.app` | ESNの学習と重み保存 |
| `pred` | `projects.tasks.pred.app` | 学習済み重みによる予測結果保存 |
| `pred.margin` | `projects.tasks.pred_margin.app` | 2クラス重みで真値3群を予測 |
| `eval.dist` | `projects.tasks.eval.dist.app` | 判定結果の分布評価 |
| `eval.dist_node` | `projects.tasks.eval.dist_node.app` | ノード出力の分布評価 |
| `eval.judge` | `projects.tasks.eval.judge.app` | 予測結果から判定結果を生成 |
| `eval.metrics` | `projects.tasks.eval.metrics.app` | 判定結果から評価指標を集計 |
| `eval.plot` | `projects.tasks.eval.plot.app` | 個別予測の可視化 |
| `analysis.pred` | `projects.tasks.analysis.pred.app` | 予測時系列の分析 |
| `analysis.margin` | `projects.tasks.analysis.margin.distribution.app` | マージン分布分析の互換入口 |
| `analysis.margin.distribution` | `projects.tasks.analysis.margin.distribution.app` | マージン分布分析 |
| `analysis.margin.output_trajectory` | `projects.tasks.analysis.margin.output_trajectory.app` | 真値3群×予測2群の出力軌跡分析 |
| `analysis.bayesian` | `projects.tasks.analysis.bayesian.app` | 判定パラメータのベイズ最適化 |

設定は実装モジュールと同じディレクトリの `cfg.yaml` を既定値とする。`--config` は任意パスではなく、同じディレクトリにある指定名のファイルを選ぶ。選択した設定は `output_dir/config.lock.yaml` へコピーする。この解決規則、CLI名、`analysis.margin` の互換入口を維持する。

## 現在の構造

```text
projects/tasks/
  cli.py
  train/
  pred/
  pred_margin/
  eval/
    dist/
    dist_node/
    judge/
    metrics/
    plot/
  analysis/
    pred/
    bayesian/
    margin/
      common.py
      distribution/
      output_trajectory/
```

- 直下の `train`、`pred`、`pred_margin` は処理段階ごとの単独アプリである。
- `eval/` は判定結果を入力とする評価・可視化アプリをまとめる。
- `analysis/` は予測結果を入力とする分析アプリをまとめる。
- `analysis/margin/common.py` は、マージン分布と3×2出力軌跡が共用する結果読込、warmup除外、真値群の集計を担う。

```text
train -> pred -> eval.judge -> eval.{dist,dist_node,metrics,plot}
          |-> analysis.pred

pred.margin -> analysis.margin.{distribution,output_trajectory}
```

`analysis.bayesian` は予測結果と `projects/utils/eval/` の判定・評価関数を直接利用する。

## 現在の共通コード

| モジュール | 主な責務 |
| --- | --- |
| `projects/utils/app_init.py` | 設定読込、既定設定パスの解決、fold読込、パラメータグリッド |
| `projects/utils/prediction.py` | 予測結果のNPZ読込・保存と有効性確認 |
| `projects/utils/weights.py` | 重み・パラメータディレクトリの読込と保存 |
| `projects/utils/eval/` | 判定、フィルタ、混同行列、指標、評価可視化 |

## 再編対象

### 評価入力処理

`eval.dist` と `eval.dist_node` は、次の入力処理を重複して持つ。

1. `judgment_results_<mode>.csv` の読込
2. 設定されたfilterの適用
3. 判定結果の `group`、`fold_index` 単位への整理
4. 対応する予測結果の読込
5. `(group, fold_index, id)` による判定結果と予測サンプルの結合

この処理を `projects/utils/eval/` へ抽出する。入力ファイルがない場合やfilter結果が空の場合をデータなしとして扱う現在の制御は維持する。一方、結合対象に入った複合キーが欠損または重複している場合は、`param_name`、`mode`、`group`、`fold_index`、`id` を特定できる明示的なエラーにする。

比率・confidence・marginなどの計算、カテゴリ別集計、中間CSV生成、作図は各アプリ固有の責務として残す。

### 予測分析

`analysis.pred.app` に同居する責務を次へ分割する。

- サンプル正規化: warmup除外、真値・予測クラスの決定
- 集計計算: 時間bin、正答率、安定性、スコア軌跡、マージン
- 作図: 個別図、true×pred図、fold横断summary
- 実行オーケストレーション: 設定、param・mode・group・foldの走査、入出力配置

`app.py` と `main(cfg)` はCLIから呼ばれるオーケストレーション入口として維持する。分割後の計算処理は、ファイルI/Oを伴わない単体テスト可能な関数を境界とする。

## 互換性を維持する出力

`<output_dir>` は選択した設定の `output_dir`、`<param>` は `build_param_str()` の結果を表す。

### `eval.dist`

- `intermediate/<param>_<mode>_ratios.csv`
- `<mode>/individual/dist_true<true_name>_argmax<pred_name>.png`
- `<mode>/dist_confusion_<mode>.png`
- `<mode>/split/dist_split_true<true_name>_<mode>.png`

### `eval.dist_node`

- `<mode>/node_<metric>_<category>.png`
  - `<metric>`: `confidence`、`margin`、`true_class_output`
  - `<category>`: `all`、`correct`、`incorrect`、各class名
- `<mode>/node_output_confusion_<mode>.png`（`node_output.enabled` の場合）

### `analysis.pred`

`modes` が設定に存在する場合は `<mode>/` を先頭に加え、存在しない既定設定では加えない。

- `[<mode>/]<param>/<group>/fold_<fold_index>/temporal_accuracy.png`
- `[<mode>/]<param>/<group>/fold_<fold_index>/score_trajectory.png`
- `[<mode>/]<param>/<group>/fold_<fold_index>/stability.png`
- `[<mode>/]<param>/<group>/fold_<fold_index>/argmax_heatmap.png`
- `[<mode>/]<param>/<group>/fold_<fold_index>/score_margin.png`
- `[<mode>/]<param>/<group>/fold_<fold_index>/temporal_accuracy_3x3.png`
- `[<mode>/]<param>/<group>/fold_<fold_index>/score_trajectory_3x3.png`
- `[<mode>/]<param>/<group>/fold_<fold_index>/stability_3x3.png`
- `[<mode>/]<param>/<group>/fold_<fold_index>/score_margin_3x3.png`
- `[<mode>/]<param>/<group>/summary/temporal_accuracy_folds.png`
- `[<mode>/]<param>/<group>/summary/stability_folds.png`

## 対象外

- `src/esn_lab/` のESN計算、学習、予測処理
- `train`、`pred`、`pred.margin`、`eval.judge`、`eval.metrics`、`eval.plot`、`analysis.margin.*`、`analysis.bayesian` の責務変更
- 各指標の定義、集計軸、図の内容、設定項目・既定値、出力階層・ファイル名の変更
- 履歴である既存 `docs/operations/runs/` の変更
- `projects/tasks/analysis_margin/` の追跡残骸整理（最終検証Phaseで扱う）
- research-hub側の移管元文書の削除（Lab側文書のmain統合後に扱う）

既存run文書に記録された `uv run -m projects.tasks.cli ...` コマンドは変更しない。再編は内部責務の移動に限定し、CLI、設定解決、入力の意味、成果物を外部契約として回帰テストする。
