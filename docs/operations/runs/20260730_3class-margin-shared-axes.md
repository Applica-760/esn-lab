# TASK-003: 3真値群マージン分布の共通軸再出力

- 実行commit: `05353a09319613679f55b0a124eae42d51fee8eb`（push済み）
- 実行場所: スパコン上の `esn-lab` リポジトリルート

## ジョブ `.sh` のコマンド

```sh
git fetch origin main
git checkout --detach 05353a09319613679f55b0a124eae42d51fee8eb
uv run -m projects.tasks.cli analysis.margin
```

## 設定と成果物

- 解析設定: `projects/tasks/analysis_margin/cfg.yaml`
- 入力: `outputs/experiments/pred_margin_beta_2class/` 配下の各 group・parameter directory の `test_results.npz`
- 出力: `outputs/experiments/analysis_margin_beta_2class/` 配下の各 parameter directory の `margin_by_sample.csv`、`margin_summary.csv`、`margin_distribution.png`
- 解析設定のlock: `outputs/experiments/analysis_margin_beta_2class/config.lock.yaml`
- 作図設定: x軸 `0.0–3.3`、20 bin、3真値群で共有するCountのy軸
