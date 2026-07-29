# TASK-002: 2クラスESNのその他マージン分析

- 実行commit: `27fb41515cb8f3be483e8f5486f54c3c479fa767`（push済み）
- 実行場所: スパコン上の `esn-lab` リポジトリルート

## ジョブ `.sh` のコマンド

```sh
git fetch origin tasks/002
git checkout --detach 27fb41515cb8f3be483e8f5486f54c3c479fa767
uv run -m projects.tasks.cli pred.margin
uv run -m projects.tasks.cli analysis.margin
```

## 設定と成果物

- 推論設定: `projects/tasks/pred_margin/cfg.yaml`
- 解析設定: `projects/tasks/analysis_margin/cfg.yaml`
- 推論結果ルート: `outputs/experiments/pred_margin_beta_2class/` 配下の各 group・parameter directory の `test_results.npz`
- 推論設定のlock: `outputs/experiments/pred_margin_beta_2class/config.lock.yaml`
- 解析成果物ルート: `outputs/experiments/analysis_margin_beta_2class/` 配下の各 parameter directory の `margin_by_sample.csv`、`margin_summary.csv`、`margin_distribution.png`
- 解析設定のlock: `outputs/experiments/analysis_margin_beta_2class/config.lock.yaml`
