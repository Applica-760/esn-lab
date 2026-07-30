# TASK-005: 2クラス出力軌跡（3×2）

スパコン上の `esn-lab` リポジトリルートで、TASK-005 の worktree branch `tasks/005` をcheckoutして実行する。

```sh
git fetch origin tasks/005
git checkout tasks/005
uv run -m projects.tasks.cli analysis.margin
```

## 設定と成果物

- 解析設定: `projects/tasks/analysis_margin/cfg.yaml`
- 入力: `outputs/experiments/pred_margin_beta_2class/` 配下の各 group・parameter directory の `test_results.npz`
- 出力: `outputs/experiments/analysis_margin_beta_2class/` 配下の各 parameter directory の `score_trajectory_3x2.png`
- 行: 真値 `foraging`、`rumination`、`other`
- 列: warmup後のフレーム単位argmax多数決による予測 `foraging`、`rumination`
- 各セル: foraging・rumination出力の時系列平均と標準偏差帯
