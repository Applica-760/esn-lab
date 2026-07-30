# TASK-005: group b test fold別3×2出力軌跡

スパコン上の `esn-lab` リポジトリルートで、`tasks/005` を取得して実行する。

```sh
git fetch origin tasks/005
git checkout tasks/005
git pull --ff-only origin tasks/005
uv run -m projects.tasks.cli analysis.margin.output_trajectory --config cfg_group_b_fold_trajectory.yaml
```

## 設定と成果物

- 設定: `projects/tasks/analysis/margin/output_trajectory/cfg_group_b_fold_trajectory.yaml`
- 入力: `outputs/experiments/pred_margin_beta_2class/b/Nx400_dens0.5_inscl0.001_rho0.9/test_results.npz`
- 対象: group `b`、test fold `0`〜`4`、warmup比 `0.1`、100 bin
- 出力: `outputs/experiments/analysis_margin_beta_2class_fold_trajectory/Nx400_dens0.5_inscl0.001_rho0.9/b/fold_{0,1,2,3,4}/score_trajectory_3x2.png`
