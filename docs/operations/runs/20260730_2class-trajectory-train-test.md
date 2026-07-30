# TASK-004: 2クラスtrajectoryの100 bin・train/test再出力

- 実行commit: `a69fd359c62f93a5b8822ee697ebcae3501405aa`（`tasks/004` branch）
- 実行場所: スパコン上の `esn-lab` リポジトリルート

## ジョブ `.sh` のコマンド

```sh
git fetch origin tasks/004
git checkout --detach a69fd359c62f93a5b8822ee697ebcae3501405aa
uv run -m projects.tasks.cli analysis.pred --config cfg_2class_trajectory.yaml
```

## 設定と成果物

- 解析設定: `projects/tasks/analysis/pred/cfg_2class_trajectory.yaml`
- 入力: `outputs/experiments/pred_2class/b/Nx400_dens0.5_inscl0.001_rho0.9/{train,test}_results`
- 対象: group `b`、fold `0`〜`4`、warmup比 `0.1`、100 bin
- 出力: `outputs/experiments/analysis_pred_2class_trajectory/{train,test}/Nx400_dens0.5_inscl0.001_rho0.9/b/fold_{0,1,2,3,4}/`
- 主要図: 各foldの `score_trajectory_3x3.png`
- 設定lock: `outputs/experiments/analysis_pred_2class_trajectory/config.lock.yaml`
