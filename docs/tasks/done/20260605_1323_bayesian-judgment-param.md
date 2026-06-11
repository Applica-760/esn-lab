# ベイズ最適化による判定ロジックパラメタ決定

## 目的・前提・方針案

### 目的
判定ロジックに係るパラメタをOptunaによるベイズ最適化で決定する。

### 前提
- ESN出力（フレームごとのスコア）はnpzとして保存済み。パラメタ評価はnpzの後処理のみで完結するため、1評価あたりのコストは非常に安価
- 現状の判定ロジック（`majority_vote`, `mean_score`）はパラメタなし。最適化のためにはパラメタを持つ判定ロジックを新設する必要がある
- データ構造は10-fold CV。`mode=train` / `mode=test` のnpzが別途存在する
- タスク新設先：`projects/tasks/analysis/bayesian`

### 採用パラメタ（確定）
- `w0, w1, w2`：クラス（foraging / rumination / other）ごとのスコア重み
- `margin_threshold`：margin(max − 2nd_max) < threshold のフレームを棄権　← ~~`confidence_threshold`~~ は棄却（どうせ無視される値にしか作用しない）
- ~~`warmup_ratio`：analysis.pred のプロットでほぼ無影響を確認済み、除外~~
- ~~`tail_weight`：フレーム位置への重みづけは不要~~
- 判定方式：argmax多数決を廃止。加重スコアの平均 → argmax へ移行

### 懸念・選択肢（決定済み）

**データリーク**：パラメタ探索は `mode=train` のnpzで行い、最終評価は `mode=test` で行う

**CV方式**：~~選択肢B（nested CV）~~ → **選択肢A採用**：全foldのtrain macro F1 を目的関数として1組決定

**最適化指標**：macro F1（3クラスバランス重視）

**実装方針**：
- 判定関数は `projects/tasks/analysis/bayesian/` ローカルに定義（実験的ロジックのため utils に置かない）
- データロードは既存の `utils/prediction.py:load_pred_results` を直接流用
- fold 単位で macro F1 を計算して平均を目的関数とする

## 計画

### Phase 1: 環境整備
- [x] `pyproject.toml` に `optuna` を追加（optuna 4.9.0）

### Phase 2: ベイズ最適化タスク実装
- [x] `projects/tasks/analysis/bayesian/cfg.yaml` 作成（pred_result_dir, groups, n_trials 等）
- [x] `projects/tasks/analysis/bayesian/app.py` 実装
  - `judge_sample` / `evaluate_samples` / `make_objective` / `run_optimization`：ローカル判定＋Optuna最適化
  - `one_process(group, fold_index, samples, ...)`：ProcessPoolExecutor から呼ぶ単位。(group, fold) ごとに独立実行
  - `compute_final_params` / `save_summary`：全結果の平均・std を集約して JSON/CSV 保存
  - データロードは `utils/prediction.py:load_pred_results` を `main` から直接利用
  - `cfg.yaml` に `n_trials`（1パターンあたりの探索回数）と `workers`（並列プロセス数）を追加
- [x] `projects/tasks/cli.py` に `"analysis.bayesian"` を登録

### Phase 4: テストセット評価
- [ ] 最適パラメタを `test_results` に適用して acc / macro F1 を出力・保存

## 実行ログ

## 結果
