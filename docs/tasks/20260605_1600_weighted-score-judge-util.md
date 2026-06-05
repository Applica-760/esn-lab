# 加重スコア判定ロジックのUtil共通化

## 目的・前提・方針案

### 目的
`analysis.bayesian` で探索した最適パラメタを `eval.judgement` で評価できるようにする。

### 前提
- `bayesian/app.py` にローカル定義された `judge_sample`（加重スコア＋margin棄権）と `utils/eval/judgment.py` の既存戦略が別実装になっており、divergence リスクがある
- `utils/eval/judgment.py` の既存判定関数は `(predictions, labels) -> dict` の固定シグネチャ。新戦略は追加パラメタが必要
- 探索済み最適パラメタ（`final_params.json` の `mean` フィールド）は `eval/judge/cfg.yaml` に手動転記する運用

### 方針
- `make_weighted_score_judge(class_weights, margin_threshold)` ファクトリ関数を `utils/eval/judgment.py` に追加し、既存 interface と統一
- `bayesian/app.py` のローカル `judge_sample` を削除し、共通関数に差し替え
- `eval/judge/cfg.yaml` に `judge_strategy: weighted_score` と `judge_params:` フィールドを新設
- `eval/judge/app.py` で `weighted_score` 戦略時に `judge_params` からファクトリを構築

## 計画

### Phase 1: Utils共通化
- [x] `projects/utils/eval/judgment.py` に `make_weighted_score_judge(class_weights, margin_threshold)` を追加
  - 内部ロジックは `bayesian/app.py:judge_sample` と同一（加重スコア平均 + margin棄権 + フォールバック）
  - 返り値は既存戦略と同形式 `{"pred_label": int, "true_label": int, "is_correct": bool}`
  - `compute_judgment_results` の `strategy` 引数を `str | callable` 両対応に変更

### Phase 2: bayesian 側の差し替え
- [x] `projects/tasks/analysis/bayesian/app.py` のローカル `judge_sample` を削除
- [x] データロードを `(preds, true_label_int)` → `(preds, labels_array)` に変更し、`evaluate_samples` でファクトリを直接呼び出し

### Phase 3: eval.judge 側の対応
- [x] `projects/tasks/eval/judge/app.py` に `_resolve_judge_fn(cfg)` を追加（`weighted_score` 時は `make_weighted_score_judge` を返す）
- [x] `projects/tasks/eval/judge/cfg.yaml` に `weighted_score` 用のコメントと `judge_params` フィールドを追加（デフォルトは既存 `majority_vote` のまま）

### Phase 4: 動作確認
- [x] import チェック・ファクトリの返り値形式を確認
- [ ] `analysis.bayesian` を実データで実行して既存出力と一致確認
- [ ] `eval.judge` で `judge_strategy: weighted_score` + `judge_params` を指定して正常出力確認

## 実行ログ

## 結果
