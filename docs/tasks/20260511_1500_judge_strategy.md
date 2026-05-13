# judge 判別ロジックの差し替え可能化

## 目的・前提・方針案

**目的**：`eval.judge` の判別ロジックを差し替え可能にし、多数決以外の戦略で実験できるようにする

**前提**：
- 現状：`compute_judgment_results` に `judge_sample_by_majority_vote` が固定埋め込み
- 入出力インターフェース：`judge_fn(predictions, labels) → {pred_label, true_label, is_correct}`
- ダウンストリーム（metrics/dist/dist_node/plot）はCSV形式が不変なので無変更

**方針**：
- `judgment.py`：レジストリ `JUDGE_STRATEGIES` と戦略関数を定義（ファイル分離しない）
- `app.py`：cfg の文字列を `compute_judgment_results` にそのまま渡すだけ
- `cfg.yaml`：`judge_strategy: majority_vote` を追加

## 計画

### Phase 1: 戦略の差し替え基盤
- [x] `judgment.py` に `JUDGE_STRATEGIES` を定義し、`compute_judgment_results` の引数を `strategy: str` に変更してレジストリ引き当てを内部化
- [x] `app.py` の `compute_and_save_judgments` に `strategy` を追加し、cfg の文字列を渡す
- [x] `cfg.yaml` に `judge_strategy: majority_vote` を追加

## 戦略候補メモ（次タスクで実装）

実装推奨順：

| 優先 | キー | ロジック | 備考 |
|---|---|---|---|
| 1 | `mean_score` | フレームスコアベクトルを平均 → argmax | ソフト集約。一様効果の確認 |
| 2 | `weighted_mean_score` | `mean_score` の集約後スコアにクラス重み w_c を乗算 → argmax | ラベル偏りへの介入。実装コスト低 |
| 3 | `threshold_vote` | クラスごとに閾値 θ_c を設定した多数決（ruminationは低めに） | θ の決定に訓練統計が必要 |
| 4 | `exp_mean_score` | exp(スコア)/frame → 平均 → argmax | 高確信フレームを強調 |

**背景**：ruminationはフレーム単位の識別信頼性が低く（対角ratioが一様分散）、50%閾値の多数決に構造的に不利。ラベル偏りに作用するには案2以降が必要。

## 実行ログ

## 結果
