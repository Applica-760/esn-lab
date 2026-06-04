# pred生値分析タスクの実装（判定ロジック設計のため）

## 目的・前提・方針案

**目的**：`train_results.npz` の生値を分析し、データに根拠を持つ判定ロジック設計のための構造的知見を得る

**前提**：
- 分析対象：`train_results.npz`（trainモードのみ。testは一切触らない）
- ESN出力は線形写像（softmaxではない）
- **リーク回避**：fold × group 単位で分析。pool後のパラメータ推定はしない
- 得た知見は**パラメータフリーな構造的ルール**として実装する（閾値フィットはしない）
- 先頭10%フレームはウォームアップとして除外（ESNのtransient期間の扱い）
- **配置**：`projects/tasks/analysis/pred/`（toolではなくtask。設定管理・繰り返し実行が必要なため）
- 出力：`outputs/tasks/analysis/pred/`
- `TASK_REGISTRY` に `"analysis.pred"` として登録
- 分析後、結果レビューと判定ロジック実装は別タスク

**分析観点**（3つに絞る）：
1. **時刻別正答率**：相対位置（0〜100%）ごとのフレーム正答率 ← 重み付けの必要性を確認
2. **スコア軌跡**：クラス別スコアの時系列平均（正解/誤分類サンプル別）← 安定化タイミングを確認
3. **within-sample安定性**：サンプル内でのargmax変動率 ← 信頼度重み付けの有効性を確認

**対象パラメータ**：`cfg.yaml` で `param_str` を固定指定（単一 param_dir を対象）

## 計画

### Phase 1: 分析タスク実装（`tasks/analysis/pred/`）

- [ ] `cfg.yaml`：pred_result_dir, groups, param_str固定, class情報
- [ ] `app.py`：group × fold 単位で `train_results.npz` を読み込み、先頭10%フレームを除外
- [ ] 時刻別正答率の計算・可視化：相対フレーム位置（0〜100%）× 正答率、group/fold別に出力
- [ ] スコア軌跡の計算・可視化：正解/誤分類サンプルのクラス別スコア平均の時系列、group/fold別
- [ ] within-sample安定性の計算・可視化：サンプル内argmax変動率の分布（正解/誤分類別）、group/fold別
- [ ] `TASK_REGISTRY` に `"analysis.pred"` を追加

### Phase 2: 結果レビュー → 手法選定（別タスク・ユーザーと協議）

### Phase 3: 判定ロジック実装（別タスク）

## 実行ログ

## 結果
