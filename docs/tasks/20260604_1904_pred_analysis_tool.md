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

- [x] `cfg.yaml`：pred_result_dir, groups, param_str固定, class情報
- [x] `app.py`：group × fold 単位で `train_results.npz` を読み込み、先頭10%フレームを除外
- [x] 時刻別正答率の計算・可視化：相対フレーム位置（0〜100%）× 正答率、group/fold別に出力
- [x] スコア軌跡の計算・可視化：正解/誤分類サンプルのクラス別スコア平均の時系列、group/fold別
- [x] within-sample安定性の計算・可視化：サンプル内argmax変動率の分布（正解/誤分類別）、group/fold別
- [x] `TASK_REGISTRY` に `"analysis.pred"` を追加

### Phase 2: 3×3プロットへの拡張

現状のプロットは true class（3パネル）軸のみ。サンプルレベルの多数決argmaxを「予測クラス」として descriptiveな軸に使い、(true class × 予測クラス) の3×3グリッドに拡張する。

- **予測クラスの定義**：サンプル内の全フレームargmaxの多数決（判定ロジック設計の前提ではなく、可視化の層別変数として使用）
- **対角セル**：正しく識別されたサンプル群
- **非対角セル**：どのクラスにどう誤分類されたかの詳細

拡張対象プロット：
- [x] `score_trajectory`：3×3パネル（true × pred）→ `score_trajectory_3x3.png`
- [x] `stability`：3×3分布 → `stability_3x3.png`
- [x] `score_margin`：3×3分布 → `score_margin_3x3.png`

### Phase 3: 結果レビュー → 手法選定（別タスク・ユーザーと協議）

### Phase 4: 判定ロジック実装（別タスク）

## 実行ログ

## 結果

## プロット一覧と解釈ガイド

### fold別出力（`fold_N/`）

#### `temporal_accuracy.png`
- **内容**：相対フレーム位置（0〜100%）ごとのフレーム正答率。true class別の折れ線
- **解釈**：折れ線が右上がりなら後半フレームが正確 → 後半重み付けの根拠になる。ほぼ水平なら時刻による重み付けの恩恵は小さい

#### `score_trajectory.png`
- **内容**：相対位置ごとのクラス別スコア平均（mean ± std シェーディング）。true class別に3パネル
- **解釈**：stdバンドが広い → fold内サンプル間でスコアが大きく異なる（平均が少数に引っ張られている可能性）。折れ線がほぼ平坦 → フレーム重み付けの効果は限定的。true classに関わらず同じクラスのスコアが常に高い → モデルが識別できていない可能性

#### `stability.png`
- **内容**：サンプル内のargmax変動率（= 隣接フレーム間でargmaxが変化した割合）のヒストグラム。true class別
- **解釈**：0付近に集中 → モデルの予測は安定している。spread が大きい → 不安定なサンプルが混在。変動率は「毎フレーム揺れているか」を捉えるため、単純な非最頻ラベル頻度より時系列のジッタを敏感に反映する

#### `argmax_heatmap.png`
- **内容**：サンプル（true class順ソート）× 相対位置のヒートマップ。色 = argmax クラス
- **解釈**：横一色の帯 → 安定サンプル。まだら → 不安定サンプル。stability ヒストグラムの定量結果を個別サンプルレベルで目視確認するための補助的プロット

#### `score_margin.png`
- **内容**：`margin = max_score − 2nd_max_score` のサンプル平均のヒストグラム。true class別
- **解釈**：マージンが大きい → モデルが決定的（1クラスに集中）。マージンが小さい → 上位2クラスが競っている。クラスによって分布形状が異なる場合、クラスごとに「決定しやすさ」が違う。信頼度の代理指標として利用可能だが、高マージン = 正しい予測とは限らない点に注意

### group別サマリー（`summary/`）

#### `temporal_accuracy_folds.png`
- **内容**：全fold の `temporal_accuracy` 曲線を重ね描き（細線 = 各fold、太線 = fold平均）。true class別に3パネル
- **解釈**：細線のばらつきが小さい → 時刻パターンはfoldをまたいで一般的な傾向。ばらつきが大きい → fold依存性が高く汎化しにくい

#### `stability_folds.png`
- **内容**：fold別の argmax 変動率分布を violin plot で横並び表示。true class別に3パネル
- **解釈**：violin の形がfold間で似ている → 安定性はfoldによらず一貫した傾向。あるfoldだけ violin が大きく異なる → そのfoldのデータに特異性がある可能性
