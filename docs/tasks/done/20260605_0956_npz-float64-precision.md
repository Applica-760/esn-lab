# NPZ保存精度をfloat64に変更

## 目的・前提・方針案

**目的**：NPZ保存時のfloat32変換による丸め誤差を排除し、JSON時代と同等の判定結果を得る

**前提**：
- 容量削減の逼迫した需要は解消済み。NPZ採用理由はバイナリによる読み書き高速化のみ
- 変更箇所は `projects/utils/prediction.py` の `save_pred_results` のみ
- 既存のNPZファイルは再生成が必要（フォーマット非互換）

**方針**：`predictions` / `labels` ともに `dtype=np.float64` で保存

## 計画

### Phase 1: dtype変更
- [x] `save_pred_results` の `dtype=np.float32` → `dtype=np.float64`（predictions・labels 両方）

### Phase 2: 動作確認
- [x] ラウンドトリップ検証（save → load で値・dtype が一致すること）

## 実行ログ

## 結果

`prediction.py` の `save_pred_results` を float64 に変更。ラウンドトリップ検証（値・dtype）で正常動作確認。既存NPZファイルは再生成が必要。