# pred_results JSON後方互換フォールバック

## 目的・前提・方針案

**目的**：`analysis.pred` タスクで旧形式（`.json`）の pred_results を読み込めるようにする

**前提**：
- `20260516_1438_pred-results-size-reduction` で JSON → NPZ に移行済み（後方互換なし）
- 旧 JSON の構造は NPZ `load_pred_results` の戻り値と同一なので `json.load` の結果をそのまま返せる
- 新規保存は引き続き NPZ。読み込み時のみフォールバック

**方針**：`projects/utils/prediction.py` のみ変更。フォールバック方式（`.json` が存在すれば JSON 読み込み）を採用

## 計画

### Phase 1: フォールバック実装
- [x] `is_valid_result_file`：`.json` 存在時は JSON として検証
- [x] `load_pred_results`：`.json` 存在時は JSON で読み込み

### Phase 2: 動作確認
- [x] NPZ パスが壊れていないことを確認（既存の呼び出しパスを python -c でインポート検証）

## 実行ログ

## 結果

`projects/utils/prediction.py` に `_json_path` ヘルパーを追加し、`is_valid_result_file` / `load_pred_results` に JSON フォールバックを実装。インポート確認・JSON/NPZ 両パスの動作確認済み。変更ファイル1件のみ。
