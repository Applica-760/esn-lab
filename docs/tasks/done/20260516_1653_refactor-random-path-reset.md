# refactor: random/pathlib/reservoir リファクタリング

## 目的・前提・方針案
- `np.random.seed` のグローバル汚染を `default_rng` で排除（再現性確保）
- `os.path` を排除し `pathlib.Path` に統一
- Reservoir state リセットを `reset_reservoir_state()` 経由に統一
- 変更はいずれも軽微。1 MD・1 commit にまとめる
- `default_rng` 移行により同一 seed でも生成値が変わる（既存実験との後方互換は破棄許容）

## 計画

### Phase 1: np.random グローバル汚染排除
- [x] `Input.__init__`: `np.random.seed` → `rng = np.random.default_rng(seed)`、`rng.uniform()` を使用
- [x] `Reservoir.make_connection`: 同様に `rng.uniform()` を使用
- [x] `Output.__init__`: `rng.standard_normal()` を使用

対象: `src/esn_lab/model/esn.py`

### Phase 2: os.path → pathlib 統一
- [x] `weights.py`: `os.path.join` / `os.path.exists` / `os.makedirs` を `Path` 操作に置換、`import os` 削除
- [x] `prediction_cli.py`: `os.path.expanduser` → `.expanduser()`、`os.path.isdir` → `.is_dir()` に置換、`import os` 削除

対象: `projects/utils/weights.py`、`projects/tools/pred_plot/prediction_cli.py`

### Phase 3: Reservoir リセット統一
- [x] `trainer.py`: `model.Reservoir.x = np.zeros(model.N_x)` → `model.Reservoir.reset_reservoir_state()`、不要な `import numpy as np` も削除

対象: `src/esn_lab/pipeline/train/trainer.py`

## 実行ログ

## 結果
全 Phase 完了。変更ファイル 4 件。