# pred_results ファイルサイズ削減

## 目的・前提・方針案

- `projects/tasks/pred` で生成される pred_results が大きすぎる
- 削減方法を検討・実装する

---

## 現状分析

### ファイル構造

```
outputs/experiments/pred_results/
  {group}/           # a〜j の10グループ
    {param_dir}/
      test_results.json    # 416MB / group
      train_results.json   # 3.7GB / group
```

**現時点の合計: 41GB**（param_dir は現在1つ。param数が増えると比例して増大）

### JSON の論理構造

```json
[
  {
    "fold_index": 0,
    "results": [
      {
        "id": "08_013319",
        "predictions": [[-0.0005255..., 0.0012118..., 0.0020418...], ...],  // T×Ny
        "labels":      [[1.0, 0.0, 0.0], ...],                              // T×Ny
      },
      ...  // ~27 sequences
    ]
  },
  ...  // 10 folds
]
```

- T = 8124 タイムステップ、Ny = 3 クラス
- `predictions`: float64、1値 ≈ 22文字（バイナリ float32 なら 4バイト）
- `labels`: one-hot の [1.0, 0.0, 0.0] 形式、文字列では短いがバイナリなら uint8 × T で十分
- `indent=2` によるインデント・改行が全体に加算

### サイズの内訳（1シーケンス・1fold あたり）

| データ | 要素数 | JSON 文字数 | float32 binary |
|--------|--------|------------|----------------|
| predictions | 8124 × 3 = 24,372 floats | ~530 KB | 95 KB |
| labels | 8124 × 3 = 24,372 floats | ~145 KB | 24 KB (uint8) |
| **合計** | | **~675 KB** | **~119 KB** |

→ **JSON 対比、float32+uint8 なら約 5.7x 小さい**（圧縮なし）

### downstream での利用状況

| task | 使用内容 | raw float が必要か |
|------|---------|------------------|
| `eval/judge` | `argmax(predictions, axis=1)` の bincount / `mean(predictions, axis=0)` | mean_score で必要 |
| `eval/dist` | `argmax(predictions, axis=1)` の割合 | 不要（argmax のみで足りる） |
| `eval/dist_node` | confidence, margin, per-node 出力（raw float） | **必要** |

`eval/dist_node` が raw float を要求するため、**predictions の完全削除は不可**。

---

## 選択肢の比較

### 案 A: `indent=None` + gzip JSON

- `json.dump(..., indent=None)` + `gzip.open` で書き込み・読み込み
- データ構造・型は一切変更なし

| 項目 | 内容 |
|------|------|
| 推定削減率 | ~3〜4x → 41GB → **~10〜14GB** |
| 変更範囲 | `save_pred_results` / `is_valid_result_file` + downstream の `open` を `gzip.open` に変更 |
| 後方互換 | 既存ファイルと非互換（拡張子 `.json.gz`） |
| リスク | 低 |

### 案 B: npz (float32) + `labels` 廃止

- 保存形式を `.npz`（NumPy 圧縮バイナリ）に変更
- `predictions` を float64 → float32 に変換
- `labels` を廃止し、代わりに `true_label: int`（多数決による真ラベル）を pred 時に計算して保存
  - `true_label = argmax(bincount(argmax(D, axis=1)))` で確定的に計算可能

| 項目 | 内容 |
|------|------|
| 推定削減率 | ~12〜18x → 41GB → **~2〜4GB** |
| 変更範囲 | `save`/`load` + `predictor.py` + downstream reader 全更新（`eval/judge`, `eval/dist`, `eval/dist_node`） |
| 後方互換 | 既存ファイルと非互換（`.npz`） |
| リスク | 中（float32 精度の確認、reader 更新漏れ） |

**float32 精度について**: ESN 出力の分類用途（argmax, mean, margin 等）では float32 で実用上問題なし。float16 は margin のような微小差演算でリスクあり。

### 案 C: 案 A + 案 B の中間（npz のみ、labels はそのまま保持）

- `labels` を uint8 one-hot として npz に含める（廃止しない）
- `true_label` 計算を downstream 側に委ねる（現状ロジックを維持）

| 項目 | 内容 |
|------|------|
| 推定削減率 | ~8〜12x → 41GB → **~4〜6GB** |
| 変更範囲 | 案 B より小さい（`predictor.py` の変更不要） |
| リスク | 中 |

---

## 意思決定ポイント

1. **削減量 vs 変更コスト**: 案 A（小変更・10GB）か 案 B（大変更・3GB）か
2. **`labels` の廃止可否**: downstream で one-hot labels が今後必要になる可能性があるか
3. **既存 41GB の扱い**: 再生成 / 変換スクリプトで移行 / 新規実行分のみ新フォーマット適用

---

## 計画

### Phase 1: pred_results ロード処理の共通化

- [x] `projects/utils/prediction.py` に `load_pred_results(path) -> list` を追加
- [x] `eval/judge/app.py` の `json.load` を `load_pred_results` に差し替え
- [x] `eval/dist/app.py` の `json.load` を `load_pred_results` に差し替え
- [x] `eval/dist_node/app.py` の `json.load` を `load_pred_results` に差し替え
- [x] `eval/plot/app.py` の `json.load` を `load_pred_results` に差し替え（計画外で発見）

### Phase 2: JSON → npz バイナリ変換によるサイズ削減

**方針**
- データ構造（キーバリュー）は変更しない。eval 側のロジックは一切触らない
- `projects/utils/prediction.py` の `save` / `load` / `is_valid_result_file` の実装のみ変更
- `pred/app.py` のパスから拡張子を除去し、拡張子の管理を `prediction.py` に集約（`_result_path` ヘルパーで `.npz` を付与）
- float64 → float32 に変換して保存（分類用途で精度問題なし）
- labels は保持（廃止しない）

**npz 内部構造**（可変長シーケンス対応のためフラット配列＋長さ配列方式）

| キー | dtype | shape | 内容 |
|------|-------|-------|------|
| `fold_indices` | int32 | [N] | 各サンプルの fold インデックス |
| `ids` | unicode | [N] | 各サンプルの ID |
| `lengths` | int32 | [N] | 各サンプルのタイムステップ数 |
| `predictions_flat` | float32 | [ΣT, Ny] | 全サンプルの predictions を連結 |
| `labels_flat` | float32 | [ΣT, Ny] | 全サンプルの labels を連結 |

load 時は `np.split(predictions_flat, cumsum(lengths)[:-1])` で元の per-sample 配列に復元し、`{"fold_index", "results": [{"id", "predictions", "labels"}]}` 構造を再構築する。

**変更ファイル**
- [x] `projects/utils/prediction.py` — `save` / `load` / `is_valid_result_file` を npz 実装に全面書き換え、`_result_path` ヘルパー追加、`import json` 削除
- [x] `projects/tasks/pred/app.py` — `f"{mode}_results.json"` → `f"{mode}_results"`（拡張子除去のみ）
- [x] `projects/tasks/eval/judge/app.py` — パス・存在チェック更新
- [x] `projects/tasks/eval/dist/app.py` — パス・存在チェック更新
- [x] `projects/tasks/eval/dist_node/app.py` — パス・存在チェック更新
- [x] `projects/tasks/eval/plot/app.py` — パス更新

## 実行ログ

## 結果

Phase 2 完了。`prediction.py` の save/load/is_valid を npz 実装に全面書き換え、呼び出し側 5 ファイルのパス文字列を合わせて更新。ラウンドトリップ検証（fold 数・id・shape・dtype）でも正常動作を確認。既存の `.json` ファイルは新フォーマットと非互換のため、再実行または変換が必要。