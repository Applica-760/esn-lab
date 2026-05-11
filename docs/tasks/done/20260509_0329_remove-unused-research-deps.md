## 目的・前提・方針案

### 削除対象パッケージと根拠

| パッケージ | 使用箇所 | 削除根拠 |
|---|---|---|
| `japanize-matplotlib` | `projects/tools/data_analysis/analyze_behavior_shift.py` のみ | AI生成の単発ツール。esn-lab コアに不要 |
| `simple-term-menu` | `projects/tools/pred_plot/prediction_cli.py` のみ | AI生成の単発ツール。esn-lab コアに不要。代替実装が必要 |
| `openpyxl` | `projects/tools/data_prep/get_300seqs.py` のみ | データセット生成用途で研究ドメイン固有。esn-lab に不要 |
| `pillow`（PIL） | `projects/tools/data_prep/get_300seqs.py` のみ | `openpyxl` と同ファイル・同理由。依存宣言も漏れていた |

- 上記はすべて `[dependency-groups].research` から削除し、使用箇所のスクリプトも修正・削除対象
- `simple-term-menu` のみ代替実装（`input()` 等）が必要

### ドキュメント・コメントの uv 対応
- conda 環境（`conda activate research` 等）を前提とした実行例・コメントが残存している可能性がある
- `uv sync` / `uv run` / `uv sync --group research` への書き換えが必要
- 対象箇所の網羅は次セッションで行う

## 計画

## Phase 1: pyproject.toml から不要パッケージを削除
- [x] `japanize-matplotlib`, `openpyxl`, `pillow`, `simple-term-menu` を `[dependency-groups].research` から削除（pillow は元々未宣言）
- [x] `uv sync --group research` で依存解決を確認

## Phase 2: `analyze_behavior_shift.py` — japanize-matplotlib 除去
- [x] `import japanize_matplotlib` 行と「日本語フォント対応」コメントを削除（フォントは DejaVu Sans 設定済み、ラベルはすべて英語なので他変更不要）

## Phase 3: `prediction_cli.py` — simple-term-menu を input() に置換
- [x] `from simple_term_menu import TerminalMenu` を削除
- [x] `menu_select_one()`: 番号付きリスト表示 + `input()` で1択選択に置換
- [x] `menu_select_multiple()`: 番号付きリスト表示 + スペース区切り `input()` で複数選択に置換

## 実行ログ

## 結果

- `pyproject.toml` の `[dependency-groups].research` から3パッケージ削除（`japanize-matplotlib`, `openpyxl`, `simple-term-menu`）
- `analyze_behavior_shift.py` の `import japanize_matplotlib` を削除
- `prediction_cli.py` の `TerminalMenu` を `input()` ベースの番号選択に置換
