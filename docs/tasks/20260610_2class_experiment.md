## 目的・前提・方針案

- 目的: 採食・反芻・その他の3クラス分類を採食・反芻の2クラスに限定し、結果の変化を観察する
- 前提: 既存3クラスNPZデータセットが存在する。3クラスと同一サンプルで比較するため、乱数の種を変えず既存データを変換して2クラスデータセットを作成する
- 内部ラベル順序: 0=other, 1=foraging, 2=rumination（`class_order: [1,2,0]` で表示上はforaging/rumination/other順に並び替え）
- 方針: 既存NPZから other(index=0) サンプルを除外し label を `[:, 1:]` にスライスして新NPZを作成。Train→Pred→Eval を2クラス設定で実行

## 計画

### Phase 1: cli.py の --config 対応
- [x] `setup_task_environment()` に optional `config_name` 引数を追加（未指定時は既存の `cfg.yaml` フォールバック維持）
- [x] `cli.py` に `--config` 引数を追加し `setup_task_environment(config_name=args.config)` に渡す

### Phase 2: 2クラスNPZ変換スクリプト作成
- [x] `projects/tools/data_prep/convert_to_2class.py` を作成
  - `is_other` / `remap_label` / `convert_npz` / `convert_dataset` / `verify_output` / `main`
  - src==dst ガードあり（オリジナル保護）
  - 変換後に shape・ラベル値の全件検証を実行

### Phase 3: 2クラス用データセット生成
- [ ] `convert_to_2class.py` を実行して 2クラスNPZを生成（既存3クラスデータセットから変換）

### Phase 4: 2クラス用 cfg ファイル作成
- [x] `projects/tasks/train/cfg_2class.yaml`（`Ny: 2`, `data_source_base_dir: dataset/10fold_npy_2class`）
- [x] `projects/tasks/pred/cfg_2class.yaml`（`weight_dir` を train output に合わせ、`Ny: 2`）
- [x] `projects/tasks/eval/judge/cfg_2class.yaml`（`pred_result_dir` / `output_dir` を 2class パスに）
- [x] `projects/tasks/eval/metrics/cfg_2class.yaml`（`class_names: ["foraging", "rumination"]`, `class_order: [0, 1]`）

### Phase 5: 2クラス実験実行・確認
- [ ] Train: `uv run -m projects.tasks.cli train --config projects/tasks/train/cfg_2class.yaml`
- [ ] Pred: `uv run -m projects.tasks.cli pred --config projects/tasks/pred/cfg_2class.yaml`
- [ ] Eval (judge): `uv run -m projects.tasks.cli eval.judge --config projects/tasks/eval/judge/cfg_2class.yaml`
- [ ] Eval (metrics): `uv run -m projects.tasks.cli eval.metrics --config projects/tasks/eval/metrics/cfg_2class.yaml`
- [ ] 3クラス結果と比較確認

## 実行ログ

## 結果
