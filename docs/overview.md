# ESN-Lab プロジェクト概要

## リポジトリ構成

```
esn-lab/
├── src/esn_lab/       # ESN コアパッケージ（汎用）
├── projects/          # 研究固有の実験コード
│   ├── tasks/         # タスクごとのアプリと設定
│   ├── utils/         # プロジェクト共通ユーティリティ
│   └── tools/         # データ準備・分析ツール
├── dataset/           # 入力データ（10-fold 分割済み）
├── outputs/           # 実験結果の出力先
└── pyproject.toml
```

---

## src/esn_lab — ESN コアパッケージ

`esn_lab` は Echo State Network (ESN, リザバーコンピューティングの一種) の **純粋な計算機能** を提供するパッケージです。研究ドメインに依存しない汎用ライブラリとして設計されており、`pip install -e .` でローカルインストールして使用します。

### パッケージ内部構造

```
src/esn_lab/
├── model/
│   └── esn.py         # ESN モデル定義（Input / Reservoir / Output）
├── optim/
│   └── optim.py       # 最適化（Tikhonov / Ridge 回帰）
├── pipeline/
│   ├── train/
│   │   └── trainer.py # 1回分の学習ループ
│   └── pred/
│       └── predictor.py # 1回分の推論
├── runner/
│   ├── train/
│   │   └── tenfold.py # 10-fold 学習の実行管理
│   └── pred/
│       └── tenfold.py # 10-fold 推論の実行管理
└── utils/
    ├── fold_splitter.py  # fold ごとのデータ分割
    └── activate_func.py  # 活性化関数
```

### Pipeline と Runner の役割分担

`esn_lab` の実行層は **pipeline** と **runner** の 2 階層に分かれています。

| 層 | モジュール | 役割 |
|----|-----------|------|
| **Pipeline** | `pipeline/train/trainer.py` | 1 つのデータセット・1 組のパラメータに対して ESN を学習し重みを返す |
| **Pipeline** | `pipeline/pred/predictor.py` | 学習済み重みを使い 1 データセットの予測値・ラベルを返す |
| **Runner** | `runner/train/tenfold.py` | 10-fold 全てに対して Trainer を呼び出し、並列実行を管理する |
| **Runner** | `runner/pred/tenfold.py` | 10-fold 全てに対して Predictor を呼び出し、並列実行を管理する |

Pipeline は「1 fold の処理」という最小単位を担い、Runner は「10-fold 全体の並列実行・結果集約」を担います。これにより各層を独立してテスト・再利用できます。

### パッケージの使用例

```python
from esn_lab.model.esn import ESN
from esn_lab.pipeline.train.trainer import train
from esn_lab.pipeline.pred.predictor import predict
from esn_lab.runner.train.tenfold import TenfoldTrainRunner

# モデル構築
esn = ESN(Nu=256, Nx=700, Ny=3, density=0.5, rho=0.9, input_scale=0.001)

# 単一 fold の学習
weights = train(esn, train_data)

# 10-fold 並列学習
runner = TenfoldTrainRunner(esn_config, tenfold_data, workers=12)
all_weights = runner.run()
```

---

## projects/ — 研究固有の実験コード

`projects/` にはこの研究ドメイン特有の知識・ロジックが集約されています。データフォーマット、評価指標、実験設定など、`esn_lab` パッケージに含めるべきでない研究固有の要素を管理します。

### tasks/ — タスクごとのアプリと設定

各タスクは `app.py`（実行ロジック）と `cfg.yaml`（実験パラメータ）のペアで構成されています。

```
projects/tasks/
├── cli.py             # タスク実行エントリポイント
├── train/
│   ├── app.py         # グリッドサーチ学習
│   └── cfg.yaml       # Nx, density, rho 等のパラメータ定義
├── pred/
│   ├── app.py         # 全パラメータ×fold の推論実行
│   └── cfg.yaml
└── eval/
    ├── judge/         # 多数決による予測→判定変換
    ├── dist/          # 予測信頼度分布の分析
    ├── dist_node/     # ノードレベルの信頼度分析
    ├── metrics/       # 精度・適合率・再現率・F1
    └── plot/          # 予測結果の可視化
```

**実行方法:**
```bash
python -m projects.tasks.cli train
python -m projects.tasks.cli pred
python -m projects.tasks.cli eval.judge
python -m projects.tasks.cli eval.metrics
```

`cfg.yaml` でグリッドサーチのパラメータ範囲や並列ワーカー数を管理し、`app.py` が `esn_lab` の Runner を呼び出して実際の計算を行います。

### utils/ — プロジェクト共通ユーティリティ

`tasks/` の各アプリから共通で使う汎用関数群です。

```
projects/utils/
├── app_init.py        # 設定読込・データローダー・パラメータグリッド生成
├── weights.py         # 重みファイルの保存・読込・パス管理
├── prediction.py      # 予測結果の JSON シリアライズ
└── eval/
    ├── judgment.py    # 多数決による判定ロジック
    ├── dist.py        # 分布統計とヒストグラム描画
    ├── filter.py      # 結果フィルタリング
    ├── metrics.py     # 分類指標の計算
    └── plot_prediction.py  # 予測可視化
```

主な機能:
- `load_config()` / `setup_task_environment()`: YAML 設定の読込と自動解決
- `tenfold_data_loader()`: 10-fold NPZ データの一括読込
- `build_param_grid()`: パラメータのデカルト積からグリッドを生成
- `save_single_weight()` / `load_tenfold_weights()`: 重みファイルの I/O とバリデーション

### tools/ — データ準備・分析ツール

実験の前処理や、実験とは独立したデータ分析を行うスクリプト群です。

```
projects/tools/
├── data_prep/
│   └── create_10fold_divisions.py  # 10-fold 分割 NPZ の生成
└── data_analysis/
    ├── create_data_table.py         # データセット集計表の作成
    ├── search_data_duplication.py   # 重複サンプルの検出
    └── analyze_behavior_shift.py    # 時系列の経時変化分析
```

---

## エンドツーエンドのワークフロー

```
[生データ (CSV)]
      ↓  tools/data_prep/create_10fold_divisions.py
[dataset/ (10-fold NPZ)]
      ↓  python -m projects.tasks.cli train
[outputs/experiments/task/Nx700_dens0.5_inscl.../fold{0-9}.npz]  ← 重みファイル
      ↓  python -m projects.tasks.cli pred
[outputs/experiments/pred/.../results.json]  ← 予測結果
      ↓  python -m projects.tasks.cli eval.judge
[judgment_results.csv]  ← 多数決判定
      ↓  python -m projects.tasks.cli eval.metrics / eval.dist / eval.plot
[精度・分布・可視化]
```

### 実験パラメータ管理

学習タスクはグリッドサーチで複数のハイパーパラメータ組み合わせを探索します。結果は `Nx700_dens0.5_inscl0.0001_rho0.9` のようなパラメータ文字列をディレクトリ名として出力するため、実験条件と成果物が 1 対 1 で対応します。すでに重みファイルが存在するパラメータ組み合わせはスキップされるため、実験を安全に再開できます。

---

## 設計思想

| 関心事 | 配置場所 |
|--------|---------|
| ESN の数理的な計算 | `src/esn_lab/` |
| 実験の設定・実行管理 | `projects/tasks/` |
| I/O・評価の共通処理 | `projects/utils/` |
| データ整備・独立分析 | `projects/tools/` |

`esn_lab` は研究プロジェクトから独立したライブラリとして再利用可能に保ち、研究固有のドメイン知識はすべて `projects/` 以下に閉じ込める構成になっています。
