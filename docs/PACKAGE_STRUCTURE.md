# パッケージ構造とモジュールの説明

このドキュメントでは、`esn-lab`パッケージの内部構造と各モジュールの役割について説明します。

## 📁 ディレクトリ構成

```
src/esn_lab/
├── cli.py                    # CLIエントリーポイント
├── model/                    # ESNモデル定義
│   ├── __init__.py
│   ├── builder.py           # モデルビルダー
│   └── esn.py              # ESNモデルクラス
├── optim/                    # 最適化関数
│   ├── __init__.py
│   ├── tikhonov.py         # Tikhonov正則化
│   └── pseudo_inverse.py   # 疑似逆行列法
├── pipeline/                 # データパイプライン
│   ├── __init__.py
│   ├── data/               # データローダー
│   │   ├── __init__.py
│   │   ├── base.py         # 抽象基底クラス
│   │   ├── csv_loader.py   # CSV形式ローダー
│   │   ├── npy_loader.py   # NPY形式ローダー
│   │   └── factory.py      # DataLoader Factory
│   ├── train.py            # 訓練パイプライン
│   ├── predict.py          # 推論パイプライン
│   └── evaluate.py         # 評価パイプライン
├── runner/                   # 実行制御
│   ├── __init__.py
│   ├── train.py            # 訓練ランナー
│   ├── predict.py          # 推論ランナー
│   └── evaluate.py         # 評価ランナー
├── setup/                    # 初期設定
│   ├── __init__.py
│   └── config.py           # 設定管理
└── utils/                    # ユーティリティ
    ├── __init__.py
    ├── logger.py           # ロギング
    ├── path.py             # パス管理
    └── visualization.py    # 可視化
```

## 🔧 主要モジュールの詳細

### 1. cli.py - コマンドラインインターフェース

パッケージのエントリーポイント。ユーザーからのコマンドを解釈し、適切なランナーを呼び出します。

**主な機能:**
- サブコマンドの解析（train, predict, evaluate）
- バリアントの選択（single, batch, tenfold等）
- 設定ファイルの読み込み
- ランナーへのディスパッチ

### 2. model/ - Echo State Networkモデル

ESNの実装と構築を担当します。

#### model/esn.py
- ESNのコアクラス定義
- リザバー状態の更新
- 出力重み行列の学習と推論

#### model/builder.py
- 設定ファイルからのモデル構築
- ハイパーパラメータの適用
- 重み行列の初期化

**主要な実装:**
```python
class ESN:
    def fit(X, y):        # 訓練
    def predict(X):       # 推論
    def update_state(u):  # リザバー状態更新
```

### 3. optim/ - 最適化関数

出力重み行列Woutの最適化アルゴリズムを提供します。

#### optim/tikhonov.py
- Tikhonov正則化（Ridge回帰）
- 過学習の抑制
- デフォルトの最適化手法

#### optim/pseudo_inverse.py
- Moore-Penrose疑似逆行列
- 高速な学習
- 正則化なし

### 4. pipeline/ - データパイプライン

データの読み込みから結果の保存までの一連の流れを管理します。

#### pipeline/data/ - データローダー

**設計パターン:** Strategy + Factory

- **base.py**: `BaseDataLoader`抽象基底クラス
- **csv_loader.py**: CSV+画像形式のデータローダー
- **npy_loader.py**: NPY前処理済みデータローダー
- **factory.py**: 設定に基づいてローダーを自動選択

**利点:**
- データソースの抽象化
- 新しい形式の追加が容易
- メモリ効率的なイテレータベース

#### pipeline/train.py
訓練の一連の流れを実装:
1. データの読み込み
2. モデルの構築
3. ESNの訓練
4. 重みの保存

#### pipeline/predict.py
推論の一連の流れを実装:
1. 保存済み重みの読み込み
2. テストデータの読み込み
3. 推論の実行
4. 結果の保存

#### pipeline/evaluate.py
評価の一連の流れを実装:
1. 推論結果の読み込み
2. 評価指標の計算
3. 混同行列の生成
4. レポートの出力

### 5. runner/ - 実行制御

各モード（single, batch, tenfold）の実行ロジックを管理します。

#### runner/train.py
- **single**: 単一データの訓練
- **batch**: バッチデータの訓練
- **tenfold**: 10-fold交差検証

#### runner/predict.py
- **single**: 単一データの推論
- **batch**: バッチデータの推論

#### runner/evaluate.py
- **run**: 単一実行の評価
- **summary**: 10-fold結果の集約
- **tenfold**: 10-fold推論と評価

### 6. setup/ - 初期設定

#### setup/config.py
- 設定ファイルの読み込み
- デフォルト値の適用
- 設定のバリデーション
- OmegaConfによる階層的設定管理

### 7. utils/ - ユーティリティ

#### utils/logger.py
- 実行ログの記録
- エラーハンドリング
- デバッグ情報の出力

#### utils/path.py
- パス解決
- ディレクトリ作成
- ファイル名生成

#### utils/visualization.py
- 混同行列の描画
- 学習曲線のプロット
- 結果の可視化

## 🔄 データフロー

### 訓練時のデータフロー

```
CLI (cli.py)
  ↓
Runner (runner/train.py)
  ↓
Pipeline (pipeline/train.py)
  ↓
DataLoader (pipeline/data/*.py) → Model Builder (model/builder.py)
  ↓                                       ↓
ESN Model (model/esn.py) ←───────────────┘
  ↓
Optimizer (optim/*.py)
  ↓
Save Weights (outputs/)
```

### 推論時のデータフロー

```
CLI (cli.py)
  ↓
Runner (runner/predict.py)
  ↓
Pipeline (pipeline/predict.py)
  ↓
Load Weights + DataLoader → ESN Model (model/esn.py)
  ↓
Predictions
  ↓
Save Results (outputs/)
```

## 🎯 設計の特徴

### 1. 責任の分離
- **CLI**: コマンド解析のみ
- **Runner**: 実行フロー制御
- **Pipeline**: ビジネスロジック
- **Model**: ESNの実装

### 2. 拡張性
- 新しいデータローダーの追加が容易
- 新しい最適化手法の追加が容易
- 新しい評価指標の追加が容易

### 3. テスタビリティ
- 各モジュールが独立してテスト可能
- モックデータの注入が容易
- ユニットテストの実装が容易

### 4. 保守性
- 明確なディレクトリ構造
- 責任の明確な分離
- ドキュメント化されたインターフェース

## 📚 関連ドキュメント

- [データローダー実装の詳細](./DATALOADER_IMPLEMENTATION.md)
- [開発メモ](./memo.md)
- [コマンドリファレンス](./commands.md)
- [英語版README](./readme-en.md)
