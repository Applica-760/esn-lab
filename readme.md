# pyesn: Echo State Network 実験用Pythonパッケージ

## 概要

`pyesn` は、Echo State Network (ESN) を用いた研究・実験を効率的に行うために開発したPythonパッケージです。

私が大学の研究で使用するために作成したもので、設定ファイルベースで柔軟な実験管理を行えるように設計しています。<br>
ある程度一般的な実装もありますが、特定ドメインにフィットさせた実装もあります。<br>
また、これまでの開発スキルを活かしたポートフォリオとしての側面も兼ねています。<br>
<br>
現在は研究に必要な基本機能を実装した段階ですが、今後も継続的に開発を行い、より汎用性の高いツールへと成長させていく予定です。


## ✨ 特徴

  * **設定ファイルに基づく柔軟な実験管理**: `YAML`形式の設定ファイルを用いて、モデルのパラメータや学習データを管理します。
  * **再現性の確保**: 実行時の設定を自動で保存し、実験の再現性を高めます。
  * **豊富な実行モード**: 単一データでの学習・予測から、複数データの一括処理、10-fold交差検証によるハイパーパラメータ探索まで、幅広い実験シナリオへの対応を目指しています。
  * **コマンドラインからの簡単な操作**: コマンドラインインターフェース（CLI）を通じて、学習、予測、評価の各プロセスを直感的に実行できます。

## 📦 インストール

```bash
pip install -e .
```

## 🚀 使い方

### 1\. 設定ファイルの初期化

まず、以下のコマンドを実行して、カレントディレクトリに `configs` ディレクトリを生成します。

```bash
pyesn init
```

これにより、各種設定ファイルのテンプレートがコピーされます。

### 2\. 設定ファイルの編集

`configs/base.yaml` に、プロジェクト全体で共通の基本設定を記述します。

```yaml:configs/base.yaml
project: "esn-research"
seeds: [2024, 706, 4410, 5385, 1029, 1219, 8380, 8931, 5963, 19800]
num_of_classes: 3

data:
    type: "complement"

model:
    name: "esn"
    Nu: 256
    Nx: 10
    Ny: 3
    density: 0.5
    input_scale: 0.01
    rho: 0.9
    optimizer: "tikhonov"
```

次に、実行したいモードに合わせて、`configs/train/single.yaml` などの各設定ファイルを編集します。

```yaml:configs/train/single.yaml
id: "sample_001"
path: "/path/to/your/data/sample_001.jpg"
class_id: 0
```

### 3\. 学習の実行

単一のデータで学習を行う場合は、以下のコマンドを実行します。

```bash
pyesn train single
```

学習が完了すると、実行結果（重みファイルやログ）が `artifacts/runs/{実行日時}_{モード}-{バリアント}/` ディレクトリに保存されます。

### 4\. 予測の実行

学習済みの重みファイルを使って予測を行うには、まず `configs/predict/single.yaml` を編集し、使用する重みファイルのパスを指定します。

```yaml:configs/predict/single.yaml
id: "test_sample_001"
path: "/path/to/your/test_data/sample_001.jpg"
class_id: 0
weight: "artifacts/runs/your_training_run_id/output_weight" # 学習済み重みへのパス
```

その後、以下のコマンドを実行します。

```bash
pyesn predict single
```

## 🛠️ コマンドラインインターフェース

`pyesn` は、以下のモードとバリアントをサポートしています。

| モード     | バリアント         | 説明                                                     |
| :--------- | :----------------- | :------------------------------------------------------- |
| `train`    | `single`           | 単一のデータで学習します。                               |
|            | `batch`            | 複数のデータを一括で学習します。                         |
|            | `tenfold_search`   | 10-fold交差検証を行い、ハイパーパラメータを探索します。 |
| `predict`  | `single`           | 単一のデータで予測を行います。                           |
|            | `batch`            | 複数のデータを一括で予測します。                         |
| `evaluate` | `run`              | 予測結果を評価します。                                   |


## 今後の展望 (ロードマップ)

このプロジェクトはまだ発展途上です。今後は以下のような機能拡張を計画しています。

  * **評価機能の強化**: 混同行列の生成や、より詳細な評価指標の可視化。
  * **可視化ツールの統合**: 学習過程やリザバーの内部状態を可視化する機能。


## 依存パッケージ

  * numpy
  * pandas
  * opencv-python
  * PyYAML
  * omegaconf
  * networkx


