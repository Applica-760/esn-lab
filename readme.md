# esn-lab

Echo State Network (ESN) の実験と学習のためのPythonパッケージです。

## 概要

esn-labは、リザバーコンピューティングの一種であるEcho State Networkの実装を提供します。リザバー層の状態ベクトルを利用した時系列データの学習と予測を行うことができます。

## インストール

```bash
pip install -e .
```

## 主要コンポーネント

## 使用例

※詳細な使用例は./projectsに掲載

```python
from esn_lab import ESN, Tikhonov, train

# モデルの初期化
model = ESN(
    N_u=1,          # 入力次元
    N_y=1,          # 出力次元
    N_x=100,        # リザバーノード数
    density=0.1,    # 結合密度
    input_scale=1.0,
    rho=0.9         # スペクトル半径
)

# 最適化器の初期化
optimizer = Tikhonov(N_x=100, N_y=1, beta=1e-6)

# 学習の実行
output_weight = train(model, optimizer, U_list, D_list)

# 出力重みの設定
model.Output.setweight(output_weight)
```

## ライセンス

MIT License

## 開発状況

Development Status: Alpha
