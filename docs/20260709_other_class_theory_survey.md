# 「その他」を扱う理論と、次に狙うプロット候補

調査日: 2026-07-09

## 背景・前提

教員の6/6宿題は2本立て。

- **① 採食・反芻の2クラス実験** → 済み（`only_fr_*` 画像 ＋ `research-slide/pages/20260606.md`）。
  「その他」を抜くと採食 test 0.76→0.92、反芻 test 0.51→0.75 と分離が改善。
- **② 「その他を扱う機械学習理論」の調査** → 本ドキュメント。

教員コメントの核心（`research-slide/docs/20260606.md`）:

> リザバーの出力は高次元空間から一次元ベクトルへの射影。採食・反芻の出力ベクトルは**直交**して取れているが、
> **その他はそれに沿わず偏在**している。だから (1,0,0) のように直交させる学習は難しいのでは。

→ これは学術的に **Open Set Recognition (OSR) / Out-of-Distribution (OOD) 検出** の問題設定にほぼそのまま対応する。
「その他」＝どのクラスにも属さない "none of the above" をどう扱うか、という分野が確立している。

## 理論マッピング（教員の直感 → 既存理論）

| 教員の言葉 | 対応する理論 | 一言 |
|---|---|---|
| 「その他は1ラベルとして扱うのが不適切」 | **Open Set Recognition / 棄却付き分類 (reject option)** | K+1クラスの最後を "none of the above" にする定式化。まさにこれ |
| 「その他は軸に沿わず偏在」 | **距離ベースOOD（Mahalanobis距離）** | 各クラスの平均・共分散を推定し、どのクラス中心からも遠いサンプルを「その他」とみなす。「偏在」の定量化に直結 |
| 「棄却戦略はよくない」(0605 FB) | **Energy / Max-Logit スコア** | softmax前の**生値(logit)**に情報が多い。マージンよりenergy scoreの方が分離が良いとの報告多数。既にlogit生値を保持しているのが強み |
| 「1位2位の差を点数に」(0605 FB) | **LogitGap / margin系スコア** | 最大logitと残りのgapで既知/未知を分離。教員の提案と学術手法が一致 |

**要点：教員が直感で言っていることは、OSR/OODの標準的な道具立てとほぼ一対一で対応する。**
②の報告は、この対応表を見せるだけで成立する。

## 筋が良さそうなプロット候補（低コスト順）

既存の解析コード（`projects/tasks/analysis/pred/app.py`）に
`analyze_score_margin`, `analyze_score_trajectory`, `analyze_argmax_heatmap`,
さらに教員FB反映済みの**1位固定3×3セル版**（`plot_score_margin_3x3` 等）が既にある。それを踏まえた候補：

### ◎ 候補A：logitマージン分布の「クラス別ヒストグラム」（最推し）
- 各サンプルの `max_logit − 2nd_logit` を、**真クラス別**（採食／反芻／その他）に重ねてヒストグラム化。
- 狙い：**その他だけマージンが小さい方に寄る**ことを見せられれば、
  「その他＝どのクラスにも決めきれない＝OSR的に棄却対象」という教員の直感を**1枚で定量化**できる。
- コスト：**最小**。`analyze_score_margin` が既にあり、集計軸を「真クラス別ヒストグラム」に変えるだけ。

### ◎ 候補B：Energyスコア（logitの生値）のクラス別分布
- softmax前のlogitから energy score を計算し、採食・反芻 vs その他で分布を比較。
- 狙い：文献で「softmaxより energy の方が既知/未知の分離が良い」と繰り返し報告。
  0605で教員が言った「生値を柔軟に使う」方向と一致。
- コスト：小〜中。energyの式を1つ足すだけ。

### ○ 候補C：出力ベクトルのPCA 2D散布図（"偏在"の可視化）
- リザバー出力（3次元スコア）を2次元に落とし、採食・反芻・その他を色分けして散布。
- 狙い：教員の「採食・反芻は直交、その他は偏在」という**幾何的主張をそのまま絵にする**。最もインパクトが大きい。
- コスト：中。「その他が両軸に沿わず散らばる」が絵で出れば、報告の主役になれる。

### △ 候補D：Mahalanobis距離のクラス別分布
- 採食・反芻それぞれの中心からの距離を計算し、その他がどちらからも遠いことを示す。
- 狙い：「偏在」の最も厳密な定量化。ただし共分散推定など実装コストは上の3つより高い。

## 明日の報告への当て方（提案）

**「①2クラス実験は完了、②理論調査で "その他=OSR/OOD問題" と位置づけられた。
次は候補A（マージンのクラス別分布）で、その他が本当に "決めきれない" クラスなのかを1枚で検証する」**

この筋なら②の宿題に答えつつ、次アクションまで示せる。
実装コストは **候補A → B → C** の順。A だけなら既存コードの小改修で1枚出せる見込み。
C は絵のインパクトが最大なので「今夜A／週末C」の二段構えが現実的。

## 参考文献

- [Recent Advances in Open Set Recognition: A Survey](https://arxiv.org/pdf/1811.08581)
- [Generalized Out-of-Distribution Detection: A Survey](https://arxiv.org/pdf/2110.11334)
- [A Unified Survey on Anomaly, Novelty, Open-Set, and Out-of-Distribution Detection](https://arxiv.org/pdf/2110.14051)
- [Energy-based Out-of-distribution Detection](https://arxiv.org/pdf/2010.03759)
- [Revisiting Logit Distributions for Reliable OOD Detection](https://arxiv.org/html/2510.20134v1)
- [A Simple Unified Framework for Detecting OOD (Mahalanobis)](https://proceedings.neurips.cc/paper/2018/file/abdeb6f575ac5c6676b747bca8d09cc2-Paper.pdf)
- [Dissecting Mahalanobis: Feature Geometry and OOD Detection](https://arxiv.org/html/2510.15202v1)
- [OOD Detection Based on Distance Metric Learning](https://manuscriptlink-society-file.s3-ap-northeast-1.amazonaws.com/kism/conference/sma2020/presentation/SMA-2020_paper_60.pdf)
