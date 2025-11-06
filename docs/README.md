# esn-lab ドキュメント

このディレクトリには、`esn-lab`パッケージの詳細なドキュメントが含まれています。

## 📚 ドキュメント一覧

### ユーザー向けドキュメント

- **[readme-en.md](./readme-en.md)** - 英語版README
  - パッケージの概要と基本的な使い方
  - インストール方法
  - クイックスタートガイド

### 開発者向けドキュメント

- **[PACKAGE_STRUCTURE.md](./PACKAGE_STRUCTURE.md)** - パッケージ構造の説明
  - ディレクトリ構成
  - 各モジュールの役割と責任
  - データフローの説明
  - 設計思想とアーキテクチャ

- **[CONFIGURATION.md](./CONFIGURATION.md)** - 設定ファイルリファレンス
  - 全設定ファイルの詳細説明
  - パラメータの説明と推奨値
  - ベストプラクティス
  - 高度な設定方法

- **[DATALOADER_IMPLEMENTATION.md](./DATALOADER_IMPLEMENTATION.md)** - データローダー実装ドキュメント
  - データローダーの抽象化設計
  - NPY形式サポートの実装詳細
  - パフォーマンス改善の説明
  - 前処理スクリプトの使い方

### 開発メモ（非公開）

以下のファイルは`.gitignore`で除外されており、個人的な開発メモです:

- **commands.md** - よく使うコマンドや環境構築メモ
- **memo.md** - 開発中のメモやTODO

## 🗂️ ドキュメントの構成

```
docs/
├── README.md                           # このファイル
├── readme-en.md                        # 英語版README
├── PACKAGE_STRUCTURE.md               # パッケージ構造
├── CONFIGURATION.md                   # 設定リファレンス
├── DATALOADER_IMPLEMENTATION.md       # データローダー実装
├── commands.md                        # コマンドメモ（非公開）
└── memo.md                            # 開発メモ（非公開）
```

## 📖 読む順番（推奨）

### 初めて使う場合
1. [../readme.md](../readme.md) - メインREADME（基本的な使い方）
2. [CONFIGURATION.md](./CONFIGURATION.md) - 設定方法の理解
3. 実際に動かしてみる

### 開発に参加する場合
1. [../readme.md](../readme.md) - メインREADME
2. [PACKAGE_STRUCTURE.md](./PACKAGE_STRUCTURE.md) - アーキテクチャの理解
3. [DATALOADER_IMPLEMENTATION.md](./DATALOADER_IMPLEMENTATION.md) - 最新実装の確認
4. [CONFIGURATION.md](./CONFIGURATION.md) - 設定の詳細

### パフォーマンス改善したい場合
1. [DATALOADER_IMPLEMENTATION.md](./DATALOADER_IMPLEMENTATION.md) - NPY形式の使用
2. [CONFIGURATION.md](./CONFIGURATION.md) - `data_source`設定の変更

## 🔄 ドキュメントの更新

ドキュメントは常に最新の実装に合わせて更新してください。

### 更新が必要なタイミング
- 新機能の追加
- APIの変更
- 設定項目の追加・変更
- アーキテクチャの変更

### 更新手順
1. 該当するドキュメントファイルを編集
2. コミットメッセージに`[docs]`プレフィックスを付ける
3. プルリクエストで変更をレビュー

## 📝 ドキュメント作成ガイドライン

- **明確さ**: 技術的に正確で、分かりやすい説明を心がける
- **例示**: 具体的なコード例や設定例を含める
- **最新性**: 実装と常に同期させる
- **構造化**: 見出しを適切に使い、読みやすくする
- **日英両対応**: 可能な限り英語版も提供

## 🔗 関連リンク

- [メインREADME](../readme.md)
- [GitHub Repository](https://github.com/Applica-760/esn-lab)
- [PyPI Package](https://pypi.org/project/esn-lab/)

## 📧 サポート

ドキュメントに関する質問や改善提案は、GitHubのIssueでお願いします。
