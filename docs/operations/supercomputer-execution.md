# スパコン実行

研究全体の方針は `research-hub:docs/operations/research-workflow.md` を参照する。

## 共通手順

1. `esn-lab` で実装・検証・commit/pushを行う。
2. 実験ごとの実行指示書を作成する。
3. スパコンで、指示書のコマンドを実行してジョブを投入する。
4. 結果、実行ログ、`config.lock.yaml`、実行commit IDを取得し、Taskへ記録する。

## 実験ごとの実行指示書

実行前に `docs/operations/runs/YYYYMMDD_<experiment>.md` を作成する。
指示書はスパコン上でそのまま実行できるコマンドだけを記載し、未置換のプレースホルダを残さない。

最低限、次を記載する。

- 対応するTask IDとpush済みcommit ID
- スパコン上の作業ディレクトリと編集対象のジョブ `.sh` ファイル
- リポジトリ更新、ジョブスクリプト編集、投入、状態確認のコマンド
- 使用する設定ファイル、出力先、結果取得コマンド

実行後は、実際に使用したcommit ID・設定・結果パスをTaskのPhase logに残す。
