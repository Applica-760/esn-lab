# スパコン実行

研究全体の方針は `research-hub:docs/operations/research-workflow.md` を参照する。

## 共通手順

1. `esn-lab` で実装・検証・commit/pushを行う。
2. 実験ごとの実行指示書を作成し、対応する `research-hub` Taskから参照する。
3. スパコンで、指示書のコマンドを実行してジョブを投入する。
4. 結果、実行ログ、`config.lock.yaml`、実行commit IDを取得し、Hub Taskへ記録する。

## 実験ごとの実行指示書

実行前に [template.md](runs/template.md) から `docs/operations/runs/YYYYMMDD_HHMM_<experiment>.md` を作成する。
指示書には、ユーザーがスパコンへ直接転記するコマンドを記載する。ジョブ `.sh` は作成しない。実際の指示書には未置換のプレースホルダを残さない。

最低限、次を記載する。

- 対応する `research-hub` Task IDとpush済みcommit ID
- checkout対象のリポジトリ、worktree名・branch・commit、スパコン上のworktreeパス
- リポジトリ更新、worktreeのcheckout、実行、状態確認のコマンド
- 使用する設定ファイル、出力先、結果取得コマンド

実行後は、実際に使用したcommit ID・設定・結果パスを対応するHub TaskのPhase logに残す。
