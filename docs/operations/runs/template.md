# 実行指示書: <experiment>

このテンプレートから `YYYYMMDD_HHMM_<experiment>.md` を作成する。`<...>` は実行前に具体値へ置換し、完成した指示書には残さない。

## 対応関係

- Hub Task: `research-hub:TASK-<NNN>`
- 対象リポジトリ: `<repository name and remote URL>`
- checkoutするworktree: `<worktree name>`
- スパコン上のworktreeパス: `<absolute worktree path>`
- branch: `<branch>`
- 実行commit: `<full pushed commit ID>`

## 直接実行するコマンド

ジョブ `.sh` は作成しない。以下には、ユーザーがスパコンへ直接転記するコマンドを記載する。

```sh
git -C <repository path> fetch origin <branch>
git -C <repository path> worktree add --detach <absolute worktree path> <full pushed commit ID>
cd <absolute worktree path>
uv run -m <module command>
```

## 設定と成果物

- 使用する設定ファイル: `<path>`
- 入力データセットの識別情報: `<identifier>`
- 出力先: `<path>`
- 実行後の `config.lock.yaml`: `<path>`
- 実行ログ: `<path>`
- Hubへ持ち込む図・集計値: `<path>`

## 実行後の記録

実際に使用したcommit ID、設定、結果パスを、対応するHub TaskのPhase logへ記録する。
