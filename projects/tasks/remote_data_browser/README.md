# リモートの階層を確認してディレクトリを取得するTask

[research-hub Issue #2](https://github.com/Applica-760/research-slide/issues/2) の独立Task。
現在はSSHによる階層一覧の取得まで実装済み。転送処理とフロントは空ファイルとして配置している。

## 接続先と保存先はcfg.yamlで指定する

- `ssh`: ホスト・ユーザー・秘密鍵のパス・ポート・接続維持設定。秘密鍵の内容は書かない。
- `remote_root`: リモートの絶対パス。初期値は確認済みのesn-labのClone先。
- `initial_path`: 最初に開く階層。`remote_root` からの相対パスで、初期値は `outputs`。
- `local_destination`: ローカル保存先。ローカルのesn-labリポジトリルートからの相対パスで、初期値は `outputs/downloads/genkai/`。現在はパスの解決のみ行い、ディレクトリ作成・転送は行わない。
- `output_dir`・`port`: 既存Taskの設定保存先と、ローカルAPIのポート。初期ポートは8001。

## バックエンドの起動と階層取得

リポジトリルートから実行する。

```sh
uv run -m projects.tasks.cli remote_data_browser
```

| GET API | 内容 |
| --- | --- |
| `/api/config` | リモートルート・初期階層・ローカル保存先を返す。SSHの認証設定は返さない。 |
| `/api/directories` | 初期階層の直下を取得する。 |
| `/api/directories?path=outputs/experiments` | 指定した相対パスの直下を取得する。 |

一覧は `path`・`parent`・`entries` を返す。各項目は `name`・`path`・`type` を持ち、typeは `directory`・`file`・`symlink`・`other`。全階層を再帰取得せず、選んだ階層だけを読む。ルートではparentはnullとなる。

リモートのPython 3をSSH経由で実行し、JSON形式で一覧を受け取る。リモート側にファイルは設置しない。絶対パス・親への遡行・リンク経由でのリモートルート外へのアクセスは拒否する。

エラーは `error: {code, message}`。対象なしは404、権限不足は403、不正パスは422、SSHタイムアウトは504、接続・応答の異常は502で返す。

## 転送とフロントは未確定のため保留する

`transfer.py` と `projects/app/remote_data_browser/` のHTML・CSS・JavaScriptは空。取得ボタンや取得APIもまだ設けない。既存ファイルの上書き・リンクの扱い・取得先での階層の保持方法を決めてから実装する。
