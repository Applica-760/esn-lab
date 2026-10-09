import json
import shlex
import subprocess
from pathlib import Path, PurePosixPath

# リモートにファイルを置かず、名前に空白・改行があってもJSONで一覧を受け取る。
LIST_DIRECTORY_SCRIPT = r"""
import json
import os
import pathlib
import sys

try:
    root = pathlib.Path(sys.argv[1]).resolve(strict=True)
    relative = pathlib.PurePosixPath(sys.argv[2])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("ルートからの相対パスを指定してください。")
    target = (root / str(relative)).resolve(strict=True)
    if target != root and root not in target.parents:
        raise ValueError("指定パスはリモートルートの範囲外です。")
    entries = []
    with os.scandir(str(target)) as children:
        for child in children:
            if child.is_symlink():
                kind = "symlink"
            elif child.is_dir(follow_symlinks=False):
                kind = "directory"
            elif child.is_file(follow_symlinks=False):
                kind = "file"
            else:
                kind = "other"
            entries.append({"name": child.name, "path": str(relative / child.name), "type": kind})
    result = {
        "path": str(relative),
        "parent": None if str(relative) == "." else str(relative.parent),
        "entries": sorted(entries, key=lambda entry: (entry["type"] != "directory", entry["name"])),
    }
except (FileNotFoundError, NotADirectoryError):
    result = {"error": {"code": "not_found", "message": "指定ディレクトリが見つかりません。"}}
except PermissionError:
    result = {"error": {
        "code": "permission_denied", "message": "ディレクトリの読み取り権限がありません。",
    }}
except ValueError as exc:
    result = {"error": {"code": "invalid_path", "message": str(exc)}}
except OSError:
    result = {"error": {"code": "remote_error", "message": "リモートの一覧取得に失敗しました。"}}
print(json.dumps(result, ensure_ascii=True))
"""


class RemoteError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def relative_path(value):
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\x00" in value:
        raise ValueError("ルートからの相対パスを指定してください。")
    return str(path)


class RemoteBrowser:
    def __init__(self, cfg):
        if not PurePosixPath(cfg.remote_root).is_absolute():
            raise ValueError("remote_rootは絶対パスで指定してください。")
        self.root = cfg.remote_root
        self.ssh = cfg.ssh

    def list_directory(self, path):
        path = relative_path(path)
        ssh = self.ssh
        command = [
            "ssh",
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            "ConnectTimeout=15",
            "-o",
            f"ServerAliveInterval={ssh['server_alive_interval']}",
            "-o",
            f"ServerAliveCountMax={ssh['server_alive_count_max']}",
            "-o",
            f"TCPKeepAlive={'yes' if ssh['tcp_keep_alive'] else 'no'}",
            "-i",
            str(Path(ssh["identity_file"]).expanduser()),
            "-p",
            str(ssh["port"]),
            "-l",
            ssh["user"],
            "--",
            ssh["host"],
            shlex.join(["python3", "-c", LIST_DIRECTORY_SCRIPT, self.root, path]),
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired as exc:
            raise RemoteError("ssh_timeout", "SSHによる一覧取得がタイムアウトしました。") from exc
        except OSError as exc:
            raise RemoteError("ssh_failed", f"SSHを起動できません: {exc}") from exc
        if result.returncode:
            raise RemoteError(
                "ssh_failed", f"SSHによる一覧取得に失敗しました: {result.stderr.strip()}"
            )
        try:
            payload = json.loads(result.stdout)
            if "error" in payload:
                raise RemoteError(payload["error"]["code"], payload["error"]["message"])
            if (
                not isinstance(payload, dict)
                or payload.get("path") != path
                or not isinstance(payload.get("entries"), list)
                or "parent" not in payload
            ):
                raise ValueError("Invalid directory response")
        except (ValueError, TypeError, KeyError) as exc:
            raise RemoteError("remote_error", "リモートからの一覧応答を解釈できません。") from exc
        return payload
