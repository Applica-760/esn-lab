import secrets
import shlex
import subprocess
from pathlib import PurePosixPath
from threading import Lock

from projects.tasks.remote_data_browser.remote import RemoteError, relative_path


class Transfer:
    def __init__(self, browser, cfg, repository):
        self.browser = browser
        self.base = PurePosixPath(relative_path(cfg.initial_path))
        self.repository = repository.resolve()
        destination = relative_path(cfg.local_destination)
        if destination == ".":
            raise ValueError("local_destinationには保存用ディレクトリを指定してください。")
        self.destination = self.repository / destination
        self.check_local(self.destination, "directory")
        self.pending = None
        self.lock = Lock()

    def check_local(self, target, kind):
        relative = target.relative_to(self.repository)
        current = self.repository
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise ValueError("保存先のシンボリックリンクは利用できません。")
            if current.exists():
                expected = kind if current == target else "directory"
                if not (current.is_dir() if expected == "directory" else current.is_file()):
                    raise ValueError(f"保存先のファイル種別が衝突しています: {current}")

    def inspect(self, path):
        selected = PurePosixPath(relative_path(path))
        relative = selected.relative_to(self.base)
        # 選択パスの途中にあるリンクも転送対象にしない。
        current = PurePosixPath(".")
        for part in selected.parts:
            listing = self.browser.list_directory(str(current))
            current = current / part
            entry = next((e for e in listing["entries"] if e["name"] == part), None)
            if entry is None:
                raise RemoteError("not_found", "取得対象が見つかりません。")
            if entry["type"] != "directory":
                raise ValueError("リンクを含まないディレクトリを指定してください。")

        entries = [(str(relative), "directory")]
        skipped = []
        queue = [selected]
        while queue:
            directory = queue.pop()
            for entry in self.browser.list_directory(str(directory))["entries"]:
                name = entry["name"]
                if not name or name in (".", "..") or "/" in name or "\x00" in name:
                    raise RemoteError("remote_error", "不正なファイル名を受信しました。")
                child = directory / name
                if entry["path"] != str(child):
                    raise RemoteError("remote_error", "一覧のパスが一致しません。")
                kind = entry["type"]
                item = str(child.relative_to(self.base))
                if kind not in ("directory", "file"):
                    skipped.append(item)
                    continue
                entries.append((item, kind))
                if kind == "directory":
                    queue.append(child)

        conflicts = []
        for item, kind in entries:
            target = self.destination / item
            self.check_local(target, kind)
            if kind == "file" and target.exists():
                conflicts.append(item)
        return {
            "path": str(selected),
            "destination": str(self.destination / str(relative)),
            "entries": sorted(entries),
            "conflicts": sorted(conflicts),
            "skipped": sorted(skipped),
        }

    def preview(self, path):
        with self.lock:
            self.pending = None
            plan = self.inspect(path)
            token = secrets.token_urlsafe(32)
            self.pending = (token, plan)
            return {key: value for key, value in plan.items() if key != "entries"} | {
                "token": token,
                "file_count": sum(kind == "file" for _, kind in plan["entries"]),
            }

    def download(self, token, overwrite):
        with self.lock:
            if self.pending is None or not secrets.compare_digest(
                self.pending[0].encode(), token.encode()
            ):
                raise RemoteError("confirmation_required", "転送前の確認をやり直してください。")
            plan = self.pending[1]
            if plan["conflicts"] and not overwrite:
                raise RemoteError("confirmation_required", "上書きの承認が必要です。")
            self.pending = None
            if self.inspect(plan["path"]) != plan:
                raise RemoteError(
                    "confirmation_required", "対象が変わりました。再確認してください。"
                )
            try:
                self.destination.mkdir(parents=True, exist_ok=True)
                source = str(PurePosixPath(self.browser.root) / self.base) + "/"
                command = [
                    "/usr/bin/rsync",
                    "-dt",
                    "--checksum",
                    "--from0",
                    "--files-from=-",
                    "--timeout=120",
                    "-e",
                    shlex.join(self.browser.ssh_command()),
                ]
                endpoints = [
                    "--",
                    f"{self.browser.ssh['host']}:{shlex.quote(source)}",
                    str(self.destination) + "/",
                ]
                conflicts = set(plan["conflicts"])
                # 上書きできるのは承認されたパスだけ。新規対象にはignore-existingを付ける。
                for replacing in (False, True):
                    items = [
                        item for item, _ in plan["entries"] if (item in conflicts) == replacing
                    ]
                    if not items:
                        continue
                    options = [] if replacing else ["--ignore-existing"]
                    # 再帰せず、確認済みの項目だけをNUL区切りで渡す。
                    manifest = b"".join(item.encode() + b"\x00" for item in items)
                    result = subprocess.run(
                        command + options + endpoints, input=manifest, capture_output=True
                    )
                    if result.returncode:
                        detail = result.stderr.decode(errors="replace").strip()
                        raise RemoteError(
                            "transfer_failed",
                            f"転送に失敗しました。転送済みファイルは残ります: {detail}",
                        )
            except OSError as exc:
                raise RemoteError("transfer_failed", f"転送を実行できません: {exc}") from exc
            return {"status": "completed", "path": plan["path"], "destination": plan["destination"]}
