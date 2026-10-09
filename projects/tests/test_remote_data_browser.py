import json
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from projects.tasks import cli
from projects.tasks.remote_data_browser import app as browser_app
from projects.tasks.remote_data_browser import remote
from projects.tasks.remote_data_browser.transfer import Transfer
from projects.utils.app_init import load_config


def config():
    return SimpleNamespace(
        ssh={
            "host": "example.org",
            "user": "user",
            "identity_file": "/keys/id_test",
            "port": 2222,
            "server_alive_interval": 60,
            "server_alive_count_max": 10,
            "tcp_keep_alive": True,
        },
        remote_root="/remote/repo",
        initial_path="outputs",
        local_destination="outputs/downloads/test",
        port=8001,
    )


def run_listing(root, path):
    result = subprocess.run(
        [sys.executable, "-c", remote.LIST_DIRECTORY_SCRIPT, str(root), path],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def test_remote_script_lists_one_level_and_preserves_names(tmp_path):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    name = "日本語 'quoted'; $(echo test)\nfolder"
    (outputs / name).mkdir()
    (outputs / name / "nested").mkdir()
    (outputs / "result.npz").write_bytes(b"test")
    (outputs / "link").symlink_to(outputs / name, target_is_directory=True)
    result = run_listing(tmp_path, "outputs")
    assert result["parent"] == "."
    assert result["entries"] == [
        {"name": name, "path": f"outputs/{name}", "type": "directory"},
        {"name": "link", "path": "outputs/link", "type": "symlink"},
        {"name": "result.npz", "path": "outputs/result.npz", "type": "file"},
    ]
    assert run_listing(tmp_path, ".")["parent"] is None
    assert run_listing(tmp_path, "outputs/missing")["error"]["code"] == "not_found"


def test_remote_script_rejects_escape_and_external_symlink(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "outside").symlink_to(tmp_path, target_is_directory=True)
    for path in ("../", "/", "outside"):
        assert run_listing(root, path)["error"]["code"] == "invalid_path"


def test_ssh_command_uses_yaml_and_quotes_paths(monkeypatch):
    calls = []
    path = "outputs/space ' ; $(touch unsafe)"

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(
                {
                    "path": path,
                    "parent": "outputs",
                    "entries": [],
                }
            ),
            "",
        )

    monkeypatch.setattr(remote.subprocess, "run", run)
    assert remote.RemoteBrowser(config()).list_directory(path)["entries"] == []
    command, kwargs = calls[0]
    assert command[command.index("-p") + 1] == "2222"
    assert command[command.index("-i") + 1] == "/keys/id_test"
    assert command[command.index("-l") + 1] == "user"
    assert command[-2] == "example.org"
    assert "StrictHostKeyChecking=yes" in command
    assert shlex.split(command[-1]) == [
        "python3",
        "-c",
        remote.LIST_DIRECTORY_SCRIPT,
        "/remote/repo",
        path,
    ]
    assert kwargs["timeout"] == 120 and not kwargs.get("shell", False)
    with pytest.raises(ValueError):
        remote.RemoteBrowser(config()).list_directory("../outside")
    assert len(calls) == 1


@pytest.mark.parametrize(
    "code,status",
    [
        ("not_found", 404),
        ("permission_denied", 403),
        ("invalid_path", 422),
        ("ssh_failed", 502),
        ("ssh_timeout", 504),
    ],
)
def test_api_errors(monkeypatch, code, status):
    def fail(self, path):
        raise remote.RemoteError(code, "確認用エラー")

    monkeypatch.setattr(remote.RemoteBrowser, "list_directory", fail)
    response = TestClient(browser_app.create_app(config())).get("/api/directories")
    assert response.status_code == status
    assert response.json()["error"] == {"code": code, "message": "確認用エラー"}


def test_config_and_default_path_are_injected(monkeypatch):
    seen = []
    monkeypatch.setattr(
        remote.RemoteBrowser,
        "list_directory",
        lambda self, path: seen.append(path) or {"path": path, "parent": ".", "entries": []},
    )
    cfg = config()
    cfg.initial_path = "another"
    api = TestClient(browser_app.create_app(cfg))
    settings = api.get("/api/config").json()
    assert settings["remote_root"] == cfg.remote_root
    assert settings["local_destination"] == cfg.local_destination
    assert settings["local_destination_absolute"] == str(
        Path(browser_app.__file__).resolve().parents[3] / cfg.local_destination
    )
    assert "ssh" not in settings
    assert api.get("/api/directories").status_code == 200
    api.get("/api/directories", params={"path": "another/child"})
    assert seen == ["another", "another/child"]
    assert api.post("/api/download").status_code == 422


@pytest.mark.parametrize("response", ["banner\n{}", "null", "{}"])
def test_bad_ssh_response_is_reported(monkeypatch, response):
    monkeypatch.setattr(
        remote.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess([], 0, response, ""),
    )
    with pytest.raises(remote.RemoteError, match="応答を解釈"):
        remote.RemoteBrowser(config()).list_directory("outputs")


def test_ssh_timeout_and_failure(monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("ssh", 120)

    monkeypatch.setattr(remote.subprocess, "run", timeout)
    with pytest.raises(remote.RemoteError) as caught:
        remote.RemoteBrowser(config()).list_directory("outputs")
    assert caught.value.code == "ssh_timeout"
    monkeypatch.setattr(
        remote.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess([], 255, "", "Permission denied"),
    )
    with pytest.raises(remote.RemoteError, match="Permission denied"):
        remote.RemoteBrowser(config()).list_directory("outputs")


def test_cli_configuration_and_start(monkeypatch):
    assert cli.TASK_REGISTRY["remote_data_browser"] == browser_app.__name__
    cfg = load_config(str(Path(browser_app.__file__).with_name("cfg.yaml")))
    assert cfg.local_destination == "outputs/downloads/genkai/"
    assert cfg.remote_root == "/home/pj24003114/ku50001550/ESN-LAB/esn-lab/"
    calls = []
    monkeypatch.setattr(browser_app.uvicorn, "run", lambda app, **kwargs: calls.append(kwargs))
    browser_app.main(cfg)
    assert calls == [{"host": "127.0.0.1", "port": 8001}]


@pytest.fixture
def transfer(tmp_path, monkeypatch):
    root = tmp_path / "remote"
    (root / "outputs" / "experiments" / "run").mkdir(parents=True)
    cfg = config()
    cfg.remote_root = str(root)
    cfg.local_destination = "downloads/genkai"
    browser = remote.RemoteBrowser(cfg)
    monkeypatch.setattr(browser, "list_directory", lambda path: run_listing(root, path))
    return Transfer(browser, cfg, tmp_path / "local")


def local_rsync(monkeypatch, transfer, calls):
    original = subprocess.run

    def run(command, **kwargs):
        if command[0] != "/usr/bin/rsync":
            return original(command, **kwargs)
        calls.append((command, kwargs))
        local = command.copy()
        index = local.index("-e")
        del local[index : index + 2]
        local[-2] = str(Path(transfer.browser.root) / "outputs") + "/"
        return original(local, **kwargs)

    monkeypatch.setattr(subprocess, "run", run)


def test_transfer_preserves_relative_tree_and_skips_links(transfer, monkeypatch):
    source = Path(transfer.browser.root) / "outputs/experiments/run"
    name = "日本語 'quoted'; $(echo unsafe)\nresult.txt"
    (source / name).write_text("new")
    (source / "empty").mkdir()
    (source / "link").symlink_to(source / name)
    (source / "dir-link").symlink_to(source, target_is_directory=True)
    calls = []
    local_rsync(monkeypatch, transfer, calls)
    plan = transfer.preview("outputs/experiments/run")
    assert plan["destination"] == str(transfer.destination / "experiments/run")
    assert plan["file_count"] == 1
    assert plan["skipped"] == ["experiments/run/dir-link", "experiments/run/link"]
    assert not transfer.destination.exists()
    assert transfer.download(plan["token"], False)["status"] == "completed"
    target = transfer.destination / "experiments/run"
    assert (target / name).read_text() == "new"
    assert (target / "empty").is_dir()
    assert not (target / "link").exists()
    assert not (target / "dir-link").exists()
    assert "--ignore-existing" in calls[0][0]
    assert b"link\x00" not in calls[0][1]["input"]


def test_api_requires_preview_and_overwrite_approval(transfer, monkeypatch):
    source = Path(transfer.browser.root) / "outputs/experiments/run/result"
    source.write_text("new")
    target = transfer.destination / "experiments/run/result"
    target.parent.mkdir(parents=True)
    target.write_text("old")
    calls = []
    local_rsync(monkeypatch, transfer, calls)
    monkeypatch.setattr(browser_app, "Transfer", lambda *args: transfer)
    api = TestClient(browser_app.create_app(config()))
    assert (
        api.post("/api/download", json={"token": "invalid", "overwrite": True}).status_code == 409
    )
    plan = api.post("/api/download/preview", json={"path": "outputs/experiments/run"}).json()
    assert plan["conflicts"] == ["experiments/run/result"]
    assert api.post("/api/download", json={"token": "不正なtoken"}).status_code == 409
    assert api.post("/api/download", json={"token": plan["token"]}).status_code == 409
    assert (
        api.post("/api/download", json={"token": plan["token"], "overwrite": "false"}).status_code
        == 422
    )
    assert target.read_text() == "old" and not calls
    response = api.post("/api/download", json={"token": plan["token"], "overwrite": True})
    assert response.status_code == 200
    assert target.read_text() == "new"
    assert len(calls) == 2
    assert "--ignore-existing" in calls[0][0]
    assert "--ignore-existing" not in calls[1][0]
    assert calls[1][1]["input"] == b"experiments/run/result\x00"
    assert (
        api.post("/api/download", json={"token": plan["token"], "overwrite": True}).status_code
        == 409
    )


def test_transfer_rejects_changed_collision_list(transfer, monkeypatch):
    source = Path(transfer.browser.root) / "outputs/experiments/run/result"
    source.write_text("new")
    plan = transfer.preview("outputs/experiments/run")
    target = transfer.destination / "experiments/run/result"
    target.parent.mkdir(parents=True)
    target.write_text("appeared after preview")
    calls = []
    local_rsync(monkeypatch, transfer, calls)
    with pytest.raises(remote.RemoteError, match="再確認"):
        transfer.download(plan["token"], True)
    assert not calls
    assert target.read_text() == "appeared after preview"


@pytest.mark.parametrize("path", ["../outside", "/outputs", ".", "other"])
def test_transfer_rejects_paths_outside_initial_directory(transfer, path):
    with pytest.raises(ValueError):
        transfer.preview(path)


def test_transfer_rejects_remote_link_and_local_link_or_type_conflict(transfer):
    source = Path(transfer.browser.root) / "outputs"
    (source / "alias").symlink_to(source / "experiments", target_is_directory=True)
    with pytest.raises(ValueError, match="リンク"):
        transfer.preview("outputs/alias/run")
    transfer.destination.mkdir(parents=True)
    target = transfer.destination / "experiments"
    target.symlink_to(source / "experiments", target_is_directory=True)
    with pytest.raises(ValueError, match="リンク"):
        transfer.preview("outputs/experiments/run")
    target.unlink()
    target.write_text("keep")
    with pytest.raises(ValueError, match="種別"):
        transfer.preview("outputs/experiments/run")
    assert target.read_text() == "keep"


@pytest.mark.parametrize("destination", ["../escape", "/absolute", "."])
def test_transfer_rejects_unsafe_destination(tmp_path, destination):
    cfg = config()
    cfg.local_destination = destination
    with pytest.raises(ValueError):
        Transfer(remote.RemoteBrowser(cfg), cfg, tmp_path)


def test_transfer_failure_keeps_completed_files_and_allows_retry(transfer, monkeypatch):
    source = Path(transfer.browser.root) / "outputs/experiments/run"
    (source / "first").write_text("first")
    (source / "second").write_text("second")
    original = subprocess.run

    def fail(command, **kwargs):
        if command[0] != "/usr/bin/rsync":
            return original(command, **kwargs)
        target = transfer.destination / "experiments/run"
        target.mkdir(parents=True)
        (target / "first").write_text("first")
        return subprocess.CompletedProcess(command, 23, b"", b"interrupted")

    monkeypatch.setattr(subprocess, "run", fail)
    plan = transfer.preview("outputs/experiments/run")
    with pytest.raises(remote.RemoteError, match="転送済みファイルは残ります"):
        transfer.download(plan["token"], False)
    assert (transfer.destination / "experiments/run/first").read_text() == "first"
    monkeypatch.setattr(subprocess, "run", original)
    local_rsync(monkeypatch, transfer, [])
    plan = transfer.preview("outputs/experiments/run")
    assert plan["conflicts"] == ["experiments/run/first"]
    transfer.download(plan["token"], True)
    assert (transfer.destination / "experiments/run/second").read_text() == "second"
