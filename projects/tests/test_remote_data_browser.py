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
    assert api.post("/api/download").status_code == 404


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
