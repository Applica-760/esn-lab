from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from projects.tasks.remote_data_browser.remote import RemoteBrowser, RemoteError, relative_path
from projects.utils.app_init import setup_task_environment


def create_app(cfg):
    browser = RemoteBrowser(cfg)
    initial_path = relative_path(cfg.initial_path)
    destination = Path(cfg.local_destination)
    if destination.is_absolute():
        raise ValueError("local_destinationはローカルの相対パスで指定してください。")
    repository = Path(__file__).resolve().parents[3]
    local_destination = (repository / destination).resolve()
    app = FastAPI(title="Remote data browser")

    @app.exception_handler(RemoteError)
    async def remote_error(request: Request, exc: RemoteError):
        status = {
            "not_found": 404,
            "permission_denied": 403,
            "invalid_path": 422,
            "ssh_timeout": 504,
        }.get(exc.code, 502)
        return JSONResponse(
            status_code=status,
            content={
                "error": {"code": exc.code, "message": str(exc)},
            },
        )

    @app.exception_handler(ValueError)
    async def invalid_path(request: Request, exc: ValueError):
        return JSONResponse(
            status_code=422,
            content={
                "error": {"code": "invalid_path", "message": str(exc)},
            },
        )

    @app.get("/api/config")
    def config():
        return {
            "remote_root": cfg.remote_root,
            "initial_path": initial_path,
            "local_destination": cfg.local_destination,
            "local_destination_absolute": str(local_destination),
        }

    @app.get("/api/directories")
    def directories(path: str | None = None):
        return browser.list_directory(initial_path if path is None else path)

    return app


def main(cfg):
    uvicorn.run(create_app(cfg), host="127.0.0.1", port=cfg.port)


if __name__ == "__main__":
    main(setup_task_environment("projects.tasks.remote_data_browser.app"))
