from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, StrictBool

from projects.tasks.remote_data_browser.remote import RemoteBrowser, RemoteError, relative_path
from projects.tasks.remote_data_browser.transfer import Transfer
from projects.utils.app_init import setup_task_environment


class PreviewRequest(BaseModel):
    path: str


class DownloadRequest(BaseModel):
    token: str
    overwrite: StrictBool = False


def create_app(cfg):
    browser = RemoteBrowser(cfg)
    initial_path = relative_path(cfg.initial_path)
    repository = Path(__file__).resolve().parents[3]
    transfer = Transfer(browser, cfg, repository)
    app = FastAPI(title="Remote data browser")
    frontend = repository / "projects" / "app" / "remote_data_browser"
    app.mount("/assets", StaticFiles(directory=frontend), name="assets")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(frontend / "index.html")

    @app.exception_handler(RemoteError)
    async def remote_error(request: Request, exc: RemoteError):
        status = {
            "not_found": 404,
            "permission_denied": 403,
            "invalid_path": 422,
            "ssh_timeout": 504,
            "confirmation_required": 409,
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
            "local_destination_absolute": str(transfer.destination),
        }

    @app.get("/api/directories")
    def directories(path: str | None = None):
        return browser.list_directory(initial_path if path is None else path)

    @app.post("/api/download/preview")
    def preview(body: PreviewRequest):
        return transfer.preview(body.path)

    @app.post("/api/download")
    def download(body: DownloadRequest):
        return transfer.download(body.token, body.overwrite)

    return app


def main(cfg):
    uvicorn.run(create_app(cfg), host="127.0.0.1", port=cfg.port)


if __name__ == "__main__":
    main(setup_task_environment("projects.tasks.remote_data_browser.app"))
