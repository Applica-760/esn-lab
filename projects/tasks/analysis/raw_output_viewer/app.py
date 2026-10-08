import base64
import logging
from threading import Lock
from typing import Literal

import uvicorn
from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from projects.tasks.analysis.raw_output_viewer.data import ResultStore
from projects.tasks.analysis.raw_output_viewer.plots import render_node_png
from projects.utils.app_init import setup_task_environment


def create_app(cfg):
    app = FastAPI(title="Raw output viewer")
    store = ResultStore(cfg.pred_result_dir)
    lock = Lock()

    @app.exception_handler(FileNotFoundError)
    async def not_found(request: Request, exc: FileNotFoundError):
        return JSONResponse(
            status_code=404, content={"error": {"code": "not_found", "message": str(exc)}}
        )

    @app.exception_handler(ValueError)
    async def invalid_data(request: Request, exc: ValueError):
        return JSONResponse(
            status_code=422, content={"error": {"code": "invalid_data", "message": str(exc)}}
        )

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "invalid_request",
                    "message": "APIの引数が不足しているか不正です。",
                }
            },
        )

    @app.get("/api/conditions")
    def conditions(group: str | None = None, parameter: str | None = None):
        with lock:
            return store.list_conditions(group, parameter)

    @app.get("/api/samples")
    def samples(group: str, parameter: str, fold: int = Query(ge=0)):
        with lock:
            return {"samples": store.list_samples(group, parameter, fold)}

    @app.get("/api/plots")
    def plots(
        group: str, parameter: str, mode: Literal["train", "test"], id: str, fold: int = Query(ge=0)
    ):
        # Matplotlibと選択中の結果キャッシュを同時に操作しない。
        with lock:
            sample = store.get_sample(group, parameter, fold, mode, id)
            images = {}
            warnings = []
            try:
                for index, node in enumerate(("foraging", "rumination")):
                    values = sample["predictions"][:, index]
                    images[node] = base64.b64encode(render_node_png(values, node)).decode("ascii")
                    if (values < -0.2).any() or (values > 1.2).any():
                        warnings.append(
                            {
                                "code": "out_of_range",
                                "node": node,
                                "message": f"{node}の出力が縦軸の範囲−0.2〜1.2を超えています。",
                            }
                        )
            except Exception:
                logging.exception("Failed to render sample %s", id)
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": {
                            "code": "render_failed",
                            "message": "グラフの生成に失敗しました。",
                        }
                    },
                )
            return {
                "group": group,
                "parameter": parameter,
                "fold": fold,
                "mode": mode,
                "id": id,
                "true_label": sample["true_label"],
                "length": len(sample["predictions"]),
                "mime_type": "image/png",
                "encoding": "base64",
                "images": images,
                "warnings": warnings,
            }

    return app


def main(cfg):
    uvicorn.run(create_app(cfg), host="127.0.0.1", port=cfg.port)


if __name__ == "__main__":
    main(setup_task_environment("projects.tasks.analysis.raw_output_viewer.app"))
