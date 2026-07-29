"""
python -m projects.tasks.cli <task_name>

python -m projects.tasks.cli train
python -m projects.tasks.cli eval.dist
"""

import argparse
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import importlib

from projects.utils.app_init import setup_task_environment

TASK_REGISTRY = {
    "train": "projects.tasks.train.app",
    "pred": "projects.tasks.pred.app",
    "pred.margin": "projects.tasks.pred_margin.app",
    "eval.dist": "projects.tasks.eval.dist.app",
    "eval.dist_node": "projects.tasks.eval.dist_node.app",
    "eval.judge": "projects.tasks.eval.judge.app",
    "eval.metrics": "projects.tasks.eval.metrics.app",
    "eval.plot": "projects.tasks.eval.plot.app",
    "analysis.pred": "projects.tasks.analysis.pred.app",
    "analysis.bayesian": "projects.tasks.analysis.bayesian.app",
}


def main():
    """CLIエントリーポイント"""
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=TASK_REGISTRY.keys())
    parser.add_argument("--config", type=str, default=None, help="cfg.yaml 以外の設定ファイル名（タスクディレクトリ内）")
    args = parser.parse_args()

    module_path = TASK_REGISTRY[args.task]
    cfg = setup_task_environment(module_path, config_name=args.config)

    module = importlib.import_module(module_path)
    module.main(cfg)


if __name__ == "__main__":
    main()
