"""
python -m projects.tasks.cli <task_name>

python -m projects.tasks.cli train
python -m projects.tasks.cli eval.dist
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import sys
import importlib
from projects.utils.app_init import setup_task_environment

TASK_REGISTRY = {
    "train": "projects.tasks.train.app",
    "pred": "projects.tasks.pred.app",
    "eval.dist": "projects.tasks.eval.dist.app",
    "eval.dist_node": "projects.tasks.eval.dist_node.app",
    "eval.judge": "projects.tasks.eval.judge.app",
    "eval.metrics": "projects.tasks.eval.metrics.app",
    "eval.plot": "projects.tasks.eval.plot.app",
}


def main():
    """CLIエントリーポイント"""
    # モジュールパスを取得
    module_path = TASK_REGISTRY[sys.argv[1]]
    
    # タスクの設定を読み込み
    cfg = setup_task_environment(module_path)
    
    # タスクモジュールを動的にインポート
    module = importlib.import_module(module_path)

    module.main(cfg)


if __name__ == "__main__":
    main()

