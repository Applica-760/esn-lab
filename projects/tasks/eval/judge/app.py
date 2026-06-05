from pathlib import Path

import numpy as np

from projects.utils.eval.judgment import (
    JUDGE_STRATEGIES,
    compute_judgment_results,
    make_weighted_score_judge,
    save_judgment_results,
)
from projects.utils.prediction import is_valid_result_file, load_pred_results
from projects.utils.weights import list_param_dirs


def _resolve_judge_fn(cfg):
    if cfg.judge_strategy == "weighted_score":
        p = cfg.judge_params
        return make_weighted_score_judge(
            np.array([p.w0, p.w1, p.w2]),
            p.margin_threshold,
        )
    return JUDGE_STRATEGIES[cfg.judge_strategy]

"""
python -m projects.apps.eval_judgement --config projects/configs/eval_judgement.yaml

目的：予測結果から判定結果を生成し、CSVファイルとして保存する。
この判定結果は、eval_metrics, eval_dist, eval_plotなどの他のappで使用される。
"""


def compute_and_save_judgments(
    param_dirs, sample_groups, mode, pred_result_dir, output_dir, strategy
):
    """
    判定結果の計算と保存
    """
    param_judgment_results = {param_dir.name: [] for param_dir in param_dirs}

    for param_dir in param_dirs:
        param_name = param_dir.name

        # すべてのgroupを処理
        for group in sample_groups:
            result_base = str(pred_result_dir / group / param_name / f"{mode}_results")

            if not is_valid_result_file(result_base):
                continue

            pred_results = load_pred_results(result_base)
            judgment_results = compute_judgment_results(pred_results, strategy, group=group)
            param_judgment_results[param_name].extend(judgment_results)

        # このparam_dirのすべてのgroupを処理し終えたので保存
        if param_judgment_results[param_name]:
            output_path = output_dir / param_name / f"judgment_results_{mode}"
            save_judgment_results(param_judgment_results[param_name], output_path)
            print(f"  Saved: {output_path}.csv")

    return param_judgment_results


def main(cfg):
    pred_result_dir = Path(cfg.pred_result_dir)
    sample_groups = cfg.sample_groups
    modes = cfg.mode  # リスト形式

    # 最初のサンプル群からパラメータディレクトリ一覧を取得
    first_group_dir = pred_result_dir / sample_groups[0]
    param_dirs = list_param_dirs(first_group_dir)

    # modeループ
    for mode in modes:
        print(f"Processing mode: {mode}")

        # 判定結果の計算と保存
        compute_and_save_judgments(
            param_dirs,
            sample_groups,
            mode,
            pred_result_dir,
            cfg.output_dir,
            strategy=_resolve_judge_fn(cfg),
        )

    print("judgment computation finished")
