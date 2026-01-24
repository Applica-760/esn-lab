import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

from projects.utils.app_init import setup_app_environment, build_param_grid
from projects.utils.weights import build_param_str
from projects.utils.eval.filter import apply_filters
from projects.utils.eval.judgment import load_judgment_results

"""
python -m projects.apps.eval_dist --config projects/configs/eval_dist.yaml

目的：それぞれどのような割合で，判別に成功，失敗しているのかなどの分析を行う．
柔軟なパターンがプロットできるように，config側で条件を組み合わせながら色々プロットしたい
これにより，
「ラベルでフィルターをかけてプロットしたら，特定のこのラベルは判別が難しいことがわかった」
「判別に失敗しているものでフィルターをかけたら，意外と多数決にギリギリ負けているだけで，惜しいものが多いことがわかった」
などの考察ができるようにすることを目標とする．
"""


def count_true_pred_ratio(predictions, labels) -> float:
    """
    judge_sample_by_majority_voteを参考に，
    「フレームごとの予測ラベルのうち，真のラベルに該当するものが全体の時系列長の何%か」を取得して，小数点以下2桁で丸めて返す
    """
    predictions = np.array(predictions)
    labels = np.array(labels)

    # フレーム単位のラベルに変換（argmax）
    pred_frames = np.argmax(predictions, axis=1)
    true_frames = np.argmax(labels, axis=1)

    # 真のラベル（多数決）を取得
    true_label = int(np.argmax(np.bincount(true_frames)))

    # 真のラベルと一致するフレーム数をカウント
    match_count = np.sum(pred_frames == true_label)
    total_frames = len(pred_frames)

    # %を算出して小数点以下2桁で丸める
    ratio = round(match_count / total_frames, 2)
    return ratio


def graph_plot(ratios: list, output_path: Path, bins: int, color: str) -> None:
    """
    横軸：count_true_pred_ratioで求めた真のラベルに該当する予測回数の%
    縦軸：その度数
    ヒストグラムなど適切なものを用いて分布がわかりやすいようにプロットする
    """
    plt.figure(figsize=(8, 6))
    plt.hist(ratios, bins=bins, range=(0, 1), color=color, edgecolor='black', alpha=0.7)
    plt.xlabel('True Label Prediction Ratio')
    plt.ylabel('Frequency')
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def collect_ratios_for_param(
    param_name: str,
    sample_groups: list,
    mode: str,
    eval_dir: Path,
    pred_result_dir: Path,
    filters: dict
) -> list:
    """
    1つのパラメータ組み合わせについて、フィルタリング後のratioリストを収集
    返り値: [{"ratio": float, "true_label": int}, ...]
    """
    # judgment_resultsをロード
    judgment_csv_path = eval_dir / param_name / f"judgment_results_{mode}.csv"
    if not judgment_csv_path.exists():
        return []
    
    judgment_results = load_judgment_results(judgment_csv_path)
    
    # フィルタリング適用
    filtered_results = apply_filters(judgment_results, filters)
    
    if not filtered_results:
        return []
    
    # group, fold_indexごとにIDをグルーピング
    grouped = defaultdict(lambda: defaultdict(list))
    for r in filtered_results:
        grouped[r["group"]][r["fold_index"]].append({
            "id": r["id"],
            "true_label": r["true_label"]
        })
    
    # 各サンプルについてratioを計算
    ratio_results = []
    
    for group in grouped:
        for fold_index in grouped[group]:
            json_path = pred_result_dir / group / param_name / f"{mode}_results.json"
            if not json_path.exists():
                continue
            
            with open(json_path, 'r') as f:
                pred_results = json.load(f)
            
            # 該当fold_indexのデータを取得
            fold_data = next((fd for fd in pred_results if fd["fold_index"] == fold_index), None)
            if fold_data is None:
                continue
            
            results = fold_data["results"]
            
            # IDからサンプルを検索してratio計算
            for item in grouped[group][fold_index]:
                sample_id = item["id"]
                true_label = item["true_label"]
                
                sample = next((s for s in results if s["id"] == sample_id), None)
                if sample is None:
                    continue
                
                ratio = count_true_pred_ratio(sample["predictions"], sample["labels"])
                ratio_results.append({
                    "ratio": ratio,
                    "true_label": true_label
                })
    
    return ratio_results


def main():
    cfg, output_dir = setup_app_environment()
    
    eval_dir = Path(cfg.eval_dir)
    pred_result_dir = Path(cfg.pred_result_dir)
    param_grid = build_param_grid(cfg)
    
    for mode in cfg.mode:
        print(f"Processing mode: {mode}")
        
        # 全パラメータ組み合わせについてデータを合算
        all_ratio_results = []
        
        for params in param_grid:
            param_name = build_param_str(params)
            ratio_results = collect_ratios_for_param(
                param_name, cfg.sample_groups, mode, eval_dir, pred_result_dir, cfg.filters
            )
            all_ratio_results.extend(ratio_results)
        
        if not all_ratio_results:
            print(f"  No data found for mode: {mode}")
            continue
        
        print(f"  Total samples: {len(all_ratio_results)}")
        
        # 全サンプルのヒストグラム
        all_ratios = [r["ratio"] for r in all_ratio_results]
        output_path = output_dir / f"dist_all_{mode}.png"
        graph_plot(all_ratios, output_path, cfg.bins, cfg.colors["all"])
        print(f"  Saved: {output_path}")
        
        # true_labelごとのヒストグラム
        for i, class_name in enumerate(cfg.class_names):
            label_index = cfg.class_order[i]  # 表示名に対応するJSONのインデックス
            label_ratios = [r["ratio"] for r in all_ratio_results if r["true_label"] == label_index]
            
            if not label_ratios:
                print(f"  No data for {class_name}")
                continue
            
            output_path = output_dir / f"dist_{class_name}_{mode}.png"
            graph_plot(label_ratios, output_path, cfg.bins, cfg.colors[class_name])
            print(f"  Saved: {output_path} (n={len(label_ratios)})")
    
    print("plot finished")


if __name__ == "__main__":
    main()