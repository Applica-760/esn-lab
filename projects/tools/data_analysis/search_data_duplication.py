"""
dataset/10fold_npy_div内のデータ重複度を分析するスクリプト

10分割されたデータ群(fold_a~fold_j)間でのサンプルID重複の程度を定量的に分析し、
Cross-sourceのペアワイズ共通ID数ヒートマップのみをoutputs配下に書き出す。
"""
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
import datetime
import matplotlib.pyplot as plt
from matplotlib.table import Table


def load_sample_ids_from_npz(npz_path: Path) -> list[dict]:
    """npzファイルから全サンプルIDとクラスを取得
    
    Args:
        npz_path: npzファイルのパス
        
    Returns:
        サンプル情報のリスト [{"id": str, "class": int}, ...]
    """
    data = np.load(npz_path)
    num_samples = int(data["num_samples"])
    samples = []
    for i in range(num_samples):
        samples.append({
            "id": str(data[f"sample_{i}_id"]),
            "class": int(data[f"sample_{i}_class"])
        })
    return samples


def analyze_source_directory(source_dir: Path, source_char: str) -> dict:
    """1つのsourceディレクトリ内の10foldsを分析
    
    Args:
        source_dir: 分析対象のディレクトリパス
        source_char: sourceディレクトリの文字 (a~j)
        
    Returns:
        分析結果の辞書
    """
    fold_data = {}  # fold_name → [samples...]
    
    # 各foldのデータを読み込み
    for fold_char in "abcdefghij":
        npz_path = source_dir / f"fold_{fold_char}.npz"
        if npz_path.exists():
            fold_data[fold_char] = load_sample_ids_from_npz(npz_path)
        else:
            print(f"  警告: {npz_path} が見つかりません")
    
    # サンプルIDの出現頻度を計算
    id_to_folds = defaultdict(list)  # sample_id → [fold_chars...]
    all_ids = []
    
    for fold_char, samples in fold_data.items():
        for sample in samples:
            sample_id = sample["id"]
            id_to_folds[sample_id].append(fold_char)
            all_ids.append(sample_id)
    
    # 重複度の統計
    total_samples = len(all_ids)
    unique_ids = len(id_to_folds)
    duplication_counts = Counter([len(folds) for folds in id_to_folds.values()])
    
    # 重複しているサンプルを抽出
    duplicated_ids = {
        sample_id: folds 
        for sample_id, folds in id_to_folds.items() 
        if len(folds) > 1
    }
    
    return {
        "fold_data": fold_data,
        "total_samples": total_samples,
        "unique_ids": unique_ids,
        "duplication_counts": duplication_counts,
        "duplicated_ids": duplicated_ids,
        "id_to_folds": id_to_folds
    }


def print_source_summary(source_char: str, analysis: dict):
    return


def _render_pairwise_heatmap(source_to_ids: dict[str, set], output_path: Path, title: str, denom: float):
    """Render normalized upper-triangle heatmap including diagonal.

    Args:
        source_to_ids: mapping from source char to set of sample IDs
        output_path: path to save png
        title: plot title
        denom: normalization denominator (e.g., 255 for overall, 85 for per-class)
    """
    sources = sorted(source_to_ids.keys())
    n = len(sources)
    if n < 2:
        return
    mat = np.zeros((n, n), dtype=float)
    for i, sa in enumerate(sources):
        for j, sb in enumerate(sources):
            common = source_to_ids[sa] & source_to_ids[sb]
            mat[i, j] = float(len(common))
    mat = mat / float(denom)
    mask = np.tril(np.ones_like(mat, dtype=bool), k=-1)
    mat_masked = np.ma.array(mat, mask=mask)
    fig, ax = plt.subplots(figsize=(1.2*n, 1.0*n))
    im = ax.imshow(mat_masked, cmap='Blues', vmin=0.0, vmax=1.0)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(sources)
    ax.set_yticklabels(sources)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(n):
        for j in range(n):
            if j >= i:
                ax.text(j, i, f"{mat[i, j]:.2f}", ha='center', va='center', color='black', fontsize=9)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def analyze_cross_source_duplication(all_analyses: dict, output_dir: Path):
    """Cross-source pairwise overlap heatmaps: overall and per class (0,1,2)."""
    
    # 各sourceから全サンプルIDを収集
    source_to_ids = {}
    for source_char in "abcdefghij":
        if source_char in all_analyses:
            analysis = all_analyses[source_char]
            all_ids = set()
            for samples in analysis['fold_data'].values():
                for sample in samples:
                    all_ids.add(sample['id'])
            source_to_ids[source_char] = all_ids
    
    # Build overall source_to_ids (all labels)
    
    _render_pairwise_heatmap(source_to_ids, output_dir / "cross_sources_pairwise_overlap_counts.png", "Pairwise common ID ratio (upper triangle) - overall", denom=255.0)

    # Build class-wise source_to_ids
    for cls in [0, 1, 2]:
        source_to_ids_cls = {}
        for source_char in "abcdefghij":
            if source_char in all_analyses:
                analysis = all_analyses[source_char]
                ids = set()
                for samples in analysis['fold_data'].values():
                    for sample in samples:
                        if sample['class'] == cls:
                            ids.add(sample['id'])
                source_to_ids_cls[source_char] = ids
        _render_pairwise_heatmap(
            source_to_ids_cls,
            output_dir / f"cross_sources_pairwise_overlap_counts_class{cls}.png",
            f"Pairwise common ID ratio (upper triangle) - class {cls}",
            denom=85.0
        )


def print_overall_summary(all_analyses: dict, output_dir: Path):
    return


def main():
    base_dir = Path(__file__).parent.parent.parent / "dataset" / "10fold_npy_div"
    outputs_dir = Path(__file__).parent.parent.parent / "outputs" / "duplication_analysis"
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = outputs_dir / timestamp
    
    if not base_dir.exists():
        print(f"エラー: {base_dir} が見つかりません")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("Cross-source pairwise overlap (upper triangle) -> outputs")
    print("="*80)
    
    all_analyses = {}
    
    # 各sourceディレクトリを分析
    for source_char in "abcdefghij":
        source_dir = base_dir / source_char
        if source_dir.exists():
            analysis = analyze_source_directory(source_dir, source_char)
            analysis["output_dir"] = output_dir
            all_analyses[source_char] = analysis
        else:
            print(f"\n警告: {source_dir} が見つかりません")
    
    # 全体サマリー
    # Skip per-source and overall summary; only cross-source outputs
    
    # Source間の重複分析
    analyze_cross_source_duplication(all_analyses, output_dir)
    
    print("\n" + "="*80)
    print("分析完了: 出力先")
    print(str(output_dir))
    print("="*80)


if __name__ == "__main__":
    main()

