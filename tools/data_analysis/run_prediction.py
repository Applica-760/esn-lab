"""
単一サンプルに対する推論を実行し、出力ノード値を記録するスクリプト

学習済み重みを用いて1つのサンプルの推論を実行し、
各時刻における出力層の値を.csv形式で保存します。

使用例:

08274
python tools/data_analysis/run_prediction.py \
    --npz_path dataset/10fold_npy_div/c/fold_b.npz \
    --sample_index 0 \
    --weight_path outputs/experiments/tenfold_integ/weights/seed-nonseed_nx-700_density-05_input_scale-0001_rho-09_acdefghij_Wout.npy \
    --output_dir outputs/prediction_outputs

09381
python tools/data_analysis/run_prediction.py \
    --npz_path dataset/10fold_npy_div/c/fold_b.npz \
    --sample_index 4 \
    --weight_path outputs/experiments/tenfold_integ/weights/seed-nonseed_nx-700_density-05_input_scale-0001_rho-09_acdefghij_Wout.npy \
    --output_dir outputs/prediction_outputs

07545
python tools/data_analysis/run_prediction.py \
    --npz_path dataset/10fold_npy_div/c/fold_b.npz \
    --sample_index 19 \
    --weight_path outputs/experiments/tenfold_integ/weights/seed-nonseed_nx-700_density-05_input_scale-0001_rho-09_acdefghij_Wout.npy \
    --output_dir outputs/prediction_outputs

09432
python tools/data_analysis/run_prediction.py \
    --npz_path dataset/10fold_npy_div/c/fold_b.npz \
    --sample_index 5 \
    --weight_path outputs/experiments/tenfold_integ/weights/seed-nonseed_nx-700_density-05_input_scale-0001_rho-09_acdefghij_Wout.npy \
    --output_dir outputs/prediction_outputs

---
06565
python tools/data_analysis/run_prediction.py \
    --npz_path dataset/10fold_npy_div/b/fold_b.npz \
    --sample_index 1 \
    --weight_path outputs/experiments/tenfold_integ/weights/seed-nonseed_nx-700_density-05_input_scale-0001_rho-09_acdefghij_Wout.npy \
    --output_dir outputs/prediction_outputs

"""
import numpy as np
import pandas as pd
import argparse
import re
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# esn_labパッケージのインポート
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from esn_lab.model.esn import ESN


def parse_weight_filename(weight_path: Path) -> dict:
    """重みファイル名からモデルパラメータを抽出
    
    Args:
        weight_path: 重みファイルのパス
        
    Returns:
        パラメータ辞書 (Nx, density, input_scale, rho)
    """
    filename = weight_path.stem  # 拡張子を除いたファイル名
    
    # パターン: seed-{seed}_nx-{Nx}_density-{density}_input_scale-{input_scale}_rho-{rho}_{folds}_Wout
    pattern = r'seed-\w+_nx-(\d+)_density-(\d+)_input_scale-(\d+)_rho-(\d+)_[a-z]+_Wout'
    match = re.match(pattern, filename)
    
    if not match:
        raise ValueError(f"重みファイル名の形式が不正です: {filename}")
    
    nx = int(match.group(1))
    density_str = match.group(2)
    input_scale_str = match.group(3)
    rho_str = match.group(4)
    
    # 小数点を復元（例: "05" -> 0.5, "0001" -> 0.001）
    # ファイル名は "0.5" -> "05", "0.001" -> "0001" のように変換されているので
    # 最初の文字の後に小数点を挿入して復元
    def restore_decimal(s: str) -> float:
        if len(s) == 0:
            return 0.0
        return float(s[0] + '.' + s[1:]) if len(s) > 1 else float(s)
    
    density = restore_decimal(density_str)
    input_scale = restore_decimal(input_scale_str)
    rho = restore_decimal(rho_str)
    
    return {
        'Nx': nx,
        'density': density,
        'input_scale': input_scale,
        'rho': rho
    }


def load_single_sample(npz_path: Path, sample_index: int) -> tuple[str, np.ndarray, int]:
    """指定されたNPZファイルから1サンプルを読み込み
    
    Args:
        npz_path: NPZファイルのパス
        sample_index: サンプルのインデックス
        
    Returns:
        (sample_id, sample_data, class_id)
    """
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZファイルが見つかりません: {npz_path}")
    
    data = np.load(npz_path, allow_pickle=True)
    
    # サンプル数を確認
    num_samples = int(data['num_samples'])
    if sample_index >= num_samples:
        raise ValueError(
            f"サンプルインデックス {sample_index} が範囲外です "
            f"(利用可能: 0-{num_samples-1})"
        )
    
    sample_data = np.array(data[f'sample_{sample_index}_data'], copy=True)
    sample_class = int(data[f'sample_{sample_index}_class'])
    sample_id = str(data[f'sample_{sample_index}_id'])
    
    return sample_id, sample_data, sample_class


def predict_with_output_recording(
    model: ESN,
    U: np.ndarray,
    Ny: int
) -> np.ndarray:
    """1サンプルの推論を実行し、出力ノード値を記録
    
    Args:
        model: ESNモデル
        U: 入力時系列データ (T, Nu)
        Ny: 出力次元数
        
    Returns:
        出力ノード値の時系列 (T, Ny)
    """
    # リザバー状態をリセット
    model.Reservoir.reset_reservoir_state()
    model.y_prev = np.zeros(Ny)
    
    T = len(U)
    Y_pred = np.zeros((T, Ny))
    
    # 時間発展
    for t in range(T):
        x_in = model.Input(U[t])
        x = model.Reservoir(x_in)
        y_pred = model.Output(x)
        Y_pred[t] = y_pred
        model.y_prev = y_pred
    
    return Y_pred


def plot_onehot_output(
    Y_onehot: np.ndarray,
    sample_id: str,
    class_id: int,
    output_path: Path
) -> None:
    """One-hot化された出力からアクティブなインデックスを折れ線グラフでプロット
    
    Args:
        Y_onehot: One-hot化された出力 (T, Ny)
        sample_id: サンプルID
        class_id: クラスID
        output_path: 保存先パス
    """
    T, Ny = Y_onehot.shape
    
    # 各時刻でアクティブなインデックスを取得
    active_indices = np.argmax(Y_onehot, axis=1)
    
    # 各クラスの出現回数をカウント
    counts = np.bincount(active_indices, minlength=3)
    count_other = int(counts[0])
    count_foraging = int(counts[1])
    count_rumination = int(counts[2])
    
    # プロット用にインデックスを変換（foraging と rumination を入れ替え）
    # 元: 0=other, 1=foraging, 2=rumination
    # 変換後: 0=other(下), 1=rumination(中), 2=foraging(上)
    index_mapping = {0: 0, 1: 2, 2: 1}
    plot_indices_mapped = np.array([index_mapping[idx] for idx in active_indices])
    
    # クラスラベルの定義（変換後の位置に対応）
    class_labels = ['other', 'rumination', 'foraging']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 散布図としてプロット（10点に1回プロット）
    plot_step = 20
    plot_indices = range(0, T, plot_step)
    ax.scatter(plot_indices, plot_indices_mapped[plot_indices], s=20, color='darkblue', alpha=0.6)
    
    # 軸ラベルとタイトル
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Behavior Class', fontsize=12)
    ax.set_title(
        f'Behavior Prediction over Time\nSample: {sample_id}, True Class: {class_id}',
        fontsize=14,
        fontweight='bold'
    )
    
    # Y軸の設定（クラスラベルを表示）
    ax.set_yticks(range(Ny))
    ax.set_yticklabels(class_labels)
    ax.set_ylim(-0.5, Ny - 0.5)
    
    # グリッド線を追加
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # カウント情報をテキストボックスで表示
    count_text = (
        f'Prediction Counts:\n'
        f'  other:      {count_other:5d} ({count_other/T*100:5.1f}%)\n'
        f'  foraging:   {count_foraging:5d} ({count_foraging/T*100:5.1f}%)\n'
        f'  rumination: {count_rumination:5d} ({count_rumination/T*100:5.1f}%)'
    )
    ax.text(0.02, 0.98, count_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # コンソールにもカウント情報を出力
    print(f"  [Prediction Counts]")
    print(f"    other:      {count_other:5d} ({count_other/T*100:5.1f}%)")
    print(f"    foraging:   {count_foraging:5d} ({count_foraging/T*100:5.1f}%)")
    print(f"    rumination: {count_rumination:5d} ({count_rumination/T*100:5.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description='学習済み重みを用いて単一サンプルの推論を実行し、出力ノード値を記録します'
    )
    parser.add_argument(
        '--npz_path',
        type=str,
        required=True,
        help='入力データのNPZファイルパス（例: dataset/10fold_npy_div/a/fold_b.npz）'
    )
    parser.add_argument(
        '--sample_index',
        type=int,
        required=True,
        help='推論するサンプルのインデックス（0から始まる）'
    )
    parser.add_argument(
        '--weight_path',
        type=str,
        required=True,
        help='学習済み重みファイルのパス（例: outputs/10fold_pred0/seed-nonseed_nx-200_...npy）'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='出力先ディレクトリ（例: outputs/prediction_outputs）'
    )
    parser.add_argument(
        '--Nu',
        type=int,
        default=256,
        help='入力次元数（デフォルト: 256）'
    )
    parser.add_argument(
        '--Ny',
        type=int,
        default=3,
        help='出力次元数（デフォルト: 3）'
    )
    
    args = parser.parse_args()
    
    # パスをPathオブジェクトに変換
    npz_path = Path(args.npz_path)
    weight_path = Path(args.weight_path)
    output_dir = Path(args.output_dir)
    
    # 重みファイルが存在するか確認
    if not weight_path.exists():
        raise FileNotFoundError(f"重みファイルが見つかりません: {weight_path}")
    
    # 重みファイル名からモデルパラメータを抽出
    print("[INFO] 重みファイルからパラメータを抽出...")
    params = parse_weight_filename(weight_path)
    print(f"  - Nx: {params['Nx']}")
    print(f"  - density: {params['density']}")
    print(f"  - input_scale: {params['input_scale']}")
    print(f"  - rho: {params['rho']}")
    print(f"  - Nu: {args.Nu}")
    print(f"  - Ny: {args.Ny}")
    
    # データロード
    print(f"\n[INFO] データをロード中: {npz_path}")
    sample_id, U, class_id = load_single_sample(npz_path, args.sample_index)
    print(f"  - Sample ID: {sample_id}")
    print(f"  - Class ID: {class_id}")
    print(f"  - Data shape: {U.shape}")
    
    # モデル構築
    print("\n[INFO] モデルを構築中...")
    model = ESN(
        N_u=args.Nu,
        N_y=args.Ny,
        N_x=params['Nx'],
        density=params['density'],
        input_scale=params['input_scale'],
        rho=params['rho']
    )
    
    # 重みロード
    print(f"[INFO] 重みをロード中: {weight_path.name}")
    weight = np.load(weight_path, allow_pickle=True)
    model.Output.setweight(weight)
    print(f"  - Weight shape: {weight.shape}")
    
    # 推論実行
    print("\n[INFO] 推論を実行中...")
    Y_pred = predict_with_output_recording(model, U, args.Ny)
    print(f"  - Output shape: {Y_pred.shape}")
    
    # 結果保存
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. そのままの出力値を保存
    output_filename = f"{sample_id}_output.csv"
    output_path = output_dir / output_filename
    
    df = pd.DataFrame(Y_pred, columns=[f'output_{i}' for i in range(args.Ny)])
    df.insert(0, 'time_step', range(len(Y_pred)))
    df.to_csv(output_path, index=False)
    
    # 2. One-hot化したデータを保存
    onehot_filename = f"{sample_id}_output_onehot.csv"
    onehot_path = output_dir / onehot_filename
    
    # 各時刻で最大値のインデックスを取得してone-hot化
    Y_onehot = np.zeros_like(Y_pred, dtype=int)
    max_indices = np.argmax(Y_pred, axis=1)
    Y_onehot[np.arange(len(Y_pred)), max_indices] = 1
    
    df_onehot = pd.DataFrame(Y_onehot, columns=[f'output_{i}' for i in range(args.Ny)])
    df_onehot.insert(0, 'time_step', range(len(Y_onehot)))
    df_onehot.to_csv(onehot_path, index=False)
    
    # 3. One-hot出力のグラフを保存
    plot_filename = f"{sample_id}_output_onehot.png"
    plot_path = output_dir / plot_filename
    
    print("\n[INFO] One-hot出力をプロット中...")
    plot_onehot_output(Y_onehot, sample_id, class_id, plot_path)
    
    print(f"\n[SUCCESS] 推論完了")
    print(f"  - そのままの出力: {output_path}")
    print(f"  - One-hot化出力: {onehot_path}")
    print(f"  - グラフ画像: {plot_path}")
    print(f"  - データ形状: {Y_pred.shape}")
    print(f"  - 出力値の範囲: [{Y_pred.min():.4f}, {Y_pred.max():.4f}]")


if __name__ == "__main__":
    main()
