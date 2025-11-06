"""
重みファイルの管理ユーティリティ

ESN訓練における重みファイルの命名、保存先決定、存在確認、スキップ判定を行う。
"""
import re
from pathlib import Path
from typing import Literal

from esn_lab.model.model_builder import get_model_param_str


# ===============================
# 命名規則関連
# ===============================

def make_weight_filename(cfg, overrides: dict | None, train_tag: str) -> str:
    """共有の命名規則で重みファイル名を生成する。

    例) seed-nonseed_nx-200_density-05_input_scale-0001_rho-09_abcdefghi_Wout.npy
    
    Args:
        cfg: 設定オブジェクト
        overrides: ハイパーパラメータの上書き辞書
        train_tag: 訓練に使用したfoldのタグ（例: "abcdefghi"）
    
    Returns:
        str: 重みファイル名
    """
    prefix = get_model_param_str(cfg=cfg, overrides=overrides)
    return f"{prefix}_{train_tag}_Wout.npy"


def _decode_decimal_token(token: str) -> float:
    """'.' を除去して符号化された10進数トークンを復元する。

    ルール: int(token) / (10 ** (len(token) - 1))
    例: "09"->0.9, "095"->0.95, "0005"->0.0005
    
    Args:
        token: 小数点を除去した数値文字列
    
    Returns:
        float: 復元された小数
    
    Raises:
        ValueError: トークンが不正な場合
    """
    if not token or not token.isdigit():
        raise ValueError(f"Invalid decimal token: {token}")
    return int(token) / (10 ** (len(token) - 1))


def parse_weight_filename(path: str | Path) -> tuple[dict, str]:
    """重みファイル名から(overrides, train_tag)を復元する。

    期待形式 (stem):
    seed-<seedid>_nx-<Nx>_density-<dd>_input_scale-<ii>_rho-<rr>_<trainletters>_Wout
    dd/ii/rr は小数点を除去した表現。
    
    Args:
        path: 重みファイルのパス
    
    Returns:
        tuple[dict, str]: (ハイパーパラメータ上書き辞書, 訓練タグ)
    
    Raises:
        ValueError: ファイル名が期待される形式でない場合
    
    Examples:
        >>> overrides, tag = parse_weight_filename("seed-nonseed_nx-200_density-05_input_scale-0001_rho-09_abcdefghi_Wout.npy")
        >>> overrides
        {'Nx': 200, 'density': 0.5, 'input_scale': 0.0001, 'rho': 0.9}
        >>> tag
        'abcdefghi'
    """
    p = Path(path)
    stem = p.stem
    if not stem.endswith("_Wout"):
        raise ValueError(f"Unexpected weight filename (no _Wout suffix): {p.name}")

    pat = re.compile(
        r"seed-(?P<seed>[^_]+)"  # seedはoverridesには使わない
        r"_nx-(?P<nx>\d+)"
        r"_density-(?P<density>\d+)"
        r"_input_scale-(?P<input>\d+)"
        r"_rho-(?P<rho>\d+)"
        r"_(?P<train>[a-j]{9})"
        r"_Wout$"
    )
    m = pat.match(stem)
    if not m:
        raise ValueError(f"Unexpected weight filename format: {p.name}")

    nx = int(m.group("nx"))
    density = _decode_decimal_token(m.group("density"))
    input_scale = _decode_decimal_token(m.group("input"))
    rho = _decode_decimal_token(m.group("rho"))
    train_tag = m.group("train")

    overrides = {"Nx": nx, "density": density, "input_scale": input_scale, "rho": rho}
    return overrides, train_tag


# ===============================
# WeightManager クラス
# ===============================

class WeightManager:
    """重みファイルの管理を担当するクラス
    
    重みファイルの保存先決定、既存ファイルの確認、
    スキップ判定ロジックを提供する。
    """

    def __init__(self, weight_dir: Path):
        """
        Args:
            weight_dir: 重みファイルを保存するディレクトリ
        """
        self.weight_dir = Path(weight_dir).expanduser().resolve()
        self.weight_dir.mkdir(parents=True, exist_ok=True)

    def get_weight_path(
        self,
        cfg,
        hp_overrides: dict,
        train_tag: str,
    ) -> Path:
        """重みファイルの保存先パスを取得
        
        Args:
            cfg: 設定オブジェクト
            hp_overrides: ハイパーパラメータの上書き
            train_tag: 訓練に使用したfoldのタグ（例: "abcdefghi"）
        
        Returns:
            Path: 重みファイルの絶対パス
        """
        filename = make_weight_filename(cfg=cfg, overrides=hp_overrides, train_tag=train_tag)
        return self.weight_dir / filename

    def should_skip_fold(
        self,
        cfg,
        hp_overrides: dict,
        all_letters: list[str],
        leave_out_letter: str,
        skip_mode: Literal["never", "if_exists", "force_retrain"] = "if_exists",
    ) -> tuple[bool, Path]:
        """指定されたfoldをスキップすべきか判定
        
        Args:
            cfg: 設定オブジェクト
            hp_overrides: ハイパーパラメータの上書き
            all_letters: 全fold ID一覧
            leave_out_letter: テスト用fold ID（学習から除外）
            skip_mode: スキップモード
                - "never": 常に実行（既存ファイルを上書き）
                - "if_exists": 既存ファイルがあればスキップ（デフォルト）
                - "force_retrain": 常に実行（既存ファイルを明示的に上書き）
        
        Returns:
            tuple[bool, Path]: (スキップすべきか, 重みファイルパス)
        """
        train_letters = [x for x in all_letters if x != leave_out_letter]
        train_tag = "".join(train_letters)
        weight_path = self.get_weight_path(cfg, hp_overrides, train_tag)

        # skip_mode に応じた判定
        if skip_mode == "force_retrain":
            # 強制再訓練: 常に実行
            if weight_path.exists():
                print(f"[FORCE] Retraining fold '{leave_out_letter}' (existing weight will be overwritten): {weight_path.name}")
            return False, weight_path

        elif skip_mode == "never":
            # スキップなし: 常に実行
            return False, weight_path

        elif skip_mode == "if_exists":
            # 既存ファイルがあればスキップ
            if weight_path.exists():
                print(f"[SKIP] Weight file found, skipping fold '{leave_out_letter}': {weight_path.name}")
                return True, weight_path
            else:
                return False, weight_path

        else:
            raise ValueError(f"Unknown skip_mode: {skip_mode}")

    def determine_tasks_to_run(
        self,
        cfg,
        hp_overrides: dict,
        all_letters: list[str],
        skip_mode: Literal["never", "if_exists", "force_retrain"] = "if_exists",
    ) -> list[str]:
        """実行すべきfold（タスク）のリストを決定
        
        Args:
            cfg: 設定オブジェクト
            hp_overrides: ハイパーパラメータの上書き
            all_letters: 全fold ID一覧
            skip_mode: スキップモード（should_skip_foldと同じ）
        
        Returns:
            list[str]: 実行すべきfold IDのリスト
        """
        tasks_to_run = []

        for leave_out_letter in all_letters:
            should_skip, _ = self.should_skip_fold(
                cfg, hp_overrides, all_letters, leave_out_letter, skip_mode
            )
            if not should_skip:
                tasks_to_run.append(leave_out_letter)

        return tasks_to_run
