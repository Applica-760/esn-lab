import re
from pathlib import Path
from esn_lab.model.model_builder import get_model_param_str

def make_weight_filename(cfg, overrides: dict | None, train_tag: str) -> str:
    """共有の命名規則で重みファイル名を生成する。

    例) seed-nonseed_nx-200_density-05_input_scale-0001_rho-09_abcdefghi_Wout.npy
    """
    prefix = get_model_param_str(cfg=cfg, overrides=overrides)
    return f"{prefix}_{train_tag}_Wout.npy"

def _decode_decimal_token(token: str) -> float:
    """'.' を除去して符号化された10進数トークンを復元する。

    ルール: int(token) / (10 ** (len(token) - 1))
    例: "09"->0.9, "095"->0.95, "0005"->0.0005
    """
    if not token or not token.isdigit():
        raise ValueError(f"Invalid decimal token: {token}")
    return int(token) / (10 ** (len(token) - 1))

def parse_weight_filename(path: str | Path) -> tuple[dict, str]:
    """重みファイル名から(overrides, train_tag)を復元する。

    期待形式 (stem):
    seed-<seedid>_nx-<Nx>_density-<dd>_input_scale-<ii>_rho-<rr>_<trainletters>_Wout
    dd/ii/rr は小数点を除去した表現。
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
