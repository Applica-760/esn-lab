from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from esn_lab.setup.config import Config
from esn_lab.utils.eval_utils import apply_filters
from esn_lab.model.model_builder import get_model_param_str


def _make_param_tag(cfg: Config, filters: dict | None, df_cols: list[str]) -> str:
    """Create a stable tag for output files based on provided filters.

    If all of Nx/density/input_scale/rho are present in filters, reuse the
    existing get_model_param_str format (with seed placeholder).
    Otherwise, build a simple key-value tag from available filters that match df columns.
    """
    filters = filters or {}
    keys = ["Nx", "density", "input_scale", "rho"]
    if all(k in filters for k in keys):
        overrides = {k: filters[k] for k in keys}
        return get_model_param_str(cfg, overrides=overrides)

    # Fallback: only include keys that exist in the dataframe columns
    usable = [(k, filters[k]) for k in sorted(filters.keys()) if k in df_cols]
    if not usable:
        return "filters_none"

    def _fmt(v):
        # sanitize for filenames (remove dots)
        s = str(v)
        return s.replace(".", "")

    return "filters_" + "_".join([f"{k}-{_fmt(v)}" for k, v in usable])


def analysis_evaluate(cfg: Config):
    ana_cfg = cfg.evaluate.analysis if cfg.evaluate else None
    if ana_cfg is None:
        raise ValueError("Config 'cfg.evaluate.analysis' not found.")

    # Resolve tenfold_root (required) and evaluation root
    tenfold_root = getattr(ana_cfg, "tenfold_root", None)
    if not tenfold_root:
        raise ValueError("Config requires 'evaluate.analysis.tenfold_root'.")
    out_root = (Path(tenfold_root).expanduser() / "eval").resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    preds_name = ana_cfg.csv_name or "evaluation_predictions.csv"
    preds_csv = (out_root / preds_name).resolve()
    if not preds_csv.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {preds_csv}")

    df = pd.read_csv(preds_csv)

    # Apply filters to select the parameter set
    filters = dict(ana_cfg.filters) if ana_cfg.filters else {}
    df_f = apply_filters(df, filters)
    if df_f.empty:
        raise ValueError("No prediction rows remain after applying cfg.evaluate.analysis.filters")

    # Build output directory for this parameter tag
    param_tag = _make_param_tag(cfg, filters, df.columns.tolist())
    # 出力先は入力CSVと同階層（ユーザ要望）
    base_out_dir = Path(ana_cfg.output_dir).expanduser().resolve() if ana_cfg.output_dir else preds_csv.parent
    # Split outputs into csv/ and images/
    csv_dir = (base_out_dir / "csv").resolve()
    images_dir = (base_out_dir / "images").resolve()
    csv_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    # 画像ベース集計（foldベースではなくサンプル毎に、どのfoldで成功/失敗したかを記録）
    required_cols = {"sample_id", "holdout_fold", "majority_success"}
    missing = [c for c in required_cols if c not in df_f.columns]
    if missing:
        raise ValueError(f"Predictions CSV is missing required columns: {missing}")

    # 型の揺れ対策（True/False または 0/1）
    ms = df_f["majority_success"]
    if ms.dtype != bool:
        df_f["majority_success"] = ms.astype(int) > 0

    # 期待ラベルは基本一定のはずだが、万一揺れがあれば first を採用しフラグで通知
    has_expected = "expected_label" in df_f.columns
    has_true = "true_label" in df_f.columns
    has_pred = "pred_label" in df_f.columns

    rows = []
    for sid, sdf in df_f.groupby("sample_id", dropna=False):
        folds_all = sorted(sdf["holdout_fold"].astype(str).unique().tolist())
        folds_success = sorted(sdf.loc[sdf["majority_success"] == True, "holdout_fold"].astype(str).tolist())
        folds_fail = sorted(sdf.loc[sdf["majority_success"] == False, "holdout_fold"].astype(str).tolist())

        rec = {
            "sample_id": sid,
            "folds_included": ",".join(folds_all),
            "folds_success": ",".join(folds_success),
            "folds_fail": ",".join(folds_fail),
            "success_count": int(len(folds_success)),
            "fail_count": int(len(folds_fail)),
        }

        if has_expected:
            exp_vals = sdf["expected_label"].dropna().astype(int)
            rec["expected_label"] = int(exp_vals.iloc[0]) if len(exp_vals) else None
            rec["expected_label_consistent"] = bool(exp_vals.nunique() <= 1)

        if has_true:
            true_vals = sdf["true_label"].dropna().astype(int)
            rec["true_label_consistent"] = bool(true_vals.nunique() <= 1)

        if has_pred:
            # 参考情報: 各foldでの予測ラベルを a:2,c:0 のように格納
            pred_map = (
                sdf[["holdout_fold", "pred_label"]]
                .dropna()
                .assign(holdout_fold=lambda x: x["holdout_fold"].astype(str), pred_label=lambda x: x["pred_label"].astype(int))
            )
            pred_map = pred_map.sort_values("holdout_fold")
            rec["pred_by_fold"] = ",".join([f"{r.holdout_fold}:{r.pred_label}" for r in pred_map.itertuples(index=False)])

        rows.append(rec)

    images_df = pd.DataFrame(rows)
    images_df["included_count"] = images_df["success_count"].fillna(0).astype(int) + images_df["fail_count"].fillna(0).astype(int)
    images_df = images_df.sort_values(["success_count", "fail_count", "sample_id"], ascending=[True, False, True])

    images_csv = csv_dir / f"analysis_images_{param_tag}.csv"
    images_df.to_csv(images_csv, index=False)

    # 極端ケースの抽出: 全成功/全失敗
    always_fail = images_df[(images_df["included_count"] > 0) & (images_df["fail_count"] == images_df["included_count"])].copy()
    always_success = images_df[(images_df["included_count"] > 0) & (images_df["success_count"] == images_df["included_count"])].copy()

    always_fail_csv = csv_dir / f"analysis_always_fail_{param_tag}.csv"
    always_success_csv = csv_dir / f"analysis_always_success_{param_tag}.csv"
    always_fail.to_csv(always_fail_csv, index=False)
    always_success.to_csv(always_success_csv, index=False)

    # サマリーグラフの出力（画像データ自体のコピーは行わない）
    try:
        dpi = int(getattr(cfg.evaluate.analysis, "dpi", 150)) if getattr(cfg, "evaluate", None) else 150

        # ヒストグラム: サンプルごとの成功率分布
        sr_df = images_df[images_df["included_count"] > 0].copy()
        if not sr_df.empty:
            sr_df["success_rate"] = sr_df["success_count"] / sr_df["included_count"].replace(0, pd.NA)
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.hist(sr_df["success_rate"], bins=11, range=(0, 1), color="C0", edgecolor="white")
            ax.set_title("Per-sample success rate distribution")
            ax.set_xlabel("success rate")
            ax.set_ylabel("#samples")
            ax.grid(True, linestyle='--', alpha=0.4)
            fig.tight_layout()
            hist_png = images_dir / f"analysis_success_rate_hist_{param_tag}.png"
            fig.savefig(hist_png, dpi=dpi)
            plt.close(fig)
            print(f"[ARTIFACT] Saved histogram: {hist_png}")

        # バーチャート: 極端ケース数（全失敗/全成功/混在）
        total = int(images_df.shape[0])
        nf = int(always_fail.shape[0])
        ns = int(always_success.shape[0])
        nm = total - nf - ns
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        bars = ax.bar(["all-fail", "mixed", "all-success"], [nf, nm, ns], color=["#d62728", "#ffbf00", "#2ca02c"])
        ax.set_title("Counts of extreme cases")
        ax.set_ylabel("#samples")
        for b in bars:
            ax.annotate(str(int(b.get_height())), xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        fig.tight_layout()
        cnt_png = images_dir / f"analysis_extreme_counts_{param_tag}.png"
        fig.savefig(cnt_png, dpi=dpi)
        plt.close(fig)
        print(f"[ARTIFACT] Saved bar chart: {cnt_png} (total={total}, all-fail={nf}, mixed={nm}, all-success={ns})")

        # クラス別の極端ケース（expected_label があれば）
        if "expected_label" in images_df.columns:
            fig, ax = plt.subplots(figsize=(5, 3.2))
            grp_fail = always_fail.groupby("expected_label").size()
            grp_succ = always_success.groupby("expected_label").size()
            idx = sorted(set(grp_fail.index).union(set(grp_succ.index)))
            vals_fail = [int(grp_fail.get(i, 0)) for i in idx]
            vals_succ = [int(grp_succ.get(i, 0)) for i in idx]
            x = range(len(idx))
            w = 0.4
            ax.bar([i - w/2 for i in x], vals_fail, width=w, label="all-fail", color="#d62728")
            ax.bar([i + w/2 for i in x], vals_succ, width=w, label="all-success", color="#2ca02c")
            ax.set_xticks(list(x))
            ax.set_xticklabels([str(i) for i in idx])
            ax.set_xlabel("expected_label")
            ax.set_ylabel("#samples")
            ax.set_title("Extreme cases by class")
            ax.legend()
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            fig.tight_layout()
            cls_png = images_dir / f"analysis_extreme_by_class_{param_tag}.png"
            fig.savefig(cls_png, dpi=dpi)
            plt.close(fig)
            print(f"[ARTIFACT] Saved class breakdown: {cls_png}")

    except Exception as e:
        print(f"[WARN] Plotting step failed: {e}")

    print(f"[INFO] Wrote analysis outputs: {images_csv}, {always_fail_csv}, {always_success_csv}")

    return
