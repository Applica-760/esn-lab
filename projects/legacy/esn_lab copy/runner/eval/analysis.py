from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2

from esn_lab.setup.config import Config
from esn_lab.utils.eval_utils import apply_filters
from esn_lab.model.model_builder import get_model_param_str
from esn_lab.pipeline.data import CSVDataLoader


def _extract_image_features(img_path: str) -> dict:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {
            "mean_intensity": None,
            "std_intensity": None,
            "edge_magnitude": None,
            "variance": None,
        }
    
    # Basic statistics
    mean_val = float(np.mean(img))
    std_val = float(np.std(img))
    var_val = float(np.var(img))
    
    # Edge detection (Sobel)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    edge_mag = float(np.mean(np.sqrt(sobelx**2 + sobely**2)))
    
    return {
        "mean_intensity": mean_val,
        "std_intensity": std_val,
        "edge_magnitude": edge_mag,
        "variance": var_val,
    }


def _create_image_grid(
    sample_ids: list[str],
    sample_id_to_path: dict[str, str],
    grid_rows: int,
    grid_cols: int,
    title: str,
    output_path: Path,
    dpi: int = 150,
    sample_metadata: dict[str, dict] | None = None,
):
    """Create a grid of images from sample IDs.
    
    Args:
        sample_ids: List of sample IDs to display
        sample_id_to_path: Mapping from sample_id to image file path
        grid_rows: Number of rows in grid
        grid_cols: Number of columns in grid
        title: Title for the plot
        output_path: Where to save the figure
        dpi: DPI for saving
        sample_metadata: Optional dict mapping sample_id to metadata dict with keys like
                        'success_count', 'fail_count', 'included_count', 'success_rate', 'expected_label'
    """
    n_samples = min(len(sample_ids), grid_rows * grid_cols)
    if n_samples == 0:
        print(f"[WARN] No samples to display for grid: {title}")
        return
    
    # ラベル名のマッピング
    label_names = {0: "other", 1: "foraging", 2: "rumination"}
    
    # 各セルを256×512の横長に (縦横比2:1)
    cell_width = 4  # インチ
    cell_height = 2  # インチ
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * cell_width, grid_rows * cell_height))
    if grid_rows == 1 and grid_cols == 1:
        axes = np.array([[axes]])
    elif grid_rows == 1 or grid_cols == 1:
        axes = axes.reshape(grid_rows, grid_cols)
    
    for idx in range(grid_rows * grid_cols):
        row = idx // grid_cols
        col = idx % grid_cols
        ax = axes[row, col]
        
        if idx < n_samples:
            sid = sample_ids[idx]
            img_path = sample_id_to_path.get(sid)
            if img_path and Path(img_path).exists():
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    ax.imshow(img, cmap='gray', aspect='auto')
                    # タイトルにサンプルIDとメタ情報を表示
                    title_parts = [f"{sid[:8]}..."]
                    if sample_metadata and sid in sample_metadata:
                        meta = sample_metadata[sid]
                        succ = meta.get('success_count', 0)
                        inc = meta.get('included_count', 0)
                        if inc > 0:
                            title_parts.append(f"({succ}/{inc})")
                        
                        # 正解ラベルの情報を追加
                        expected_label = meta.get('expected_label')
                        if expected_label is not None and not pd.isna(expected_label):
                            try:
                                label_id = int(expected_label)
                                label_name = label_names.get(label_id, f"label{label_id}")
                                
                                # Always-failサンプルの場合、誤分類パターンを表示
                                if succ == 0 and inc > 0 and "misclassification_pattern" in meta:
                                    mis_pattern = meta["misclassification_pattern"]
                                    # 予測ラベルを出現回数の多い順にソート
                                    sorted_preds = sorted(mis_pattern.items(), key=lambda x: x[1], reverse=True)
                                    pred_strs = [f"{label_names.get(pred_id, f'label{pred_id}')}:{count}" 
                                                for pred_id, count in sorted_preds]
                                    title_parts.append(f"[true:{label_name}, pred:{','.join(pred_strs)}]")
                                else:
                                    title_parts.append(f"[{label_name}]")
                            except (ValueError, TypeError):
                                pass
                    ax.set_title(" ".join(title_parts), fontsize=8)
                else:
                    ax.text(0.5, 0.5, "Read Error", ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, "Not Found", ha='center', va='center', transform=ax.transAxes)
        
        ax.axis('off')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"[ARTIFACT] Saved image grid: {output_path}")


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

    # パス解決の優先順位:
    # 1. predictions_csv が明示的に指定されている
    # 2. experiment_name から自動補完（推奨）
    
    predictions_csv_explicit = getattr(ana_cfg, "predictions_csv", None)
    experiment_name = getattr(ana_cfg, "experiment_name", None)
    
    # predictions_csv の決定
    if predictions_csv_explicit:
        preds_csv = Path(predictions_csv_explicit).expanduser().resolve()
        out_root = preds_csv.parent
    elif experiment_name:
        exp_base = Path("outputs/experiments") / experiment_name / "eval"
        preds_csv = (exp_base / "evaluation_predictions.csv").resolve()
        out_root = exp_base.resolve()
        print(f"[INFO] Using experiment: {experiment_name}")
    else:
        raise ValueError("Config requires either 'predictions_csv' or 'experiment_name'.")
    
    out_root.mkdir(parents=True, exist_ok=True)
    
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
            rec["true_label"] = int(true_vals.iloc[0]) if len(true_vals) else None
            rec["true_label_consistent"] = bool(true_vals.nunique() <= 1)
            # expected_labelがない場合、true_labelを使用
            if not has_expected:
                rec["expected_label"] = rec["true_label"]

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

    # ==============================
    # 新機能: 誤判別の一貫性分析（Always-failサンプル限定）
    # ==============================
    try:
        from collections import Counter
        
        # ラベルID -> 名前のマッピング（0:other, 1:foraging, 2:rumination）
        label_names_map = {0: "other", 1: "foraging", 2: "rumination"}
        
        # Always-failサンプルについて誤判別パターンを集計
        misclass_rows = []
        for idx, row in always_fail.iterrows():
            sid = row["sample_id"]
            expected = row.get("expected_label")
            pred_by_fold_str = row.get("pred_by_fold", "")
            used_count = int(row["included_count"])
            
            if pd.isna(expected) or not pred_by_fold_str:
                continue
            
            expected = int(expected)
            
            # "a:1,b:2,c:0" -> [1, 2, 0]
            try:
                predictions = [int(p.split(":")[1]) for p in pred_by_fold_str.split(",") if ":" in p]
            except (ValueError, IndexError):
                continue
            
            # Always-failなので全て誤判別（念のため expected と異なるもののみカウント）
            misclassifications = [p for p in predictions if p != expected]
            
            # クラス別カウント
            mis_counts = Counter(misclassifications)
            
            # 一貫性スコア: 最も多い誤判別 ÷ 総誤判別回数
            total_misclass = len(misclassifications)
            max_misclass = max(mis_counts.values()) if mis_counts else 0
            consistency_score = max_misclass / total_misclass if total_misclass > 0 else 0.0
            
            misclass_rows.append({
                "sample_id": sid,
                "used_count": used_count,
                "true_label": label_names_map.get(expected, f"label{expected}"),
                "foraging": mis_counts.get(1, 0),
                "rumination": mis_counts.get(2, 0),
                "other": mis_counts.get(0, 0),
                "consistency_score": consistency_score,
            })
        
        if misclass_rows:
            # DataFrameに変換
            misclass_df = pd.DataFrame(misclass_rows)
            
            # ソート: 使用回数降順 → 一貫性スコア降順
            misclass_df = misclass_df.sort_values(
                ["used_count", "consistency_score"],
                ascending=[False, False]
            ).reset_index(drop=True)
            
            # インデックスを1始まりに
            misclass_df.index = misclass_df.index + 1
            
            # CSV出力
            misclass_csv = csv_dir / f"analysis_misclassification_consistency_{param_tag}.csv"
            misclass_df.to_csv(misclass_csv, index=True, index_label="index")
            print(f"[ARTIFACT] Saved misclassification consistency CSV: {misclass_csv}")
            
            # テーブル画像の作成
            fig, ax = plt.subplots(figsize=(12, max(6, len(misclass_df) * 0.3)))
            ax.axis('tight')
            ax.axis('off')
            
            # 表示用にconsistency_scoreを除外して整形
            table_data = misclass_df[["sample_id", "used_count", "true_label", "foraging", "rumination", "other"]].copy()
            
            # sample_idを短縮表示（最初8文字）
            table_data["sample_id"] = table_data["sample_id"].apply(lambda x: f"{x[:12]}..." if len(x) > 12 else x)
            
            # ヘッダー
            col_labels = ["Index", "Sample ID", "Used", "True Label", "Foraging", "Rumination", "Other"]
            
            # データを2次元リストに変換（インデックス付き）
            table_values = []
            for idx, row in table_data.iterrows():
                table_values.append([
                    str(idx),
                    row["sample_id"],
                    str(row["used_count"]),
                    row["true_label"],
                    str(row["foraging"]),
                    str(row["rumination"]),
                    str(row["other"]),
                ])
            
            # 表示件数を制限（大きすぎる場合）
            max_display = 50
            if len(table_values) > max_display:
                table_values = table_values[:max_display]
                table_note = f"(Showing top {max_display} of {len(misclass_df)} samples)"
            else:
                table_note = f"(Total: {len(misclass_df)} samples)"
            
            table = ax.table(
                cellText=table_values,
                colLabels=col_labels,
                cellLoc='center',
                loc='center',
                colWidths=[0.0512, 0.128, 0.0512, 0.096, 0.0832, 0.0832, 0.0832],
            )
            
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.5)
            
            # ヘッダー行のスタイル
            for i in range(len(col_labels)):
                cell = table[(0, i)]
                cell.set_facecolor('#4472C4')
                cell.set_text_props(weight='bold', color='white')
            
            # データ行の交互背景色
            for i in range(1, len(table_values) + 1):
                for j in range(len(col_labels)):
                    cell = table[(i, j)]
                    if i % 2 == 0:
                        cell.set_facecolor('#F2F2F2')
            
            fig.suptitle(f"Misclassification Consistency Analysis (Always-Fail Samples)\n{table_note}", 
                        fontsize=12, fontweight='bold', y=0.98)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            
            misclass_table_png = images_dir / f"analysis_misclassification_table_{param_tag}.png"
            fig.savefig(misclass_table_png, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            print(f"[ARTIFACT] Saved misclassification table image: {misclass_table_png}")
        else:
            print("[INFO] No always-fail samples found for misclassification consistency analysis")
    
    except Exception as e:
        print(f"[WARN] Misclassification consistency analysis failed: {e}")

    # ==============================
    # 新機能1: 成功率スペクトラム画像グリッド表示
    # ==============================
    if ana_cfg.csv_dir and ana_cfg.export_images:
        try:
            csv_dir = Path(ana_cfg.csv_dir).expanduser().resolve()
            data_loader = CSVDataLoader(csv_dir)
            all_fold_ids = data_loader.get_available_folds()
            
            # sample_id -> image_path のマッピングを作成（注: 画像パス情報を取得する必要がある）
            sample_id_to_path = {}
            
            # CSVから画像パス情報を取得
            for fold_id in all_fold_ids:
                csv_path = csv_dir / f"10fold_{fold_id}.csv"
                df = pd.read_csv(csv_path, usecols=["file_path"])
                for file_path in df["file_path"]:
                    sid = Path(file_path).stem
                    sample_id_to_path[sid] = file_path
            
            # sample_id -> metadata のマッピングを作成
            sample_metadata = {}
            for idx, row in images_df.iterrows():
                sid = row["sample_id"]
                inc = row["included_count"]
                # expected_labelの取得（存在しない場合はNone）
                exp_label = row.get("expected_label") if "expected_label" in images_df.columns else None
                
                meta = {
                    "success_count": row["success_count"],
                    "fail_count": row["fail_count"],
                    "included_count": inc,
                    "success_rate": row["success_count"] / inc if inc > 0 else 0.0,
                    "expected_label": exp_label,
                }
                
                # Always-failサンプルの場合、誤分類パターンを解析
                if row["fail_count"] == inc and inc > 0 and "pred_by_fold" in row:
                    pred_by_fold_str = row.get("pred_by_fold", "")
                    if pred_by_fold_str and isinstance(pred_by_fold_str, str):
                        # "a:1,b:2,c:0" -> [1, 2, 0]
                        try:
                            predictions = [int(p.split(":")[1]) for p in pred_by_fold_str.split(",") if ":" in p]
                            if predictions:
                                from collections import Counter
                                pred_counts = Counter(predictions)
                                meta["misclassification_pattern"] = dict(pred_counts)
                                meta["dominant_misprediction"] = pred_counts.most_common(1)[0][0]
                        except (ValueError, IndexError) as e:
                            pass  # パース失敗時はスキップ
                
                sample_metadata[sid] = meta
                # デバッグ用（最初の3件のみ）
                if idx < 3:
                    print(f"[DEBUG] sample_id={sid}, expected_label={exp_label}")
            
            # 成功率でソート済みのimages_dfから各カテゴリを抽出
            fail_samples = always_fail["sample_id"].tolist()
            success_samples = always_success["sample_id"].tolist()
            
            # 混在データ (0 < success_count < included_count)
            mixed = images_df[
                (images_df["included_count"] > 0) &
                (images_df["success_count"] > 0) &
                (images_df["success_count"] < images_df["included_count"])
            ].copy()
            # 成功率が低い順にソート
            mixed["success_rate"] = mixed["success_count"] / mixed["included_count"]
            mixed = mixed.sort_values("success_rate")
            mixed_samples = mixed["sample_id"].tolist()
            
            # グリッドサイズ (各カテゴリから45枚ずつ表示: 3x3を5ページ)
            grid_size = 3  # 3x3
            n_per_grid = grid_size * grid_size  # 9
            n_total = n_per_grid * 5  # 45
            
            def _sample_diverse_by_trial_count(sample_ids: list[str], metadata: dict, n_target: int) -> list[str]:
                """試行回数が偏らないようにサンプリング。
                
                異なる試行回数のグループから均等に選択し、合計n_target件を返す。
                """
                if len(sample_ids) <= n_target:
                    return sample_ids
                
                # 試行回数ごとにグループ化
                by_count = {}
                for sid in sample_ids:
                    cnt = metadata.get(sid, {}).get("included_count", 0)
                    if cnt not in by_count:
                        by_count[cnt] = []
                    by_count[cnt].append(sid)
                
                # 試行回数の多い順にソート (信頼度の高いものを優先)
                sorted_counts = sorted(by_count.keys(), reverse=True)
                
                # ラウンドロビンで選択
                selected = []
                idx_map = {cnt: 0 for cnt in sorted_counts}
                
                while len(selected) < n_target:
                    added = False
                    for cnt in sorted_counts:
                        if idx_map[cnt] < len(by_count[cnt]):
                            selected.append(by_count[cnt][idx_map[cnt]])
                            idx_map[cnt] += 1
                            added = True
                            if len(selected) >= n_target:
                                break
                    if not added:  # すべてのグループを使い切った
                        break
                
                return selected
            
            # 全失敗データのグリッド (5ページ分)
            if fail_samples:
                fail_diverse = _sample_diverse_by_trial_count(fail_samples, sample_metadata, n_total)
                for page in range(5):
                    start_idx = page * n_per_grid
                    end_idx = start_idx + n_per_grid
                    page_samples = fail_diverse[start_idx:end_idx]
                    if page_samples:
                        _create_image_grid(
                            sample_ids=page_samples,
                            sample_id_to_path=sample_id_to_path,
                            grid_rows=grid_size,
                            grid_cols=grid_size,
                            title=f"Always-Fail Samples (Page {page+1}/5)",
                            output_path=images_dir / f"grid_always_fail_{param_tag}_p{page+1}.png",
                            dpi=dpi,
                            sample_metadata=sample_metadata,
                        )
            
            # 混在データのグリッド (低成功率側)
            if mixed_samples:
                mixed_diverse = _sample_diverse_by_trial_count(mixed_samples, sample_metadata, n_total)
                for page in range(5):
                    start_idx = page * n_per_grid
                    end_idx = start_idx + n_per_grid
                    page_samples = mixed_diverse[start_idx:end_idx]
                    if page_samples:
                        _create_image_grid(
                            sample_ids=page_samples,
                            sample_id_to_path=sample_id_to_path,
                            grid_rows=grid_size,
                            grid_cols=grid_size,
                            title=f"Mixed Samples (Low Success Rate, Page {page+1}/5)",
                            output_path=images_dir / f"grid_mixed_{param_tag}_p{page+1}.png",
                            dpi=dpi,
                            sample_metadata=sample_metadata,
                        )
            
            # 全成功データのグリッド (5ページ分)
            if success_samples:
                success_diverse = _sample_diverse_by_trial_count(success_samples, sample_metadata, n_total)
                for page in range(5):
                    start_idx = page * n_per_grid
                    end_idx = start_idx + n_per_grid
                    page_samples = success_diverse[start_idx:end_idx]
                    if page_samples:
                        _create_image_grid(
                            sample_ids=page_samples,
                            sample_id_to_path=sample_id_to_path,
                            grid_rows=grid_size,
                            grid_cols=grid_size,
                            title=f"Always-Success Samples (Page {page+1}/5)",
                            output_path=images_dir / f"grid_always_success_{param_tag}_p{page+1}.png",
                            dpi=dpi,
                            sample_metadata=sample_metadata,
                        )
            
            # スペクトラム表示: 成功率順に並べた総合グリッド (横長)
            spectrum_samples = images_df[images_df["included_count"] > 0].copy()
            spectrum_samples["success_rate"] = spectrum_samples["success_count"] / spectrum_samples["included_count"]
            spectrum_samples = spectrum_samples.sort_values("success_rate")
            spectrum_ids = spectrum_samples["sample_id"].tolist()
            
            # 等間隔でサンプリング (最大24枚: 3行x8列)
            spectrum_rows, spectrum_cols = 3, 8
            n_spectrum = spectrum_rows * spectrum_cols
            if len(spectrum_ids) > n_spectrum:
                indices = np.linspace(0, len(spectrum_ids) - 1, n_spectrum, dtype=int)
                spectrum_ids_sampled = [spectrum_ids[i] for i in indices]
            else:
                spectrum_ids_sampled = spectrum_ids
            
            if spectrum_ids_sampled:
                _create_image_grid(
                    sample_ids=spectrum_ids_sampled,
                    sample_id_to_path=sample_id_to_path,
                    grid_rows=spectrum_rows,
                    grid_cols=spectrum_cols,
                    title="Success Rate Spectrum (Low → High)",
                    output_path=images_dir / f"grid_spectrum_{param_tag}.png",
                    dpi=dpi,
                    sample_metadata=sample_metadata,
                )
            
        except Exception as e:
            print(f"[WARN] Image grid generation failed: {e}")
    
    # ==============================
    # 新機能2: 特徴量分析プロット
    # ==============================
    if ana_cfg.csv_dir:
        try:
            csv_dir = Path(ana_cfg.csv_dir).expanduser().resolve()
            data_loader = CSVDataLoader(csv_dir)
            all_fold_ids = data_loader.get_available_folds()
            
            # sample_id -> image_path のマッピング
            sample_id_to_path = {}
            for fold_id in all_fold_ids:
                csv_path = csv_dir / f"10fold_{fold_id}.csv"
                df = pd.read_csv(csv_path, usecols=["file_path"])
                for file_path in df["file_path"]:
                    sid = Path(file_path).stem
                    sample_id_to_path[sid] = file_path
            
            # 各サンプルの特徴量を抽出
            print("[INFO] Extracting image features for analysis...")
            feature_rows = []
            for _, row in images_df.iterrows():
                sid = row["sample_id"]
                img_path = sample_id_to_path.get(sid)
                if not img_path:
                    continue
                
                features = _extract_image_features(img_path)
                inc_count = row["included_count"]
                success_rate = row["success_count"] / inc_count if inc_count > 0 else None
                
                feature_rows.append({
                    "sample_id": sid,
                    "success_rate": success_rate,
                    "success_count": row["success_count"],
                    "fail_count": row["fail_count"],
                    "included_count": inc_count,
                    "expected_label": row.get("expected_label"),
                    **features,
                })
            
            features_df = pd.DataFrame(feature_rows)
            features_csv = csv_dir / f"analysis_features_{param_tag}.csv"
            features_df.to_csv(features_csv, index=False)
            print(f"[ARTIFACT] Saved features CSV: {features_csv}")
            
            # 特徴量の散布図プロット
            valid_features = features_df.dropna(subset=["mean_intensity", "std_intensity", "edge_magnitude", "success_rate"])
            
            if not valid_features.empty:
                # 2x2 サブプロット: 各特徴量 vs 成功率
                fig, axes = plt.subplots(2, 2, figsize=(10, 8))
                
                # クラス別に色分け (expected_labelがある場合)
                has_labels = "expected_label" in valid_features.columns and not valid_features["expected_label"].isna().all()
                if has_labels:
                    num_classes = int(valid_features["expected_label"].max() + 1) if valid_features["expected_label"].max() >= 0 else 3
                    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
                    color_map = {i: colors[i] for i in range(num_classes)}
                else:
                    color_map = None
                
                feature_pairs = [
                    ("mean_intensity", "Mean Intensity"),
                    ("std_intensity", "Contrast (Std Dev)"),
                    ("edge_magnitude", "Edge Magnitude"),
                    ("variance", "Variance"),
                ]
                
                for idx, (feat_col, feat_label) in enumerate(feature_pairs):
                    ax = axes[idx // 2, idx % 2]
                    
                    if has_labels and color_map:
                        for cls_id in sorted(valid_features["expected_label"].dropna().unique()):
                            cls_data = valid_features[valid_features["expected_label"] == cls_id]
                            ax.scatter(
                                cls_data[feat_col],
                                cls_data["success_rate"],
                                c=[color_map[int(cls_id)]],
                                label=f"Class {int(cls_id)}",
                                alpha=0.6,
                                s=30,
                            )
                        ax.legend(fontsize=8)
                    else:
                        ax.scatter(
                            valid_features[feat_col],
                            valid_features["success_rate"],
                            c=valid_features["success_rate"],
                            cmap="RdYlGn",
                            alpha=0.6,
                            s=30,
                        )
                    
                    ax.set_xlabel(feat_label, fontsize=10)
                    ax.set_ylabel("Success Rate", fontsize=10)
                    ax.set_ylim(-0.05, 1.05)
                    ax.grid(True, linestyle='--', alpha=0.3)
                
                fig.suptitle("Image Features vs Success Rate", fontsize=14, fontweight='bold')
                fig.tight_layout(rect=[0, 0, 1, 0.96])
                features_plot = images_dir / f"analysis_features_scatter_{param_tag}.png"
                fig.savefig(features_plot, dpi=dpi, bbox_inches='tight')
                plt.close(fig)
                print(f"[ARTIFACT] Saved feature scatter plot: {features_plot}")
                
                # ヒートマップ: 特徴量の相関
                feature_cols = ["mean_intensity", "std_intensity", "edge_magnitude", "variance", "success_rate"]
                corr_data = valid_features[feature_cols].dropna()
                if len(corr_data) > 1:
                    corr_matrix = corr_data.corr()
                    
                    fig, ax = plt.subplots(figsize=(7, 6))
                    im = ax.imshow(corr_matrix, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
                    ax.set_xticks(range(len(feature_cols)))
                    ax.set_yticks(range(len(feature_cols)))
                    ax.set_xticklabels([c.replace("_", " ").title() for c in feature_cols], rotation=45, ha="right", fontsize=9)
                    ax.set_yticklabels([c.replace("_", " ").title() for c in feature_cols], fontsize=9)
                    
                    # 相関係数を表示
                    for i in range(len(feature_cols)):
                        for j in range(len(feature_cols)):
                            text = ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                                         ha="center", va="center", color="black", fontsize=9)
                    
                    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    cbar.set_label("Correlation", fontsize=10)
                    ax.set_title("Feature Correlation Matrix", fontsize=12, fontweight='bold')
                    fig.tight_layout()
                    corr_plot = images_dir / f"analysis_features_correlation_{param_tag}.png"
                    fig.savefig(corr_plot, dpi=dpi, bbox_inches='tight')
                    plt.close(fig)
                    print(f"[ARTIFACT] Saved correlation heatmap: {corr_plot}")
            
        except Exception as e:
            print(f"[WARN] Feature analysis failed: {e}")

    return
