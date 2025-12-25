from __future__ import annotations

import pandas as pd


def apply_filters(df: pd.DataFrame, filters: dict | None) -> pd.DataFrame:
    """Apply simple equality-ish filters to a DataFrame.

    - Numeric columns compared with absolute tolerance 1e-9
    - Non-numeric compared by string equality
    - Missing columns are reported and skipped
    """
    if not filters:
        return df
    out = df.copy()
    for key, val in filters.items():
        if key not in out.columns:
            print(f"[WARN] Filter column not found: {key}. Skipped.")
            continue
        series = out[key]
        if pd.api.types.is_numeric_dtype(series) and isinstance(val, (int, float)):
            tol = 1e-9
            out = out[(series - float(val)).abs() < tol]
        else:
            out = out[series.astype(str) == str(val)]
    return out
