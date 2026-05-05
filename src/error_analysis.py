from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data import LABEL_ID_TO_NAME
from src.utils import ensure_dir


def build_error_analysis(pred_export: pd.DataFrame) -> pd.DataFrame:
    df = pred_export.copy()
    df["is_correct"] = df["label_id"].astype(int) == df["pred_label_id"].astype(int)
    df["pred_label"] = df["pred_label_id"].map(LABEL_ID_TO_NAME)
    df["error_type"] = "correct"
    df.loc[(df["label_id"] == 0) & (df["pred_label_id"] == 1), "error_type"] = "false_positive"
    df.loc[(df["label_id"] == 1) & (df["pred_label_id"] == 0), "error_type"] = "false_negative"
    errors = df[~df["is_correct"]].copy()
    if len(errors) == 0:
        errors = df.sort_values("confidence", ascending=False).head(10).copy()
        errors["note"] = "No errors found in this small run; inspect high-confidence predictions instead."
    else:
        errors["note"] = ""
        errors = errors.sort_values("confidence", ascending=False)
    keep_cols = [
        "text", "label", "label_id", "pred_label", "pred_label_id",
        "prob_negative", "prob_positive", "confidence", "error_type", "note",
    ]
    return errors[[c for c in keep_cols if c in errors.columns]].reset_index(drop=True)


def save_error_analysis(errors: pd.DataFrame, out_dir: str | Path, min_expected: int = 10) -> None:
    out_dir = ensure_dir(out_dir)
    errors.to_csv(out_dir / "error_analysis.csv", index=False)
    lines = [
        "# Error Analysis Summary",
        "",
        f"- Number of rows in error file: {len(errors)}",
        f"- Minimum rows students should discuss: {min_expected}",
        "",
        "## Error type counts",
        "",
    ]
    if "error_type" in errors.columns:
        counts = errors["error_type"].value_counts().to_dict()
        for key, val in counts.items():
            lines.append(f"- `{key}`: {val}")
    lines += [
        "",
        "## Student notes",
        "Điền nhận xét: các lỗi có liên quan tới câu dài, phủ định, chuyển ý, mỉa mai, hoặc từ hiếm không?",
    ]
    (out_dir / "error_analysis_summary.md").write_text("\n".join(lines), encoding="utf-8")
