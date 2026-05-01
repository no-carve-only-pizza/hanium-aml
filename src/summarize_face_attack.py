"""Summarize targeted face attack metadata files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize targeted face attack runs.")
    parser.add_argument("--metadata-dir", type=Path, default=Path("outputs/attacks"))
    parser.add_argument("--out", type=Path, default=Path("outputs/attacks/face_attack_summary.csv"))
    args = parser.parse_args()

    paths = sorted(args.metadata_dir.glob("**/metadata_targeted*.csv"))
    if not paths:
        raise FileNotFoundError(f"No metadata files found in {args.metadata_dir}")

    rows = []
    for path in paths:
        df = pd.read_csv(path)
        clean_correct = df["pred_before"] == df["true_label"] if "clean_correct" not in df else bool_series(df["clean_correct"])
        success = bool_series(df["success"])
        success_on_clean = clean_correct & success
        clean_count = int(clean_correct.sum())
        row = {
            "metadata_file": str(path),
            "attack": df.get("attack", pd.Series([path.parent.name])).iloc[0],
            "epsilon": float(df["epsilon"].iloc[0]) if "epsilon" in df.columns else None,
            "samples": len(df),
            "clean_correct_samples": clean_count,
            "target_success_rate_all": float(success.mean()),
            "target_success_rate_on_clean": float(success_on_clean.sum() / clean_count) if clean_count else 0.0,
            "clean_accuracy_on_subset": float(clean_correct.mean()),
            "avg_true_conf_before": float(df["true_conf_before"].mean()),
            "avg_true_conf_after": float(df["true_conf_after"].mean()),
            "avg_target_conf_before": float(df["target_conf_before"].mean()),
            "avg_target_conf_after": float(df["target_conf_after"].mean()),
            "avg_target_conf_gain": float(df["target_conf_gain"].mean()),
            "avg_l2": float(df["l2"].mean()),
            "avg_linf": float(df["linf"].mean()),
            "avg_time_sec": float(df["time_sec"].mean()),
        }
        if "alpha" in df.columns:
            row["alpha"] = float(df["alpha"].iloc[0])
        if "steps" in df.columns:
            row["steps"] = int(df["steps"].iloc[0])
        if "max_queries" in df.columns:
            row["max_queries"] = int(df["max_queries"].iloc[0])
        if "queries_used" in df.columns:
            row["avg_queries_used"] = float(df["queries_used"].mean())
        if "theta" in df.columns:
            row["theta"] = float(df["theta"].iloc[0])
        if "pixels_per_step" in df.columns:
            row["pixels_per_step"] = int(df["pixels_per_step"].iloc[0])
        if "changed_channels" in df.columns:
            row["avg_changed_channels"] = float(df["changed_channels"].mean())
        if "coords_per_iter" in df.columns:
            row["coords_per_iter"] = int(df["coords_per_iter"].iloc[0])
        if "finite_diff_h" in df.columns:
            row["finite_diff_h"] = float(df["finite_diff_h"].iloc[0])
        if "learning_rate" in df.columns:
            row["learning_rate"] = float(df["learning_rate"].iloc[0])
        if "iterations" in df.columns:
            row["avg_iterations"] = float(df["iterations"].mean())
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values(["attack", "epsilon", "metadata_file"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved: {args.out.resolve()}")


if __name__ == "__main__":
    main()
