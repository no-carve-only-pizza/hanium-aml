"""Summarize targeted face attack metadata files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize targeted face FGSM runs.")
    parser.add_argument("--metadata-dir", type=Path, default=Path("outputs/attacks/fgsm_face"))
    parser.add_argument("--out", type=Path, default=Path("outputs/attacks/fgsm_face/summary.csv"))
    args = parser.parse_args()

    paths = sorted(args.metadata_dir.glob("metadata_targeted_eps*.csv"))
    if not paths:
        raise FileNotFoundError(f"No metadata files found in {args.metadata_dir}")

    rows = []
    for path in paths:
        df = pd.read_csv(path)
        rows.append({
            "metadata_file": str(path),
            "epsilon": float(df["epsilon"].iloc[0]),
            "samples": len(df),
            "target_success_rate": float(df["success"].mean()),
            "clean_accuracy_on_subset": float((df["pred_before"] == df["true_label"]).mean()),
            "avg_true_conf_before": float(df["true_conf_before"].mean()),
            "avg_true_conf_after": float(df["true_conf_after"].mean()),
            "avg_target_conf_before": float(df["target_conf_before"].mean()),
            "avg_target_conf_after": float(df["target_conf_after"].mean()),
            "avg_target_conf_gain": float(df["target_conf_gain"].mean()),
            "avg_l2": float(df["l2"].mean()),
            "avg_linf": float(df["linf"].mean()),
            "avg_time_sec": float(df["time_sec"].mean()),
        })

    summary = pd.DataFrame(rows).sort_values("epsilon")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved: {args.out.resolve()}")


if __name__ == "__main__":
    main()
