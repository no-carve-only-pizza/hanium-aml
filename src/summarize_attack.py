"""Summarize FGSM metadata CSV files into one compact experiment table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize attack metadata CSV files.")
    parser.add_argument("--metadata-dir", type=Path, default=Path("outputs/attacks/fgsm"))
    parser.add_argument("--out", type=Path, default=Path("outputs/attacks/fgsm/summary.csv"))
    args = parser.parse_args()

    paths = sorted(args.metadata_dir.glob("metadata_targeted_eps*.csv"))
    if not paths:
        raise FileNotFoundError(f"No targeted metadata files found in {args.metadata_dir}")

    rows = []
    for path in paths:
        df = pd.read_csv(path)
        rows.append({
            "metadata_file": str(path),
            "attack": df["attack"].iloc[0],
            "targeted": bool(df["targeted"].iloc[0]),
            "epsilon": float(df["epsilon"].iloc[0]),
            "target_class_id": int(df["target_class_id"].iloc[0]),
            "target_class_name": df["target_class_name"].iloc[0],
            "samples": len(df),
            "target_success_rate": float(df["success"].mean()),
            "avg_target_confidence_before": float(df["target_confidence_before"].mean()),
            "avg_target_confidence_after": float(df["target_confidence_after"].mean()),
            "avg_l0": float(df["l0"].mean()),
            "avg_l2": float(df["l2"].mean()),
            "avg_linf": float(df["linf"].mean()),
            "avg_cmd": float(df["cmd"].mean()),
            "avg_time_sec": float(df["time_sec"].mean()),
        })

    summary = pd.DataFrame(rows).sort_values("epsilon")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved: {args.out.resolve()}")


if __name__ == "__main__":
    main()
