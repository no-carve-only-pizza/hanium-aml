"""Summarize defense result CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize defense results.")
    parser.add_argument("--defense-root", type=Path, default=Path("outputs/defenses"))
    parser.add_argument("--out", type=Path, default=Path("outputs/defenses/defense_summary.csv"))
    args = parser.parse_args()

    paths = sorted(args.defense_root.glob("**/*results*.csv"))
    paths = [path for path in paths if "template" not in path.name]
    if not paths:
        raise FileNotFoundError(f"No defense result CSV files found under {args.defense_root}")

    rows = []
    for path in paths:
        df = pd.read_csv(path)
        df = df[df.get("status", "ok") == "ok"] if "status" in df.columns else df
        if df.empty:
            continue
        before = bool_series(df["attack_success_before_defense"])
        after = bool_series(df["attack_success_after_defense"])
        recovered = bool_series(df["recovered"])
        target_drop = df["target_conf_before_defense"].astype(float) - df["target_conf_after_defense"].astype(float)
        for attack_family, group in df.groupby("attack_family"):
            g_before = bool_series(group["attack_success_before_defense"])
            g_after = bool_series(group["attack_success_after_defense"])
            g_recovered = bool_series(group["recovered"])
            g_drop = group["target_conf_before_defense"].astype(float) - group["target_conf_after_defense"].astype(float)
            rows.append({
                "result_file": str(path),
                "defense": group["defense"].iloc[0],
                "defense_params": group["defense_params"].iloc[0],
                "attack_family": attack_family,
                "samples": len(group),
                "defense_success_rate": float(1 - g_after[g_before].mean()) if g_before.any() else 0.0,
                "recovery_rate": float(g_recovered[g_before].mean()) if g_before.any() else 0.0,
                "avg_target_conf_drop": float(g_drop.mean()),
                "avg_defense_time_sec": float(group["defense_time_sec"].astype(float).mean()),
            })
        rows.append({
            "result_file": str(path),
            "defense": df["defense"].iloc[0],
            "defense_params": df["defense_params"].iloc[0],
            "attack_family": "ALL",
            "samples": len(df),
            "defense_success_rate": float(1 - after[before].mean()) if before.any() else 0.0,
            "recovery_rate": float(recovered[before].mean()) if before.any() else 0.0,
            "avg_target_conf_drop": float(target_drop.mean()),
            "avg_defense_time_sec": float(df["defense_time_sec"].astype(float).mean()),
        })

    summary = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved: {args.out.resolve()}")


if __name__ == "__main__":
    main()
