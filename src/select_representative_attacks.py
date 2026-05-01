"""Select representative attack samples for reporting and team review."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def pick_rows(group: pd.DataFrame, success: bool, count: int) -> pd.DataFrame:
    clean = group[bool_series(group["clean_correct"])]
    subset = clean[bool_series(clean["success_on_clean"]) == success].copy()
    if subset.empty:
        return subset

    if success:
        # Prefer strong, visually meaningful successes with higher target gain.
        subset = subset.sort_values(
            by=["target_conf_gain", "linf", "l2"],
            ascending=[False, True, True],
        )
    else:
        # Prefer failures where target confidence still moved, useful for analysis.
        subset = subset.sort_values(
            by=["target_conf_gain", "linf", "l2"],
            ascending=[False, True, True],
        )
    return subset.head(count)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select representative attack rows.")
    parser.add_argument("--attack-index", type=Path, default=Path("outputs/attacks/attack_index.csv"))
    parser.add_argument("--out", type=Path, default=Path("outputs/attacks/representative_samples.csv"))
    parser.add_argument("--count", type=int, default=3)
    args = parser.parse_args()

    index = pd.read_csv(args.attack_index)
    if index.empty:
        raise ValueError(f"No rows in {args.attack_index}")

    selected = []
    for family, group in index.groupby("attack_family"):
        successes = pick_rows(group, success=True, count=args.count)
        failures = pick_rows(group, success=False, count=args.count)
        if not successes.empty:
            successes = successes.assign(representative_type="success")
            selected.append(successes)
        if not failures.empty:
            failures = failures.assign(representative_type="failure")
            selected.append(failures)

    if not selected:
        raise ValueError("No representative rows selected")

    reps = pd.concat(selected, ignore_index=True)
    cols = [
        "representative_type",
        "sample_id",
        "attack_family",
        "attack",
        "file",
        "adv_file",
        "perturbation_file",
        "success_on_clean",
        "true_name",
        "target_name",
        "pred_before_name",
        "pred_after_name",
        "epsilon",
        "theta",
        "alpha",
        "steps",
        "max_queries",
        "queries_used",
        "l0",
        "l2",
        "linf",
        "time_sec",
        "target_conf_gain",
    ]
    existing_cols = [col for col in cols if col in reps.columns]
    reps = reps[existing_cols]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    reps.to_csv(args.out, index=False)

    print(f"Selected rows: {len(reps)}")
    print("Rows by attack_family/type:")
    print(reps.groupby(["attack_family", "representative_type"]).size().to_string())
    print(f"Saved: {args.out.resolve()}")


if __name__ == "__main__":
    main()
