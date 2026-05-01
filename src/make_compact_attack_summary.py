"""Create a compact report-ready attack summary table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def fmt_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def fmt_float(value, digits: int = 4) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def setting_for(row: pd.Series) -> str:
    attack = str(row["attack"])
    if attack == "targeted_fgsm":
        return f"eps={row['epsilon']:.3f}"
    if attack == "targeted_pgd":
        return f"eps={row['epsilon']:.3f}, alpha={row.get('alpha', 0):.4f}, steps={int(row.get('steps', 0))}"
    if attack == "targeted_square":
        return f"eps={row['epsilon']:.3f}, queries={int(row.get('max_queries', 0))}"
    if attack == "targeted_jsma_multi_pixel":
        return f"theta={row.get('theta', 0):.3f}, steps={int(row.get('steps', 0))}, k={int(row.get('pixels_per_step', 0))}"
    if attack == "targeted_zoo":
        return f"eps={row['epsilon']:.3f}, queries={int(row.get('max_queries', 0))}"
    return ""


def note_for(row: pd.Series) -> str:
    attack = str(row["attack"])
    if attack == "targeted_fgsm":
        return "fast one-step white-box baseline"
    if attack == "targeted_pgd":
        return "strong iterative white-box baseline"
    if attack == "targeted_square":
        return "query-based black-box baseline"
    if attack == "targeted_jsma_multi_pixel":
        return "sparse multi-pixel saliency attack"
    if attack == "targeted_zoo":
        return "finite-difference black-box baseline"
    return ""


def choose_report_rows(summary: pd.DataFrame) -> pd.DataFrame:
    picks = []
    for attack, group in summary.groupby("attack"):
        if attack == "targeted_fgsm":
            # Best successful FGSM setting by ASR on clean.
            picks.append(group.sort_values("target_success_rate_on_clean", ascending=False).head(1))
        elif attack == "targeted_pgd":
            # Prefer the clean epsilon/alpha matched setting eps=0.03 if present.
            candidate = group[(group["epsilon"] == 0.03) & (group.get("alpha", pd.Series(index=group.index)) == 0.003)]
            picks.append(candidate.head(1) if not candidate.empty else group.sort_values("target_success_rate_on_clean", ascending=False).head(1))
        elif attack == "targeted_square":
            picks.append(group.sort_values("target_success_rate_on_clean", ascending=False).head(1))
        elif attack == "targeted_jsma_multi_pixel":
            picks.append(group.head(1))
        elif attack == "targeted_zoo":
            picks.append(group.sort_values("target_success_rate_on_clean", ascending=False).head(1))
        else:
            picks.append(group.sort_values("target_success_rate_on_clean", ascending=False).head(1))
    return pd.concat(picks, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create compact attack summary for reports.")
    parser.add_argument("--summary", type=Path, default=Path("outputs/attacks/face_attack_summary.csv"))
    parser.add_argument("--out", type=Path, default=Path("outputs/attacks/compact_attack_summary.csv"))
    parser.add_argument("--include-all", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    summary = pd.read_csv(args.summary)
    if summary.empty:
        raise ValueError(f"No rows in {args.summary}")

    rows = summary if args.include_all else choose_report_rows(summary)
    compact_rows = []
    for _, row in rows.iterrows():
        compact_rows.append({
            "attack": row["attack"],
            "setting": setting_for(row),
            "samples": int(row["samples"]),
            "clean_accuracy": fmt_pct(float(row["clean_accuracy_on_subset"])),
            "target_ASR_on_clean": fmt_pct(float(row["target_success_rate_on_clean"])),
            "avg_l2": fmt_float(row.get("avg_l2")),
            "avg_linf": fmt_float(row.get("avg_linf")),
            "avg_time_sec": fmt_float(row.get("avg_time_sec")),
            "avg_queries_used": fmt_float(row.get("avg_queries_used"), digits=2),
            "note": note_for(row),
        })

    compact = pd.DataFrame(compact_rows)
    order = ["targeted_fgsm", "targeted_pgd", "targeted_square", "targeted_jsma_multi_pixel", "targeted_zoo"]
    compact["_order"] = compact["attack"].apply(lambda x: order.index(x) if x in order else len(order))
    compact = compact.sort_values("_order").drop(columns=["_order"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    compact.to_csv(args.out, index=False)
    print(compact.to_string(index=False))
    print(f"\nSaved: {args.out.resolve()}")


if __name__ == "__main__":
    main()
