"""Create report-friendly attack summary figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ATTACK_LABELS = {
    "targeted_fgsm": "FGSM",
    "targeted_pgd": "PGD",
    "targeted_square": "Square",
    "targeted_jsma_multi_pixel": "JSMA",
    "targeted_zoo": "ZOO",
}


def pct_to_float(value: str) -> float:
    return float(str(value).replace("%", ""))


def load_compact(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"No rows in {path}")
    df["attack_label"] = df["attack"].map(ATTACK_LABELS).fillna(df["attack"])
    df["target_ASR_on_clean_pct"] = df["target_ASR_on_clean"].apply(pct_to_float)
    df["clean_accuracy_pct"] = df["clean_accuracy"].apply(pct_to_float)
    return df


def save_bar(
    df: pd.DataFrame,
    column: str,
    ylabel: str,
    title: str,
    out_path: Path,
    *,
    ylim: tuple[float, float] | None = None,
    color: str = "#4062BB",
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(df["attack_label"], df[column], color=color, edgecolor="#202124", linewidth=0.8)
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Targeted attack")
    ax.grid(axis="y", alpha=0.25)
    if ylim:
        ax.set_ylim(*ylim)
    for bar in bars:
        height = bar.get_height()
        label = f"{height:.1f}%" if "pct" in column else f"{height:.3g}"
        ax.text(bar.get_x() + bar.get_width() / 2, height, label, ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_tradeoff(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["avg_l2"], df["target_ASR_on_clean_pct"], s=90, color="#2E8B57", edgecolor="#202124")
    for _, row in df.iterrows():
        ax.annotate(row["attack_label"], (row["avg_l2"], row["target_ASR_on_clean_pct"]), xytext=(6, 5), textcoords="offset points")
    ax.set_title("Targeted Attack Success vs Perturbation Size", fontsize=14, pad=12)
    ax.set_xlabel("Average L2 perturbation")
    ax.set_ylabel("Target ASR on clean samples (%)")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot compact targeted attack summary.")
    parser.add_argument("--compact", type=Path, default=Path("outputs/attacks/compact_attack_summary.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/attacks/figures"))
    args = parser.parse_args()

    df = load_compact(args.compact)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    save_bar(
        df,
        "target_ASR_on_clean_pct",
        "Target ASR on clean samples (%)",
        "Targeted Attack Success Rate",
        args.out_dir / "attack_success_rate.png",
        ylim=(0, 105),
        color="#4062BB",
    )
    save_bar(
        df,
        "avg_time_sec",
        "Average time per image (sec)",
        "Average Attack Runtime",
        args.out_dir / "attack_runtime.png",
        color="#D1603D",
    )
    save_bar(
        df,
        "avg_linf",
        "Average Linf perturbation",
        "Average Linf Perturbation",
        args.out_dir / "attack_linf.png",
        color="#6B5B95",
    )
    save_tradeoff(df, args.out_dir / "attack_success_l2_tradeoff.png")

    print(f"Saved figures to: {args.out_dir.resolve()}")
    for path in sorted(args.out_dir.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()
