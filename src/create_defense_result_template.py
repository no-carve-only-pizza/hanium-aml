"""Create an empty defense_results.csv template from attack_index.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFENSE_COLUMNS = [
    "sample_id",
    "attack_family",
    "attack",
    "defense",
    "defense_params",
    "input_adv_file",
    "defended_file",
    "pred_before_defense",
    "pred_after_defense",
    "target_label",
    "true_label",
    "attack_success_before_defense",
    "attack_success_after_defense",
    "recovered",
    "target_conf_before_defense",
    "target_conf_after_defense",
    "defense_time_sec",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create defense result CSV template.")
    parser.add_argument("--attack-index", type=Path, default=Path("outputs/attacks/attack_index.csv"))
    parser.add_argument("--out", type=Path, default=Path("outputs/defenses/defense_results_template.csv"))
    parser.add_argument("--only-success-on-clean", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    attack_index = pd.read_csv(args.attack_index)
    if args.only_success_on_clean:
        attack_index = attack_index[
            (attack_index["clean_correct"] == True) &
            (attack_index["success_on_clean"] == True)
        ]

    template = pd.DataFrame({
        "sample_id": attack_index["sample_id"],
        "attack_family": attack_index["attack_family"],
        "attack": attack_index["attack"],
        "defense": "",
        "defense_params": "",
        "input_adv_file": attack_index["adv_file"],
        "defended_file": "",
        "pred_before_defense": attack_index["pred_after"],
        "pred_after_defense": "",
        "target_label": attack_index["target_label"],
        "true_label": attack_index["true_label"],
        "attack_success_before_defense": attack_index["success_on_clean"],
        "attack_success_after_defense": "",
        "recovered": "",
        "target_conf_before_defense": attack_index["target_conf_after"],
        "target_conf_after_defense": "",
        "defense_time_sec": "",
    }, columns=DEFENSE_COLUMNS)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(args.out, index=False)
    print(f"Template rows: {len(template)}")
    print(f"Saved: {args.out.resolve()}")


if __name__ == "__main__":
    main()
