"""Create panels for rows selected by select_representative_attacks.py."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from make_face_attack_panels import make_panel


def main() -> None:
    parser = argparse.ArgumentParser(description="Create panels for representative attack samples.")
    parser.add_argument("--representatives", type=Path, default=Path("outputs/attacks/representative_samples.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/attack_panels/representatives"))
    args = parser.parse_args()

    with args.representatives.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows in {args.representatives}")

    made = 0
    for row in rows:
        family = row.get("attack_family", "attack")
        rep_type = row.get("representative_type", "sample")
        sample_id = row.get("sample_id", f"{made + 1:02d}")
        out = args.out_dir / family / rep_type / f"{sample_id}.jpg"
        make_panel(row, out)
        made += 1

    print(f"Created {made} representative panels in {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
