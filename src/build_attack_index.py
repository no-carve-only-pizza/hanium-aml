"""Build a unified attack index for defense/web integration.

This script reads all attack metadata CSV files under outputs/attacks and writes
a normalized index with one row per attacked image. Defense modules can consume
this single CSV instead of handling attack-specific metadata formats.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import pandas as pd

COMMON_COLUMNS = [
    "sample_id",
    "attack",
    "attack_family",
    "source_metadata",
    "file",
    "adv_file",
    "perturbation_file",
    "success",
    "clean_correct",
    "success_on_clean",
    "true_label",
    "true_name",
    "target_label",
    "target_name",
    "pred_before",
    "pred_before_name",
    "pred_after",
    "pred_after_name",
    "epsilon",
    "theta",
    "alpha",
    "steps",
    "pixels_per_step",
    "max_queries",
    "queries_used",
    "coords_per_iter",
    "finite_diff_h",
    "learning_rate",
    "iterations",
    "l0",
    "l2",
    "linf",
    "time_sec",
    "target_conf_before",
    "target_conf_after",
    "target_conf_gain",
]


def boolify(value) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def attack_family(path: Path, attack_name: str) -> str:
    if "fgsm" in attack_name or "fgsm" in path.parts:
        return "fgsm"
    if "pgd" in attack_name or "pgd" in path.parts:
        return "pgd"
    if "square" in attack_name or "square" in path.parts:
        return "square"
    if "jsma" in attack_name or "jsma" in path.parts:
        return "jsma"
    if "zoo" in attack_name or "zoo" in path.parts:
        return "zoo"
    return attack_name or path.parent.name


def make_sample_id(row: pd.Series, metadata_path: Path) -> str:
    key = "|".join([
        str(row.get("file", "")),
        str(row.get("adv_file", "")),
        str(row.get("attack", "")),
        str(row.get("epsilon", "")),
        str(row.get("theta", "")),
        str(row.get("target_label", "")),
    ])
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    return f"atk_{digest}"


def normalize_metadata(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=COMMON_COLUMNS)

    attack_name = str(df.get("attack", pd.Series([path.parent.name])).iloc[0])
    family = attack_family(path, attack_name)

    rows = []
    for _, row in df.iterrows():
        pred_before = row.get("pred_before", "")
        true_label = row.get("true_label", "")
        clean_correct = row.get("clean_correct", None)
        if clean_correct is None or pd.isna(clean_correct):
            clean_correct = pred_before == true_label
        success = boolify(row.get("success", False))
        clean_correct = boolify(clean_correct)
        success_on_clean = row.get("success_on_clean", None)
        if success_on_clean is None or pd.isna(success_on_clean):
            success_on_clean = success and clean_correct
        else:
            success_on_clean = boolify(success_on_clean)

        normalized = {
            "sample_id": make_sample_id(row, path),
            "attack": attack_name,
            "attack_family": family,
            "source_metadata": str(path),
            "file": row.get("file", ""),
            "adv_file": row.get("adv_file", ""),
            "perturbation_file": row.get("perturbation_file", ""),
            "success": success,
            "clean_correct": clean_correct,
            "success_on_clean": success_on_clean,
            "true_label": row.get("true_label", ""),
            "true_name": row.get("true_name", ""),
            "target_label": row.get("target_label", ""),
            "target_name": row.get("target_name", ""),
            "pred_before": row.get("pred_before", ""),
            "pred_before_name": row.get("pred_before_name", ""),
            "pred_after": row.get("pred_after", ""),
            "pred_after_name": row.get("pred_after_name", ""),
            "epsilon": row.get("epsilon", ""),
            "theta": row.get("theta", ""),
            "alpha": row.get("alpha", ""),
            "steps": row.get("steps", ""),
            "pixels_per_step": row.get("pixels_per_step", ""),
            "max_queries": row.get("max_queries", ""),
            "queries_used": row.get("queries_used", ""),
            "coords_per_iter": row.get("coords_per_iter", ""),
            "finite_diff_h": row.get("finite_diff_h", ""),
            "learning_rate": row.get("learning_rate", ""),
            "iterations": row.get("iterations", ""),
            "l0": row.get("l0", ""),
            "l2": row.get("l2", ""),
            "linf": row.get("linf", ""),
            "time_sec": row.get("time_sec", ""),
            "target_conf_before": row.get("target_conf_before", ""),
            "target_conf_after": row.get("target_conf_after", ""),
            "target_conf_gain": row.get("target_conf_gain", ""),
        }
        rows.append(normalized)

    return pd.DataFrame(rows, columns=COMMON_COLUMNS)


def is_smoke_test_metadata(path: Path) -> bool:
    """Exclude tiny smoke-test metadata without dropping real query sweeps."""
    name = path.stem
    return bool(re.search(r"(^|_)queries20($|_)", name) or re.search(r"(^|_)steps3($|_)", name))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified attack_index.csv.")
    parser.add_argument("--metadata-root", type=Path, default=Path("outputs/attacks"))
    parser.add_argument("--out", type=Path, default=Path("outputs/attacks/attack_index.csv"))
    parser.add_argument("--include-smoke-tests", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-all-attack-dirs", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    face_attack_dirs = {"fgsm_face", "pgd_face", "square_face", "jsma_face", "zoo_face"}
    paths = sorted(args.metadata_root.glob("**/metadata_*.csv"))
    paths = [path for path in paths if path.name != "metadata.csv"]
    if not args.include_all_attack_dirs:
        paths = [path for path in paths if path.parent.name in face_attack_dirs]
    if not args.include_smoke_tests:
        paths = [path for path in paths if not is_smoke_test_metadata(path)]
    if not paths:
        raise FileNotFoundError(f"No metadata CSV files found under {args.metadata_root}")

    frames = [normalize_metadata(path) for path in paths]
    index = pd.concat(frames, ignore_index=True)
    index = index.drop_duplicates(subset=["sample_id"])
    index = index.sort_values(["attack_family", "attack", "file", "target_label"]).reset_index(drop=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(args.out, index=False)

    print(f"Metadata files: {len(paths)}")
    print(f"Indexed attacks: {len(index)}")
    print("Rows by attack_family:")
    print(index.groupby("attack_family").size().to_string())
    print(f"Saved: {args.out.resolve()}")


if __name__ == "__main__":
    main()
