"""Train 50- and/or 57-feature models in the dated isolated experiment folder."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = EXPERIMENT_ROOT / "code"
OUTPUT_ROOT = EXPERIMENT_ROOT / "outputs"
ACTIVE_FEATURE_COUNT = 57

sys.path.insert(0, str(CODE_ROOT))
import train_beauty_score_prediction_models_random_vis as trainer  # noqa: E402


_original_loader = trainer.load_dataset_random_only


def configure_feature_group(feature_count: int) -> Path:
    global ACTIVE_FEATURE_COUNT
    ACTIVE_FEATURE_COUNT = feature_count
    suffix = f"{feature_count}_features"
    trainer.DATASET_RUNS = [
        ("female", EXPERIMENT_ROOT / "derived_data" / f"female_{suffix}.csv"),
        ("male", EXPERIMENT_ROOT / "derived_data" / f"male_{suffix}.csv"),
    ]
    group_root = OUTPUT_ROOT / suffix
    trainer.OUTPUT_ROOT = group_root
    trainer.SENSITIVITY_ROOT = group_root / "sensitivity_analysis_5_metrics"
    return group_root


def load_experiment_split():
    result = list(_original_loader())
    train_df, test_df, train_ids, test_ids = result[:4]
    for frame, split_name in ((train_df, "train"), (test_df, "test")):
        if "split" in frame.columns:
            frame["split"] = f"random_{split_name}"
        if "split_policy_20260803" in frame.columns:
            frame["split_policy_20260803"] = (
                f"random_{split_name}_measurement_expansion_7_{ACTIVE_FEATURE_COUNT}_features"
            )
    for frame, split_name in ((train_ids, "train"), (test_ids, "test")):
        if "split" in frame.columns:
            frame["split"] = f"random_{split_name}"
        if "split_policy_20260803" in frame.columns:
            frame["split_policy_20260803"] = (
                f"random_{split_name}_measurement_expansion_7_{ACTIVE_FEATURE_COUNT}_features"
            )
    return tuple(result)


trainer.load_dataset_random_only = load_experiment_split


def correct_report_text(group_root: Path) -> None:
    replacements = {
        "Female and male cohorts are trained, tuned, evaluated, and interpreted separately; only their corresponding source CSV files are used.": (
            "Female and male cohorts are handled separately; the additional male records are merged before the male cohort is split."
        ),
        "no supplementary data": "additional male annotations merged before random splitting",
    }
    for path in group_root.rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        for old, new in replacements.items():
            text = text.replace(old, new)
        path.write_text(text, encoding="utf-8")

    old_policy = (
        f"random split only, test_size={trainer.base.TEST_SIZE}, "
        f"random_state={trainer.base.RANDOM_STATE}; no supplementary data"
    )
    new_policy = (
        f"random split after merging additional male annotations, test_size={trainer.base.TEST_SIZE}, "
        f"random_state={trainer.base.RANDOM_STATE}"
    )
    for path in group_root.rglob("metrics_summary.csv"):
        frame = pd.read_csv(path, encoding="utf-8-sig")
        for column in frame.select_dtypes(include="object").columns:
            frame[column] = frame[column].replace(old_policy, new_policy)
        frame.to_csv(path, index=False, encoding="utf-8-sig")


def run_feature_group(feature_count: int, force: bool = False) -> None:
    group_root = configure_feature_group(feature_count)
    for _, dataset_path in trainer.DATASET_RUNS:
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Derived dataset not found: {dataset_path}. Run code/build_dataset.py first."
            )

    report_path = group_root / "report.md"
    if report_path.exists() and not force:
        print(f"[Skip completed] {feature_count}-feature results already exist: {group_root}")
        return
    if group_root.exists() and any(group_root.iterdir()) and not force:
        raise FileExistsError(
            f"Output directory exists but has no completion report: {group_root}. "
            "Inspect it first, or pass --force to continue into that directory."
        )

    group_root.mkdir(parents=True, exist_ok=True)
    trainer.main()
    correct_report_text(group_root)
    print(f"Experiment model outputs: {group_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        choices=("50", "57", "all"),
        default="all",
        help="Feature group(s) to train. Completed groups are skipped by default.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Continue into an existing feature-group output directory.",
    )
    args = parser.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    feature_groups = (50, 57) if args.features == "all" else (int(args.features),)
    for feature_count in feature_groups:
        run_feature_group(feature_count, force=args.force)


if __name__ == "__main__":
    main()
