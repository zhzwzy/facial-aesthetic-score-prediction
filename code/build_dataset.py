"""Build isolated 50-feature and 57-feature score-prediction datasets."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np
import pandas as pd


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = EXPERIMENT_ROOT / "inputs"
OUTPUT_ROOT = EXPERIMENT_ROOT / "derived_data"
MEASUREMENT_PREFIX = "measurement__"
NATURAL_LABEL = "\u81ea\u7136"
LEGACY_SAMPLE_NAME_COLUMN = "\u6837\u672c\u540d"
LEGACY_PERSON_NAME_COLUMN = "\u59d3\u540d"
NAME_COLUMN_CANDIDATES = ("name", LEGACY_PERSON_NAME_COLUMN, "sample_name", LEGACY_SAMPLE_NAME_COLUMN)

NEW_FEATURE_COLUMNS = [
    f"{MEASUREMENT_PREFIX}Ns-Gl({NATURAL_LABEL})(mm)",
    f"{MEASUREMENT_PREFIX}Prn-Ns({NATURAL_LABEL})(mm)",
    f"{MEASUREMENT_PREFIX}Ns-Sn({NATURAL_LABEL})(mm)",
    f"{MEASUREMENT_PREFIX}Sn-UL({NATURAL_LABEL})(mm)",
    f"{MEASUREMENT_PREFIX}Gl-Ns-Prn(°)",
    f"{MEASUREMENT_PREFIX}Ns-Prn-Sn(°)",
    f"{MEASUREMENT_PREFIX}Prn-Sn-UL(°)",
]


def load_reference_metrics_module():
    """Load the bundled geometric measurement implementation."""
    source = Path(__file__).resolve().with_name("metric_reference.py")
    if not source.exists():
        raise FileNotFoundError(f"Existing metric implementation not found: {source}")
    spec = importlib.util.spec_from_file_location("score_pred_reference_metrics", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load metric implementation: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


METRICS = load_reference_metrics_module()


def find_name_column(columns: Iterable[str]) -> str:
    column = next((candidate for candidate in NAME_COLUMN_CANDIDATES if candidate in columns), None)
    if column is None:
        raise ValueError("No subject-name column was found in the input table")
    return column


def finite_point(points: Mapping[str, Tuple[float, float]], name: str) -> Tuple[float, float]:
    point = points.get(name)
    if point is None or not METRICS.is_valid_point(point):
        raise ValueError(f"Required landmark {name!r} is missing or invalid")
    return float(point[0]), float(point[1])


def angle_at(
    first: Tuple[float, float],
    vertex: Tuple[float, float],
    last: Tuple[float, float],
) -> float:
    """Return the included 0–180 degree angle between the two rays."""
    ax, ay = first[0] - vertex[0], first[1] - vertex[1]
    bx, by = last[0] - vertex[0], last[1] - vertex[1]
    a_length = math.hypot(ax, ay)
    b_length = math.hypot(bx, by)
    if a_length == 0 or b_length == 0:
        raise ValueError("Cannot calculate an angle when a ray has zero length")
    dot = ax * bx + ay * by
    cross = ax * by - ay * bx
    return math.degrees(math.atan2(abs(cross), dot))


def calculate_new_features(
    rotated_points: Mapping[str, Tuple[float, float]],
) -> Dict[str, float]:
    """Calculate the seven requested measures in natural-head-position coordinates."""
    ns = finite_point(rotated_points, "Ns")
    glabella = finite_point(rotated_points, "G")
    prn = finite_point(rotated_points, "Prn")
    sn = finite_point(rotated_points, "Sn")
    upper_lip = finite_point(rotated_points, "UL")
    r1 = finite_point(rotated_points, "R1")
    r2 = finite_point(rotated_points, "R2")
    scale_pixels_per_mm = METRICS.calculate_distance(r1, r2) / 10.0
    if not math.isfinite(scale_pixels_per_mm) or scale_pixels_per_mm <= 0:
        raise ValueError("Invalid R1–R2 scale; expected a positive 10 mm reference")

    return {
        NEW_FEATURE_COLUMNS[0]: (glabella[0] - ns[0]) / scale_pixels_per_mm,
        NEW_FEATURE_COLUMNS[1]: abs(prn[0] - ns[0]) / scale_pixels_per_mm,
        NEW_FEATURE_COLUMNS[2]: abs(sn[1] - ns[1]) / scale_pixels_per_mm,
        NEW_FEATURE_COLUMNS[3]: (upper_lip[1] - sn[1]) / scale_pixels_per_mm,
        NEW_FEATURE_COLUMNS[4]: angle_at(glabella, ns, prn),
        NEW_FEATURE_COLUMNS[5]: angle_at(ns, prn, sn),
        NEW_FEATURE_COLUMNS[6]: angle_at(prn, sn, upper_lip),
    }


def load_points_for_stage(stage: str) -> Dict[str, Dict[str, Tuple[float, float]]]:
    point_root = INPUT_ROOT / "landmarks"
    sources = [
        point_root / "prompt_generation" / f"{stage}.csv",
        point_root / "core" / f"{stage}.csv",
    ]
    points: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for path in sources:
        if not path.exists():
            raise FileNotFoundError(f"Landmark CSV snapshot missing: {path}")
        # Match the existing score-prediction builder: core rows override duplicate
        # prompt-generation rows.
        points.update(METRICS.load_points_csv(str(path)))
    return points


def calculate_features_for_image(
    image_name: str,
    stage: str,
    point_maps: Mapping[str, Mapping[str, Dict[str, Tuple[float, float]]]],
    rotation_map: Mapping[str, float],
) -> Dict[str, float]:
    points = point_maps[stage].get(image_name)
    if points is None:
        raise ValueError(f"No landmark row for {image_name!r} ({stage})")
    if image_name not in rotation_map:
        raise ValueError(f"No natural-head-position rotation angle for {image_name!r}")
    rotated = METRICS.rotate_points(points, float(rotation_map[image_name]))
    return calculate_new_features(rotated)


def json_points(annotation_path: Path) -> Tuple[Dict[str, Tuple[float, float]], float]:
    payload = json.loads(annotation_path.read_text(encoding="utf-8-sig"))
    points: Dict[str, Tuple[float, float]] = {}
    for item in payload.get("points", []):
        name = str(item["name"])
        if name in points:
            raise ValueError(f"Duplicate landmark {name!r} in {annotation_path}")
        xy = item.get("xy")
        if not isinstance(xy, list) or len(xy) != 2:
            raise ValueError(f"Invalid coordinate for landmark {name!r} in {annotation_path}")
        points[name] = (float(xy[0]), float(xy[1]))
    if "rotation" not in payload:
        raise ValueError(f"Missing JSON rotation value in {annotation_path}")
    return points, float(payload["rotation"])


def calculate_legacy_features_from_json(
    annotation_path: Path,
    expected_feature_columns: Iterable[str],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    points, angle = json_points(annotation_path)
    rotated = METRICS.rotate_points(points, angle)
    legacy_values = METRICS.metric_value_dict(rotated)
    legacy_features: Dict[str, float] = {}
    for column in expected_feature_columns:
        metric_name = column[len(MEASUREMENT_PREFIX) :]
        value = legacy_values.get(metric_name)
        if value is None or not math.isfinite(float(value)):
            raise ValueError(
                f"Legacy feature {metric_name!r} is missing or non-finite for {annotation_path.name}"
            )
        legacy_features[column] = float(value)
    return legacy_features, calculate_new_features(rotated)


def make_added_male_rows(
    male_columns: Iterable[str],
    legacy_feature_columns: Iterable[str],
) -> pd.DataFrame:
    annotation_root = INPUT_ROOT / "new_male_annotations"
    annotation_files = sorted(annotation_root.rglob("*.json"))
    if len(annotation_files) != 8:
        raise ValueError(f"Expected exactly 8 new male JSON annotations, found {len(annotation_files)}")

    labels_path = INPUT_ROOT / "labels" / "supplementary_dataset.csv"
    label_df = pd.read_csv(labels_path, encoding="utf-8-sig")
    required = {"stage", "beauty_score", "is_beautiful"}
    person_name_column = next(
        (candidate for candidate in ("name", LEGACY_PERSON_NAME_COLUMN) if candidate in label_df.columns),
        None,
    )
    if person_name_column is None:
        required.add("name or the original localized subject-name column")
        person_name_column = LEGACY_PERSON_NAME_COLUMN
    missing = required - set(label_df.columns)
    if missing:
        raise ValueError(f"Supplementary score snapshot missing columns: {sorted(missing)}")

    male_column_list = list(male_columns)
    legacy_column_list = list(legacy_feature_columns)
    rows = []
    seen_names = set()
    for annotation_path in annotation_files:
        name = annotation_path.stem
        if name in seen_names:
            raise ValueError(f"Multiple annotations found for new male {name!r}")
        seen_names.add(name)
        matches = label_df[label_df[person_name_column].astype(str).str.strip() == name]
        if len(matches) != 1:
            raise ValueError(f"Expected one source rating for {name!r}, found {len(matches)}")
        label = matches.iloc[0]
        if str(label["stage"]).strip() != "post":
            raise ValueError(f"Expected a postoperative rating for {name!r}")

        legacy_features, new_features = calculate_legacy_features_from_json(
            annotation_path, legacy_column_list
        )
        score = float(label["beauty_score"])
        row = {column: None for column in male_column_list}
        row.update(
            {
                "stage": "post",
                "split": "not_assigned_before_random_split",
                "image": f"{name}.bmp",
                "beauty_score": score,
                "is_beautiful": int(label["is_beautiful"]),
            }
        )
        for name_column in ("sample_name", "name", LEGACY_SAMPLE_NAME_COLUMN, LEGACY_PERSON_NAME_COLUMN):
            if name_column in row:
                row[name_column] = name
        row.update(legacy_features)
        row.update(new_features)
        rows.append(row)

    return pd.DataFrame(rows, columns=male_column_list)


def prepare_base_dataset(
    path: Path,
    point_maps: Mapping[str, Mapping[str, Dict[str, Tuple[float, float]]]],
    rotation_map: Mapping[str, float],
    expected_rows: int,
) -> Tuple[pd.DataFrame, list[str]]:
    df = pd.read_excel(path)
    if len(df) != expected_rows:
        raise ValueError(f"{path.name}: expected {expected_rows} rows, found {len(df)}")
    legacy_feature_columns = [
        column for column in df.columns if str(column).startswith(MEASUREMENT_PREFIX)
    ]
    if len(legacy_feature_columns) != 50:
        raise ValueError(f"{path.name}: expected 50 source features, found {len(legacy_feature_columns)}")
    if any(column in df.columns for column in NEW_FEATURE_COLUMNS):
        raise ValueError(f"{path.name} already contains one or more new feature columns")

    added_features = []
    for _, row in df.iterrows():
        image_name = str(row["image"]).strip()
        stage = str(row["stage"]).strip()
        if stage not in {"pre", "post"}:
            raise ValueError(f"Unexpected stage {stage!r} for {image_name!r}")
        added_features.append(
            calculate_features_for_image(image_name, stage, point_maps, rotation_map)
        )
    new_feature_df = pd.DataFrame(added_features, index=df.index)
    for column in NEW_FEATURE_COLUMNS:
        df[column] = new_feature_df[column]

    # The workbooks carry a previous train/test label. All records are re-split by
    # the unified trainer after the male additions are made.
    if "split" in df.columns:
        df["split"] = "not_assigned_before_random_split"
    return df, legacy_feature_columns


def validate_output(
    df: pd.DataFrame,
    expected_features: int,
    expected_rows: int,
    label: str,
) -> None:
    feature_columns = [
        column for column in df.columns if str(column).startswith(MEASUREMENT_PREFIX)
    ]
    if len(df) != expected_rows:
        raise ValueError(f"{label}: expected {expected_rows} output rows, found {len(df)}")
    if len(feature_columns) != expected_features:
        raise ValueError(f"{label}: expected {expected_features} features, found {len(feature_columns)}")
    numeric = df[feature_columns + ["beauty_score"]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        bad_columns = numeric.columns[~np.isfinite(numeric.to_numpy(dtype=float)).all(axis=0)].tolist()
        raise ValueError(f"{label}: non-finite feature/target values found in {bad_columns}")
    sample_name_column = next((column for column in NAME_COLUMN_CANDIDATES if column in df.columns), None)
    if sample_name_column is not None and df[sample_name_column].duplicated().any():
        duplicated = df.loc[df[sample_name_column].duplicated(keep=False), sample_name_column].astype(str).tolist()
        raise ValueError(f"{label}: duplicate sample IDs found: {duplicated[:10]}")


def main() -> None:
    point_maps = {stage: load_points_for_stage(stage) for stage in ("pre", "post")}
    rotation_map = METRICS.load_rotation_map(str(INPUT_ROOT / "landmarks" / "core" / "rotation.csv"))

    female, female_legacy_columns = prepare_base_dataset(
        INPUT_ROOT / "base_data" / "female.xlsx", point_maps, rotation_map, expected_rows=675
    )
    male, male_legacy_columns = prepare_base_dataset(
        INPUT_ROOT / "base_data" / "male.xlsx", point_maps, rotation_map, expected_rows=264
    )
    if female_legacy_columns != male_legacy_columns:
        raise ValueError("female.xlsx and male.xlsx use different feature names or feature order")

    male_name_column = find_name_column(male.columns)
    male_names = set(male[male_name_column].astype(str))
    added = make_added_male_rows(male.columns, male_legacy_columns)
    added_name_column = find_name_column(added.columns)
    added_names = set(added[added_name_column].astype(str))
    overlap = male_names & added_names
    if overlap:
        raise ValueError(f"New male samples overlap existing male patients: {sorted(overlap)}")
    male = pd.concat([male, added], ignore_index=True)

    final_feature_count = len(female_legacy_columns) + len(NEW_FEATURE_COLUMNS)
    validate_output(female, final_feature_count, 675, "female")
    validate_output(male, final_feature_count, 272, "male")

    female_50 = female.drop(columns=NEW_FEATURE_COLUMNS).copy()
    male_50 = male.drop(columns=NEW_FEATURE_COLUMNS).copy()
    validate_output(female_50, len(female_legacy_columns), 675, "female (50 features)")
    validate_output(male_50, len(male_legacy_columns), 272, "male (50 features)")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    female_path = OUTPUT_ROOT / "female_57_features.csv"
    male_path = OUTPUT_ROOT / "male_57_features.csv"
    female_50_path = OUTPUT_ROOT / "female_50_features.csv"
    male_50_path = OUTPUT_ROOT / "male_50_features.csv"
    female.to_csv(female_path, index=False, encoding="utf-8-sig")
    male.to_csv(male_path, index=False, encoding="utf-8-sig")
    female_50.to_csv(female_50_path, index=False, encoding="utf-8-sig")
    male_50.to_csv(male_50_path, index=False, encoding="utf-8-sig")

    print(f"Female: {len(female)} rows, {final_feature_count} features -> {female_path}")
    print(f"Male: {len(male)} rows ({len(male) - len(added)} base + {len(added)} added), "
          f"{final_feature_count} features -> {male_path}")
    print(f"Female: {len(female_50)} rows, {len(female_legacy_columns)} features -> {female_50_path}")
    print(f"Male: {len(male_50)} rows ({len(male_50) - len(added)} base + {len(added)} added), "
          f"{len(male_legacy_columns)} features -> {male_50_path}")
    print(f"Added male annotation records: {len(added_names)}")
    print("Random split is intentionally deferred to the training script (test_size=0.2, random_state=42).")


if __name__ == "__main__":
    main()
