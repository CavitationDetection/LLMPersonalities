"""Aggregate downstream scores over repeats, tasks, and dimensions.

N/A values are excluded rather than scored as zero. Task means use available
repeats, dimension means use available task means, and all outputs retain
counts and N/A rates so variable coverage remains auditable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_SCORES_PATH = ROOT / "outputs" / "downstream_scores_long.csv"
OUTPUT_TASK_PATH = ROOT / "outputs" / "downstream_task_aggregates.csv"
OUTPUT_DIMENSION_LONG_PATH = ROOT / "outputs" / "downstream_dimension_aggregates_long.csv"
OUTPUT_DIMENSION_WIDE_PATH = ROOT / "outputs" / "downstream_dimension_aggregates.csv"
OUTPUT_STABILITY_PATH = ROOT / "outputs" / "repeat_stability.csv"

EXPECTED_REPEATS = {1, 2, 3, 4, 5}
DIMENSION_NAMES = {
    "Care": "MFV_Care",
    "Fairness": "MFV_Fairness",
    "Liberty": "MFV_Liberty",
    "Authority": "MFV_Authority",
    "Ingroup": "MFV_Ingroup",
    "Purity": "MFV_Purity",
    "Instrumental_Harm": "OUS_IH_downstream",
    "Impartial_Beneficence": "OUS_IB_downstream",
}
WIDE_DIMENSIONS = [
    "MFV_Care",
    "MFV_Fairness",
    "MFV_Liberty",
    "MFV_Authority",
    "MFV_Ingroup",
    "MFV_Loyalty",
    "MFV_Purity",
    "OUS_IH_downstream",
    "OUS_IB_downstream",
]


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().eq("true")


def validate_input(frame: pd.DataFrame) -> None:
    required = {
        "model_name",
        "instrument",
        "scale",
        "dimension",
        "task_id",
        "language",
        "repeat_id",
        "score_name",
        "score_value",
        "response_valid",
        "is_na",
        "score_available",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Downstream score file is missing columns: {missing}")
    repeats = set(pd.to_numeric(frame["repeat_id"], errors="raise").astype(int))
    if repeats != EXPECTED_REPEATS:
        raise RuntimeError(f"Expected repeats {sorted(EXPECTED_REPEATS)}, found {sorted(repeats)}")
    duplicate_columns = ["model_name", "task_id", "language", "repeat_id"]
    duplicates = frame.duplicated(duplicate_columns, keep=False)
    if duplicates.any():
        sample = frame.loc[duplicates, duplicate_columns].head().to_dict("records")
        raise RuntimeError(f"Duplicate downstream records found: {sample}")


def aggregate_tasks(frame: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "model_name",
        "instrument",
        "scale",
        "dimension",
        "task_id",
        "language",
        "score_name",
    ]
    rows = []
    for key, group in frame.groupby(group_columns, sort=True, dropna=False):
        scores = group.loc[group["score_available"], "score_value"].dropna()
        row = dict(zip(group_columns, key))
        row.update(
            {
                "mean_score": scores.mean() if not scores.empty else np.nan,
                "std_score": scores.std(ddof=1) if len(scores) > 1 else 0.0 if len(scores) == 1 else np.nan,
                "median_score": scores.median() if not scores.empty else np.nan,
                "n_valid": int(len(scores)),
                "n_na": int(group["is_na"].sum()),
                "n_invalid": int((~group["response_valid"]).sum()),
                "n_total": int(len(group)),
            }
        )
        row["na_rate"] = row["n_na"] / row["n_total"] if row["n_total"] else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_columns).reset_index(drop=True)


def aggregate_dimensions(task_aggregates: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["model_name", "language", "instrument", "scale", "dimension"]
    rows = []
    for key, group in task_aggregates.groupby(group_columns, sort=True, dropna=False):
        task_scores = group["mean_score"].dropna()
        task_stability = group.loc[group["n_valid"].gt(1), "std_score"].dropna()
        row = dict(zip(group_columns, key))
        row.update(
            {
                "downstream_dimension": DIMENSION_NAMES[row["dimension"]],
                "mean_score": task_scores.mean() if not task_scores.empty else np.nan,
                "mean_repeat_std": task_stability.mean() if not task_stability.empty else np.nan,
                "n_tasks": int(len(group)),
                "n_tasks_scored": int(group["mean_score"].notna().sum()),
                "n_valid_repeats": int(group["n_valid"].sum()),
                "n_na": int(group["n_na"].sum()),
                "n_invalid": int(group["n_invalid"].sum()),
                "n_total": int(group["n_total"].sum()),
            }
        )
        row["na_rate"] = row["n_na"] / row["n_total"] if row["n_total"] else np.nan
        rows.append(row)

    result = pd.DataFrame(rows)
    loyalty = result.loc[result["downstream_dimension"].eq("MFV_Ingroup")].copy()
    loyalty["downstream_dimension"] = "MFV_Loyalty"
    result = pd.concat([result, loyalty], ignore_index=True)
    return result.sort_values(["model_name", "language", "downstream_dimension"]).reset_index(drop=True)


def make_dimension_wide(dimension_long: pd.DataFrame) -> pd.DataFrame:
    wide = dimension_long.pivot(
        index=["model_name", "language"],
        columns="downstream_dimension",
        values="mean_score",
    ).reset_index()
    wide.columns.name = None
    for column in WIDE_DIMENSIONS:
        if column not in wide.columns:
            wide[column] = np.nan
    return wide[["model_name", "language", *WIDE_DIMENSIONS]].sort_values(
        ["model_name", "language"]
    ).reset_index(drop=True)


def make_stability(dimension_long: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "model_name",
        "instrument",
        "dimension",
        "downstream_dimension",
        "language",
        "mean_score",
        "mean_repeat_std",
        "n_tasks",
        "n_valid_repeats",
        "n_na",
        "na_rate",
    ]
    return dimension_long.loc[
        ~dimension_long["downstream_dimension"].eq("MFV_Loyalty"), columns
    ].sort_values(["model_name", "instrument", "dimension", "language"])


def main() -> None:
    scores = pd.read_csv(INPUT_SCORES_PATH)
    validate_input(scores)
    scores["repeat_id"] = pd.to_numeric(scores["repeat_id"], errors="raise").astype(int)
    scores["score_value"] = pd.to_numeric(scores["score_value"], errors="coerce")
    for column in ["response_valid", "is_na", "score_available"]:
        scores[column] = as_bool(scores[column])

    task_aggregates = aggregate_tasks(scores)
    dimension_long = aggregate_dimensions(task_aggregates)
    dimension_wide = make_dimension_wide(dimension_long)
    stability = make_stability(dimension_long)

    OUTPUT_TASK_PATH.parent.mkdir(parents=True, exist_ok=True)
    task_aggregates.to_csv(OUTPUT_TASK_PATH, index=False)
    dimension_long.to_csv(OUTPUT_DIMENSION_LONG_PATH, index=False)
    dimension_wide.to_csv(OUTPUT_DIMENSION_WIDE_PATH, index=False)
    stability.to_csv(OUTPUT_STABILITY_PATH, index=False)

    print(f"Task aggregates: {len(task_aggregates)} -> {OUTPUT_TASK_PATH}")
    print(f"Dimension aggregates long: {len(dimension_long)} -> {OUTPUT_DIMENSION_LONG_PATH}")
    print(f"Dimension aggregates wide: {len(dimension_wide)} -> {OUTPUT_DIMENSION_WIDE_PATH}")
    print(f"Repeat stability: {len(stability)} -> {OUTPUT_STABILITY_PATH}")


if __name__ == "__main__":
    main()
