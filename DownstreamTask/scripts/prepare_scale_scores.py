"""Prepare PromptA, PromptB, and PromptMean scale scores for analysis.

Each prompt condition is first averaged over its five scale repeats.
PromptMean is computed only when both prompt-specific means are available.
Only official MFQ-30 and OUS-9 dimensions required downstream are retained.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
API_EXP_ROOT = ROOT.parent
PROMPT_A_PATH = (
    API_EXP_ROOT
    / "统计"
    / "PromptA"
    / "最终结果_原始文献复核"
    / "final_scores_by_run_language_official_only.csv"
)
PROMPT_B_PATH = (
    API_EXP_ROOT
    / "统计"
    / "PromptB"
    / "最终结果_原始文献复核"
    / "final_scores_by_run_language_official_only.csv"
)
MODEL_LIST_PATH = ROOT / "data" / "model_list.json"
OUTPUT_LONG_PATH = ROOT / "data" / "scale_scores_prompt_long.csv"
OUTPUT_MEAN_PATH = ROOT / "data" / "scale_scores_prompt_mean.csv"

EXPECTED_REPEATS = {1, 2, 3, 4, 5}
SCALE_DIMENSIONS = {
    "MFQ-30": {"Harm", "Fairness", "Ingroup", "Authority", "Purity"},
    "OUS-9": {"Instrumental_Harm", "Impartial_Beneficence"},
}
KEY_COLUMNS = ["model_name", "language", "scale", "dimension"]


def selected_models() -> list[str]:
    payload = json.loads(MODEL_LIST_PATH.read_text(encoding="utf-8"))
    return [
        item if isinstance(item, str) else str(item["model_name"])
        for item in payload
        if isinstance(item, str) or item.get("enabled", True)
    ]


def read_prompt_file(path: Path, prompt_condition: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "model",
        "repeat_num",
        "language",
        "prompt",
        "memory",
        "order",
        "scale",
        "dimension",
        "score",
        "na_rate",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"{path} is missing columns: {missing}")

    frame = frame.loc[
        frame["scale"].isin(SCALE_DIMENSIONS)
        & frame["memory"].astype(str).str.lower().eq("none")
        & frame["order"].astype(str).str.lower().eq("forward")
        & frame["language"].isin(["en", "zh"])
    ].copy()
    frame = frame.loc[
        frame.apply(lambda row: row["dimension"] in SCALE_DIMENSIONS[row["scale"]], axis=1)
    ].copy()
    frame["repeat_num"] = pd.to_numeric(frame["repeat_num"], errors="raise").astype(int)
    frame = frame.loc[frame["repeat_num"].isin(EXPECTED_REPEATS)].copy()
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
    frame["na_rate"] = pd.to_numeric(frame["na_rate"], errors="coerce")
    frame["model_name"] = frame["model"].astype(str)
    frame["prompt_condition"] = prompt_condition

    expected_prompt = "A" if prompt_condition == "PromptA" else "B"
    unexpected_prompt = frame.loc[~frame["prompt"].astype(str).str.upper().eq(expected_prompt)]
    if not unexpected_prompt.empty:
        raise RuntimeError(f"Unexpected prompt labels in {path}: {unexpected_prompt['prompt'].unique()}")

    duplicate_columns = [*KEY_COLUMNS, "repeat_num"]
    duplicates = frame.duplicated(duplicate_columns, keep=False)
    if duplicates.any():
        sample = frame.loc[duplicates, duplicate_columns].head().to_dict("records")
        raise RuntimeError(f"Duplicate scale-score rows in {path}: {sample}")
    return frame


def aggregate_prompt(frame: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["model_name", "prompt_condition", "language", "scale", "dimension"]
    rows = []
    for key, group in frame.groupby(group_columns, sort=True, dropna=False):
        scores = group["score"].dropna()
        row = dict(zip(group_columns, key))
        row.update(
            {
                "scale_score": scores.mean() if not scores.empty else np.nan,
                "scale_score_std": scores.std(ddof=1) if len(scores) > 1 else 0.0 if len(scores) == 1 else np.nan,
                "mean_na_rate": group["na_rate"].mean(),
                "n_valid_runs": int(group["score"].notna().sum()),
                "n_total_runs": int(len(group)),
                "prompt_a_score": np.nan,
                "prompt_b_score": np.nan,
                "prompt_a_na_rate": np.nan,
                "prompt_b_na_rate": np.nan,
                "n_prompt_conditions": 1 if not scores.empty else 0,
                "prompt_mean_eligible": False,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_prompt_mean(primary: pd.DataFrame) -> pd.DataFrame:
    component_columns = [
        *KEY_COLUMNS,
        "scale_score",
        "mean_na_rate",
        "n_valid_runs",
        "n_total_runs",
    ]
    prompt_a = primary.loc[primary["prompt_condition"].eq("PromptA"), component_columns].copy()
    prompt_b = primary.loc[primary["prompt_condition"].eq("PromptB"), component_columns].copy()
    merged = prompt_a.merge(
        prompt_b,
        on=KEY_COLUMNS,
        how="outer",
        suffixes=("_a", "_b"),
        validate="one_to_one",
    )
    eligible = merged["scale_score_a"].notna() & merged["scale_score_b"].notna()
    result = merged[KEY_COLUMNS].copy()
    result["prompt_condition"] = "PromptMean"
    result["scale_score"] = np.where(
        eligible,
        (merged["scale_score_a"] + merged["scale_score_b"]) / 2,
        np.nan,
    )
    result["scale_score_std"] = np.nan
    result["mean_na_rate"] = merged[["mean_na_rate_a", "mean_na_rate_b"]].mean(axis=1)
    result["n_valid_runs"] = merged[["n_valid_runs_a", "n_valid_runs_b"]].fillna(0).sum(axis=1).astype(int)
    result["n_total_runs"] = merged[["n_total_runs_a", "n_total_runs_b"]].fillna(0).sum(axis=1).astype(int)
    result["prompt_a_score"] = merged["scale_score_a"]
    result["prompt_b_score"] = merged["scale_score_b"]
    result["prompt_a_na_rate"] = merged["mean_na_rate_a"]
    result["prompt_b_na_rate"] = merged["mean_na_rate_b"]
    result["n_prompt_conditions"] = merged[["scale_score_a", "scale_score_b"]].notna().sum(axis=1)
    result["prompt_mean_eligible"] = eligible
    ordered_columns = [
        "model_name",
        "prompt_condition",
        "language",
        "scale",
        "dimension",
        "scale_score",
        "scale_score_std",
        "mean_na_rate",
        "n_valid_runs",
        "n_total_runs",
        "prompt_a_score",
        "prompt_b_score",
        "prompt_a_na_rate",
        "prompt_b_na_rate",
        "n_prompt_conditions",
        "prompt_mean_eligible",
    ]
    return result[ordered_columns]


def main() -> None:
    models = selected_models()
    prompt_a = read_prompt_file(PROMPT_A_PATH, "PromptA")
    prompt_b = read_prompt_file(PROMPT_B_PATH, "PromptB")

    available_a = set(prompt_a["model_name"])
    available_b = set(prompt_b["model_name"])
    missing_a = sorted(set(models) - available_a)
    missing_b = sorted(set(models) - available_b)
    if missing_a or missing_b:
        raise RuntimeError(f"Models missing from scale files: PromptA={missing_a}, PromptB={missing_b}")
    prompt_a = prompt_a.loc[prompt_a["model_name"].isin(models)].copy()
    prompt_b = prompt_b.loc[prompt_b["model_name"].isin(models)].copy()

    primary = pd.concat(
        [aggregate_prompt(prompt_a), aggregate_prompt(prompt_b)],
        ignore_index=True,
    )
    prompt_mean = build_prompt_mean(primary)
    combined = pd.concat([primary, prompt_mean], ignore_index=True, sort=False)
    combined = combined.sort_values(
        ["prompt_condition", "model_name", "language", "scale", "dimension"]
    ).reset_index(drop=True)
    prompt_mean = prompt_mean.sort_values(KEY_COLUMNS).reset_index(drop=True)

    OUTPUT_LONG_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_LONG_PATH, index=False)
    prompt_mean.to_csv(OUTPUT_MEAN_PATH, index=False)

    print(f"Selected models: {len(models)}")
    print(f"PromptA/B rows after five-repeat aggregation: {len(primary)}")
    print(
        f"PromptMean rows: {len(prompt_mean)} "
        f"eligible={int(prompt_mean['prompt_mean_eligible'].sum())}"
    )
    print(f"Combined scale scores: {len(combined)} -> {OUTPUT_LONG_PATH}")
    print(f"PromptMean components: {len(prompt_mean)} -> {OUTPUT_MEAN_PATH}")


if __name__ == "__main__":
    main()
