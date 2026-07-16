"""Relate model-level scale scores to model-level downstream behavior.

The analysis uses Spearman correlations with pairwise-complete models, BH-FDR,
leave-one-model-out checks, prompt-condition comparisons, and bilingual
stability summaries. Repeats are aggregated before correlation and are never
treated as independent observations.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
SCALE_SCORES_PATH = ROOT / "data" / "scale_scores_prompt_long.csv"
DOWNSTREAM_DIMENSION_PATH = ROOT / "outputs" / "downstream_dimension_aggregates_long.csv"
OUTPUT_ANALYSIS_PATH = ROOT / "outputs" / "analysis_summary.csv"
OUTPUT_MFQ_LONG_PATH = ROOT / "outputs" / "mfq_mfv_correlation_long.csv"
OUTPUT_MAIN_PATH = ROOT / "outputs" / "main_matched_correlations.csv"
OUTPUT_OUS_PATH = ROOT / "outputs" / "ous_correlation_summary.csv"
OUTPUT_PAIRS_PATH = ROOT / "outputs" / "correlation_model_pairs.csv"
OUTPUT_PROMPT_COMPARISON_PATH = ROOT / "outputs" / "prompt_condition_comparison.csv"
OUTPUT_LANGUAGE_PATH = ROOT / "outputs" / "language_consistency.csv"
OUTPUT_DIAGONAL_PATH = ROOT / "outputs" / "mfq_diagonal_specificity.csv"
OUTPUT_README_PATH = ROOT / "outputs" / "ANALYSIS_SUMMARY.md"
FIGURES_DIR = ROOT / "outputs" / "figures"

PROMPT_CONDITIONS = ["PromptA", "PromptB", "PromptMean"]
LANGUAGES = ["en", "zh"]
MFQ_DIMENSIONS = ["Harm", "Fairness", "Ingroup", "Authority", "Purity"]
MFV_DIMENSIONS = ["MFV_Care", "MFV_Fairness", "MFV_Loyalty", "MFV_Authority", "MFV_Purity"]
MFQ_DIAGONAL = {
    "Harm": "MFV_Care",
    "Fairness": "MFV_Fairness",
    "Ingroup": "MFV_Loyalty",
    "Authority": "MFV_Authority",
    "Purity": "MFV_Purity",
}
OUS_DIAGONAL = {
    "Instrumental_Harm": "OUS_IH_downstream",
    "Impartial_Beneficence": "OUS_IB_downstream",
}
DOWNSTREAM_LANGUAGE_DIMENSIONS = [
    "MFV_Care",
    "MFV_Fairness",
    "MFV_Liberty",
    "MFV_Authority",
    "MFV_Loyalty",
    "MFV_Purity",
    "OUS_IH_downstream",
    "OUS_IB_downstream",
]
MODEL_LABELS = {
    "gpt-5.4-2026-03-05": "GPT-5.4",
    "gpt-5.4-mini-2026-03-17": "GPT-5.4 mini",
    "gemini-2.5-flash": "Gemini Flash",
    "gemini-2.5-flash-lite": "Gemini Lite",
    "claude-sonnet-4-5-20250929": "Claude Sonnet",
    "claude-haiku-4-5-20251001": "Claude Haiku",
    "deepseek-v3.2": "DeepSeek",
    "doubao-seed-1-8-251228": "Doubao",
    "kimi-k2-250905": "Kimi",
}


def safe_spearman(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(valid) < 3 or valid["x"].nunique() < 2 or valid["y"].nunique() < 2:
        return np.nan, np.nan
    result = spearmanr(valid["x"], valid["y"])
    return float(result.statistic), float(result.pvalue)


def leave_one_out_rhos(pairs: pd.DataFrame) -> list[float]:
    if len(pairs) < 5:
        return []
    values = []
    for index in pairs.index:
        subset = pairs.drop(index=index)
        rho, _ = safe_spearman(subset["scale_score"], subset["downstream_score"])
        if not math.isnan(rho):
            values.append(rho)
    return values


def paired_data(
    scales: pd.DataFrame,
    downstream: pd.DataFrame,
    prompt_condition: str,
    language: str,
    scale: str,
    scale_dimension: str,
    downstream_dimension: str,
) -> pd.DataFrame:
    scale_rows = scales.loc[
        scales["prompt_condition"].eq(prompt_condition)
        & scales["language"].eq(language)
        & scales["scale"].eq(scale)
        & scales["dimension"].eq(scale_dimension),
        ["model_name", "scale_score", "mean_na_rate", "n_valid_runs"],
    ].copy()
    downstream_rows = downstream.loc[
        downstream["language"].eq(language)
        & downstream["downstream_dimension"].eq(downstream_dimension),
        ["model_name", "mean_score", "mean_repeat_std", "na_rate", "n_valid_repeats"],
    ].rename(
        columns={
            "mean_score": "downstream_score",
            "mean_repeat_std": "downstream_repeat_std",
            "na_rate": "downstream_na_rate",
            "n_valid_repeats": "downstream_n_valid_repeats",
        }
    )
    pairs = scale_rows.merge(downstream_rows, on="model_name", how="inner", validate="one_to_one")
    return pairs.dropna(subset=["scale_score", "downstream_score"]).sort_values("model_name").reset_index(drop=True)


def correlation_record(
    pairs: pd.DataFrame,
    *,
    prompt_condition: str,
    language: str,
    scale: str,
    scale_dimension: str,
    downstream_dimension: str,
    relation_type: str,
) -> dict:
    rho, p_value = safe_spearman(pairs["scale_score"], pairs["downstream_score"])
    loo = leave_one_out_rhos(pairs)
    downstream_range = 4.0 if downstream_dimension.startswith("MFV_") else 100.0
    mean_repeat_std = pairs["downstream_repeat_std"].mean() if not pairs.empty else np.nan
    return {
        "prompt_condition": prompt_condition,
        "scale": scale,
        "scale_dimension": scale_dimension,
        "downstream_dimension": downstream_dimension,
        "language": language,
        "relation_type": relation_type,
        "expected_direction": "positive" if relation_type == "matched" else "exploratory",
        "n_models": int(len(pairs)),
        "spearman_rho": rho,
        "p_value": p_value,
        "mean_scale_score": pairs["scale_score"].mean() if not pairs.empty else np.nan,
        "mean_scale_na_rate": pairs["mean_na_rate"].mean() if not pairs.empty else np.nan,
        "mean_downstream_score": pairs["downstream_score"].mean() if not pairs.empty else np.nan,
        "mean_downstream_na_rate": pairs["downstream_na_rate"].mean() if not pairs.empty else np.nan,
        "mean_repeat_std": mean_repeat_std,
        "normalized_repeat_std": mean_repeat_std / downstream_range if not math.isnan(mean_repeat_std) else np.nan,
        "loo_min_rho": min(loo) if loo else np.nan,
        "loo_max_rho": max(loo) if loo else np.nan,
        "loo_positive_fraction": sum(value > 0 for value in loo) / len(loo) if loo else np.nan,
    }


def add_fdr(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["p_fdr_bh"] = np.nan
    valid = frame["p_value"].dropna().sort_values()
    if valid.empty:
        return frame
    count = len(valid)
    adjusted = valid.to_numpy() * count / np.arange(1, count + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    frame.loc[valid.index, "p_fdr_bh"] = np.minimum(adjusted, 1.0)
    return frame


def assign_evidence(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["evidence_level"] = "exploratory"
    frame["notes"] = "Off-diagonal exploratory association."
    matched = frame["relation_type"].eq("matched")
    for index, row in frame.loc[matched].iterrows():
        counterpart = frame.loc[
            matched
            & frame["prompt_condition"].eq(row["prompt_condition"])
            & frame["scale"].eq(row["scale"])
            & frame["scale_dimension"].eq(row["scale_dimension"])
            & frame["downstream_dimension"].eq(row["downstream_dimension"])
            & ~frame["language"].eq(row["language"])
        ]
        rho = row["spearman_rho"]
        same_positive_direction = (
            not math.isnan(rho)
            and rho > 0
            and not counterpart.empty
            and counterpart["spearman_rho"].notna().all()
            and counterpart["spearman_rho"].gt(0).all()
        )
        loo_robust = not math.isnan(row["loo_positive_fraction"]) and row["loo_positive_fraction"] == 1
        stable = not math.isnan(row["normalized_repeat_std"]) and row["normalized_repeat_std"] <= 0.15
        if same_positive_direction and rho >= 0.50 and loo_robust and stable:
            level = "strong"
        elif rho >= 0.30 and rho > 0 and stable:
            level = "moderate"
        else:
            level = "weak_or_none"
        notes = []
        if row["n_models"] < 9:
            notes.append(f"pairwise n={int(row['n_models'])} due to missing scale score")
        if not same_positive_direction:
            notes.append("direction is not positive in both languages")
        if not loo_robust:
            notes.append("leave-one-model-out direction is not uniformly positive")
        if not stable:
            notes.append("downstream repeat variability exceeds threshold")
        if row["prompt_condition"] == "PromptMean":
            notes.append("secondary equal-weight PromptA/PromptB sensitivity analysis")
        frame.at[index, "evidence_level"] = level
        frame.at[index, "notes"] = "; ".join(notes) if notes else "criteria satisfied"
    return frame


def build_correlations(scales: pd.DataFrame, downstream: pd.DataFrame):
    rows = []
    pair_rows = []
    for prompt_condition in PROMPT_CONDITIONS:
        for language in LANGUAGES:
            for scale_dimension in MFQ_DIMENSIONS:
                for downstream_dimension in MFV_DIMENSIONS:
                    relation_type = (
                        "matched"
                        if MFQ_DIAGONAL[scale_dimension] == downstream_dimension
                        else "off_diagonal"
                    )
                    pairs = paired_data(
                        scales,
                        downstream,
                        prompt_condition,
                        language,
                        "MFQ-30",
                        scale_dimension,
                        downstream_dimension,
                    )
                    rows.append(
                        correlation_record(
                            pairs,
                            prompt_condition=prompt_condition,
                            language=language,
                            scale="MFQ-30",
                            scale_dimension=scale_dimension,
                            downstream_dimension=downstream_dimension,
                            relation_type=relation_type,
                        )
                    )
                    for _, pair in pairs.iterrows():
                        pair_rows.append(
                            {
                                "prompt_condition": prompt_condition,
                                "language": language,
                                "scale": "MFQ-30",
                                "scale_dimension": scale_dimension,
                                "downstream_dimension": downstream_dimension,
                                "relation_type": relation_type,
                                **pair.to_dict(),
                            }
                        )
            for scale_dimension, downstream_dimension in OUS_DIAGONAL.items():
                pairs = paired_data(
                    scales,
                    downstream,
                    prompt_condition,
                    language,
                    "OUS-9",
                    scale_dimension,
                    downstream_dimension,
                )
                rows.append(
                    correlation_record(
                        pairs,
                        prompt_condition=prompt_condition,
                        language=language,
                        scale="OUS-9",
                        scale_dimension=scale_dimension,
                        downstream_dimension=downstream_dimension,
                        relation_type="matched",
                    )
                )
                for _, pair in pairs.iterrows():
                    pair_rows.append(
                        {
                            "prompt_condition": prompt_condition,
                            "language": language,
                            "scale": "OUS-9",
                            "scale_dimension": scale_dimension,
                            "downstream_dimension": downstream_dimension,
                            "relation_type": "matched",
                            **pair.to_dict(),
                        }
                    )
    correlations = assign_evidence(add_fdr(pd.DataFrame(rows)))
    pairs = pd.DataFrame(pair_rows)
    return correlations, pairs


def write_mfq_matrices(mfq: pd.DataFrame) -> None:
    for prompt_condition in PROMPT_CONDITIONS:
        for language in LANGUAGES:
            subset = mfq.loc[
                mfq["prompt_condition"].eq(prompt_condition) & mfq["language"].eq(language)
            ]
            for value, suffix in [
                ("spearman_rho", ""),
                ("p_value", "_pvalues"),
                ("n_models", "_n_models"),
            ]:
                matrix = subset.pivot(
                    index="scale_dimension", columns="downstream_dimension", values=value
                ).reindex(index=MFQ_DIMENSIONS, columns=MFV_DIMENSIONS)
                matrix.to_csv(
                    ROOT
                    / "outputs"
                    / f"mfq_mfv_correlation_matrix_{prompt_condition}_{language}{suffix}.csv"
                )


def build_prompt_comparison(main: pd.DataFrame) -> pd.DataFrame:
    index_columns = ["scale", "scale_dimension", "downstream_dimension", "language"]
    rho = main.pivot(index=index_columns, columns="prompt_condition", values="spearman_rho")
    n_models = main.pivot(index=index_columns, columns="prompt_condition", values="n_models")
    result = rho.reset_index()
    result.columns.name = None
    for prompt in PROMPT_CONDITIONS:
        if prompt not in result:
            result[prompt] = np.nan
        result = result.rename(columns={prompt: f"rho_{prompt}"})
    result["rho_delta_b_minus_a"] = result["rho_PromptB"] - result["rho_PromptA"]
    result["prompt_a_b_direction_change"] = (
        np.sign(result["rho_PromptA"]) != np.sign(result["rho_PromptB"])
    ) & result[["rho_PromptA", "rho_PromptB"]].notna().all(axis=1)
    n_models = n_models.add_prefix("n_").reset_index()
    n_models.columns.name = None
    result = result.merge(n_models, on=index_columns, how="left", validate="one_to_one")
    return result.sort_values(index_columns).reset_index(drop=True)


def build_diagonal_specificity(mfq: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (prompt_condition, language), group in mfq.groupby(
        ["prompt_condition", "language"], sort=True
    ):
        diagonal = group.loc[group["relation_type"].eq("matched"), "spearman_rho"].dropna()
        off_diagonal = group.loc[
            group["relation_type"].eq("off_diagonal"), "spearman_rho"
        ].dropna()
        rows.append(
            {
                "prompt_condition": prompt_condition,
                "language": language,
                "n_diagonal": int(len(diagonal)),
                "n_off_diagonal": int(len(off_diagonal)),
                "mean_diagonal_rho": diagonal.mean(),
                "median_diagonal_rho": diagonal.median(),
                "mean_off_diagonal_rho": off_diagonal.mean(),
                "median_off_diagonal_rho": off_diagonal.median(),
                "mean_abs_diagonal_rho": diagonal.abs().mean(),
                "mean_abs_off_diagonal_rho": off_diagonal.abs().mean(),
                "abs_specificity_difference": diagonal.abs().mean() - off_diagonal.abs().mean(),
                "signed_specificity_difference": diagonal.mean() - off_diagonal.mean(),
                "diagonal_abs_stronger": diagonal.abs().mean() > off_diagonal.abs().mean(),
            }
        )
    return pd.DataFrame(rows)


def build_language_consistency(downstream: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dimension in DOWNSTREAM_LANGUAGE_DIMENSIONS:
        subset = downstream.loc[
            downstream["downstream_dimension"].eq(dimension),
            ["model_name", "language", "mean_score", "mean_repeat_std", "na_rate"],
        ]
        en = subset.loc[subset["language"].eq("en")].drop(columns="language").add_suffix("_en")
        zh = subset.loc[subset["language"].eq("zh")].drop(columns="language").add_suffix("_zh")
        pairs = en.merge(
            zh,
            left_on="model_name_en",
            right_on="model_name_zh",
            how="inner",
            validate="one_to_one",
        )
        rho, p_value = safe_spearman(pairs["mean_score_en"], pairs["mean_score_zh"])
        rows.append(
            {
                "dimension": dimension,
                "en_mean": pairs["mean_score_en"].mean(),
                "zh_mean": pairs["mean_score_zh"].mean(),
                "en_std": pairs["mean_score_en"].std(ddof=1),
                "zh_std": pairs["mean_score_zh"].std(ddof=1),
                "en_zh_difference": pairs["mean_score_zh"].mean() - pairs["mean_score_en"].mean(),
                "spearman_rho": rho,
                "p_value": p_value,
                "n_models": int(len(pairs)),
                "mean_en_repeat_std": pairs["mean_repeat_std_en"].mean(),
                "mean_zh_repeat_std": pairs["mean_repeat_std_zh"].mean(),
                "mean_en_na_rate": pairs["na_rate_en"].mean(),
                "mean_zh_na_rate": pairs["na_rate_zh"].mean(),
            }
        )
    return pd.DataFrame(rows)


def file_fragment(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")


def plot_scatter(pairs: pd.DataFrame, row: pd.Series) -> None:
    if pairs.empty:
        return
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8.2, 6.2), facecolor="#f4f0e6")
    axis.set_facecolor("#fbf8f0")
    axis.scatter(
        pairs["scale_score"],
        pairs["downstream_score"],
        s=80,
        c="#0f6b68",
        edgecolor="#183a3a",
        linewidth=0.8,
        alpha=0.9,
    )
    for _, pair in pairs.iterrows():
        axis.annotate(
            MODEL_LABELS.get(pair["model_name"], pair["model_name"]),
            (pair["scale_score"], pair["downstream_score"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7.4,
            color="#263333",
        )
    axis.grid(color="#c9c2b3", linewidth=0.7, alpha=0.45)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_xlabel(f"{row['scale']} {row['scale_dimension']} score")
    axis.set_ylabel(row["downstream_dimension"])
    rho_text = "NA" if pd.isna(row["spearman_rho"]) else f"{row['spearman_rho']:.3f}"
    p_text = "NA" if pd.isna(row["p_value"]) else f"{row['p_value']:.3g}"
    axis.set_title(
        f"{row['prompt_condition']} | {row['language']} | rho={rho_text}, p={p_text}, n={int(row['n_models'])}",
        loc="left",
        fontsize=12,
        fontweight="bold",
        color="#173b3a",
    )
    figure.tight_layout()
    filename = (
        f"{row['prompt_condition']}_{row['language']}_{file_fragment(row['scale_dimension'])}"
        f"_vs_{file_fragment(row['downstream_dimension'])}.png"
    )
    figure.savefig(FIGURES_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_heatmap(matrix: pd.DataFrame, prompt_condition: str, language: str) -> None:
    figure, axis = plt.subplots(figsize=(8.2, 6.6), facecolor="#f4f0e6")
    values = matrix.to_numpy(dtype=float)
    image = axis.imshow(values, cmap="RdYlBu_r", vmin=-1, vmax=1, aspect="auto")
    axis.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=35, ha="right")
    axis.set_yticks(range(len(matrix.index)), matrix.index)
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            label = "NA" if np.isnan(value) else f"{value:.2f}"
            axis.text(column_index, row_index, label, ha="center", va="center", fontsize=9)
    axis.set_title(
        f"MFQ-MFV Spearman matrix | {prompt_condition} | {language}",
        loc="left",
        fontweight="bold",
        color="#173b3a",
    )
    figure.colorbar(image, ax=axis, label="Spearman rho", shrink=0.82)
    figure.tight_layout()
    figure.savefig(
        FIGURES_DIR / f"mfq_mfv_heatmap_{prompt_condition}_{language}.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_language(downstream: pd.DataFrame, row: pd.Series) -> None:
    dimension = row["dimension"]
    subset = downstream.loc[
        downstream["downstream_dimension"].eq(dimension),
        ["model_name", "language", "mean_score"],
    ]
    pairs = subset.pivot(index="model_name", columns="language", values="mean_score").dropna()
    if pairs.empty:
        return
    figure, axis = plt.subplots(figsize=(7.4, 6.2), facecolor="#f4f0e6")
    axis.set_facecolor("#fbf8f0")
    axis.scatter(pairs["en"], pairs["zh"], s=80, c="#c65d2e", edgecolor="#62341f")
    low = min(pairs["en"].min(), pairs["zh"].min())
    high = max(pairs["en"].max(), pairs["zh"].max())
    axis.plot([low, high], [low, high], linestyle="--", color="#6f746e", linewidth=1)
    for model, pair in pairs.iterrows():
        axis.annotate(
            MODEL_LABELS.get(model, model),
            (pair["en"], pair["zh"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7.4,
        )
    axis.grid(color="#c9c2b3", linewidth=0.7, alpha=0.45)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_xlabel("English mean")
    axis.set_ylabel("Chinese mean")
    axis.set_title(
        f"{dimension} language consistency | rho={row['spearman_rho']:.3f}",
        loc="left",
        fontweight="bold",
        color="#713719",
    )
    figure.tight_layout()
    figure.savefig(
        FIGURES_DIR / f"language_{file_fragment(dimension)}_en_vs_zh.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def write_summary_markdown(
    main: pd.DataFrame, language: pd.DataFrame, diagonal: pd.DataFrame
) -> None:
    n_fdr_significant = int(main["p_fdr_bh"].lt(0.05).sum())
    lines = [
        "# Downstream correlation analysis",
        "",
        "PromptA and PromptB are primary analyses. PromptMean is a secondary sensitivity analysis computed as the equal-weight mean of the two prompt-specific five-repeat scale means.",
        "",
        "The correlation unit is the model. Spearman p-values are exploratory because there are only nine models and multiple relations.",
        f"Among the 42 matched tests, {n_fdr_significant} have Benjamini-Hochberg FDR-adjusted p < 0.05.",
        "",
        "## Matched relations",
        "",
        "| Prompt | Language | Scale relation | n | rho | p | Evidence |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for _, row in main.sort_values(
        ["scale", "scale_dimension", "prompt_condition", "language"]
    ).iterrows():
        relation = f"{row['scale_dimension']} -> {row['downstream_dimension']}"
        rho = "NA" if pd.isna(row["spearman_rho"]) else f"{row['spearman_rho']:.3f}"
        p_value = "NA" if pd.isna(row["p_value"]) else f"{row['p_value']:.3g}"
        lines.append(
            f"| {row['prompt_condition']} | {row['language']} | {relation} | "
            f"{int(row['n_models'])} | {rho} | {p_value} | {row['evidence_level']} |"
        )
    lines.extend(
        [
            "",
            "## MFQ diagonal specificity",
            "",
            "Positive absolute specificity means diagonal correlations are stronger on average than off-diagonal correlations.",
            "",
            "| Prompt | Language | Mean diagonal rho | Mean off-diagonal rho | Absolute specificity |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for _, row in diagonal.sort_values(["prompt_condition", "language"]).iterrows():
        lines.append(
            f"| {row['prompt_condition']} | {row['language']} | "
            f"{row['mean_diagonal_rho']:.3f} | {row['mean_off_diagonal_rho']:.3f} | "
            f"{row['abs_specificity_difference']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Language consistency",
            "",
            "| Downstream dimension | n | rho | p | zh-en mean |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in language.iterrows():
        rho = "NA" if pd.isna(row["spearman_rho"]) else f"{row['spearman_rho']:.3f}"
        p_value = "NA" if pd.isna(row["p_value"]) else f"{row['p_value']:.3g}"
        lines.append(
            f"| {row['dimension']} | {int(row['n_models'])} | {rho} | {p_value} | "
            f"{row['en_zh_difference']:.3f} |"
        )
    OUTPUT_README_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    scales = pd.read_csv(SCALE_SCORES_PATH)
    downstream = pd.read_csv(DOWNSTREAM_DIMENSION_PATH)
    correlations, pairs = build_correlations(scales, downstream)

    mfq = correlations.loc[correlations["scale"].eq("MFQ-30")].copy()
    main = correlations.loc[correlations["relation_type"].eq("matched")].copy()
    ous = main.loc[main["scale"].eq("OUS-9")].copy()
    prompt_comparison = build_prompt_comparison(main)
    language = build_language_consistency(downstream)
    diagonal = build_diagonal_specificity(mfq)

    correlations.to_csv(OUTPUT_ANALYSIS_PATH, index=False)
    mfq.to_csv(OUTPUT_MFQ_LONG_PATH, index=False)
    main.to_csv(OUTPUT_MAIN_PATH, index=False)
    ous.to_csv(OUTPUT_OUS_PATH, index=False)
    pairs.to_csv(OUTPUT_PAIRS_PATH, index=False)
    prompt_comparison.to_csv(OUTPUT_PROMPT_COMPARISON_PATH, index=False)
    language.to_csv(OUTPUT_LANGUAGE_PATH, index=False)
    diagonal.to_csv(OUTPUT_DIAGONAL_PATH, index=False)
    write_mfq_matrices(mfq)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for _, row in main.iterrows():
        relation_pairs = paired_data(
            scales,
            downstream,
            row["prompt_condition"],
            row["language"],
            row["scale"],
            row["scale_dimension"],
            row["downstream_dimension"],
        )
        plot_scatter(relation_pairs, row)
    for prompt_condition in PROMPT_CONDITIONS:
        for language_code in LANGUAGES:
            subset = mfq.loc[
                mfq["prompt_condition"].eq(prompt_condition)
                & mfq["language"].eq(language_code)
            ]
            matrix = subset.pivot(
                index="scale_dimension", columns="downstream_dimension", values="spearman_rho"
            ).reindex(index=MFQ_DIMENSIONS, columns=MFV_DIMENSIONS)
            plot_heatmap(matrix, prompt_condition, language_code)
    for _, row in language.iterrows():
        plot_language(downstream, row)
    write_summary_markdown(main, language, diagonal)

    print(f"All correlation rows: {len(correlations)} -> {OUTPUT_ANALYSIS_PATH}")
    print(f"MFQ matrix rows: {len(mfq)} -> {OUTPUT_MFQ_LONG_PATH}")
    print(f"Matched relations: {len(main)} -> {OUTPUT_MAIN_PATH}")
    print(f"OUS matched rows: {len(ous)} -> {OUTPUT_OUS_PATH}")
    print(f"Model pairs: {len(pairs)} -> {OUTPUT_PAIRS_PATH}")
    print(f"Prompt comparison: {len(prompt_comparison)} -> {OUTPUT_PROMPT_COMPARISON_PATH}")
    print(f"Language consistency: {len(language)} -> {OUTPUT_LANGUAGE_PATH}")
    print(f"MFQ diagonal specificity: {len(diagonal)} -> {OUTPUT_DIAGONAL_PATH}")
    print(f"Figures: {len(list(FIGURES_DIR.glob('*.png')))} -> {FIGURES_DIR}")
    print(f"Summary: {OUTPUT_README_PATH}")


if __name__ == "__main__":
    main()
