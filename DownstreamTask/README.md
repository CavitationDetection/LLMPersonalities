# Downstream Task Code

This directory contains code for building, API-calling, strictly parsing, five-repeat aggregating, and correlating downstream scenario tasks with PromptA/PromptB scale scores. Only code is included here; the task JSON, model list, raw responses, and analysis results are generated or provided as external runtime files.

## Script Execution Order

```text
scripts/build_downstream_tasks.py
        ↓
scripts/run_downstream.py
        ↓
scripts/parse_downstream.py
        ↓
scripts/aggregate_downstream.py
        ↓
scripts/prepare_scale_scores.py
        ↓
scripts/analyze_correlations.py
```

`Submit_Downstream.sh` runs the first three steps and prints a quality summary; `Run_Part2_Analysis.sh` runs the last three.

## Script Responsibilities

### `build_downstream_tasks.py`

- Defines 18 MFV scenario items and 6 OUS downstream items in code.
- Generates frozen Chinese/English system/user prompts for each item.
- Writes `prompts/downstream_tasks.json`.
- The task JSON is a generated artifact; changing items or prompts requires editing this script and starting a new formal result batch.

### `run_downstream.py`

- Reads enabled models from `data/model_list.json`.
- Generates a unique request key per model, task, language, and repeat.
- Payload contains only `model` and `messages`; no `temperature`, `top_p`, `seed`, or `max_tokens`.
- Resends the identical prompt on failure or invalid format, up to 5 times; no "repair prompt" is appended.
- Existing records are skipped by unique key unless overwrite is explicitly set.
- Validates `prompt_sha256` for already-written records to prevent mixing prompt versions.
- Outputs `outputs/downstream_raw_responses.jsonl`.

### `parse_downstream.py`

- Re-parses raw responses independently by task type, without relying on the online validator's format decision.
- MFV accepts integers 1 to 5; OUS IH accepts integers 1 to 7; OUS IB accepts only integer `A=... B=... C=...` allocations summing to 100.
- Only a response exactly equal to `N/A` is treated as a valid N/A.
- Outputs `downstream_parsed_responses.csv` and `downstream_scores_long.csv`.

### `aggregate_downstream.py`

- For each model, item, and language, averages available scores across 5 repeats.
- N/A is not scored as zero and is not imputed; it is excluded from the mean denominator, and `n_na`/`na_rate` are retained.
- If all five repeats for an item are N/A, the item mean is empty.
- At the dimension level, available item means are averaged with equal weight, while the actual number of scored items is saved.
- Outputs task aggregates, dimension long/wide tables, and repeat stability tables.

### `prepare_scale_scores.py`

- Reads official scale-dimension CSVs for PromptA and PromptB.
- Each prompt condition is first aggregated over 5 repeats within model, language, scale, and dimension.
- `PromptMean` is computed only when both prompt-specific means are available, using `(PromptA + PromptB) / 2`.
- Retains only the MFQ-30 and OUS-9 official dimensions required by downstream analysis.

### `analyze_correlations.py`

- Computes Spearman correlations at the model level; the 5 repeats are never treated as independent observations.
- MFQ-30 vs. MFV outputs the full 5×5 matrix and theoretical matched relations.
- OUS outputs matched correlations for Instrumental Harm and Impartial Beneficence.
- Missing values use pairwise complete cases; N/A rates are quality indicators only, not scored as zero or used as correlation weights.
- Computes BH-FDR, leave-one-model-out ranges, language consistency, prompt-condition comparisons, and figures.

## Full N/A Data Path

A valid N/A is recorded in the parse table as:

```text
response_valid = true
is_na = true
score_available = false
score_value = missing
```

Example: if one item has 4 numeric responses and 1 N/A across five repeats, the item mean uses only the 4 numeric values and `na_rate=0.2`. If one item in a three-item dimension is entirely N/A across all five repeats, the dimension mean uses the remaining two items and records `n_tasks_scored=2`. A model is pair-wise excluded from a correlation only when the entire downstream dimension has no available score.

## Run Generation and Parsing

```bash
sbatch Submit_Downstream.sh
```

Main environment variables:

```text
DOWNSTREAM_REPEAT_START       default 1
DOWNSTREAM_N_REPEATS          default 5
DOWNSTREAM_LANGUAGES          default "zh en"
DOWNSTREAM_TIMEOUT            default 90
DOWNSTREAM_MAX_RETRIES        default 5
DOWNSTREAM_RETRY_BACKOFF_BASE default 2
DOWNSTREAM_RETRY_BACKOFF_MAX  default 20
DOWNSTREAM_OVERWRITE          default 0
DOWNSTREAM_DRY_RUN            default 0
```

Dry-run sends no API requests:

```bash
DOWNSTREAM_DRY_RUN=1 python scripts/run_downstream.py
```

## Run Aggregation and Correlation Analysis

```bash
./Run_Part2_Analysis.sh
```

Scale input paths:

```text
../统计/PromptA/最终结果_原始文献复核/final_scores_by_run_language_official_only.csv
../统计/PromptB/最终结果_原始文献复核/final_scores_by_run_language_official_only.csv
```

Main outputs:

```text
outputs/downstream_task_aggregates.csv
outputs/downstream_dimension_aggregates_long.csv
outputs/repeat_stability.csv
outputs/analysis_summary.csv
outputs/main_matched_correlations.csv
outputs/mfq_mfv_correlation_long.csv
outputs/ous_correlation_summary.csv
outputs/correlation_model_pairs.csv
outputs/ANALYSIS_SUMMARY.md
outputs/figures/
```

With only 9 models, all significance results should be interpreted as exploratory and reported together with N/A rates, repeat stability, language direction, and leave-one-model-out results.
