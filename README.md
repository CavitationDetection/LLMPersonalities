# Task23 API Experiment Code

This directory is a **code-only consolidation** of `Task23/API_Exp`, preserving the formal implementations for scale measurement, downstream task generation, and their correlation analysis. Consolidated on 2026-07-16.

This directory does **not** include item banks, model lists, API keys, raw responses, scored results, images, Slurm logs, or historical archives. At runtime, place the external inputs listed below at the same relative paths.

## Directory Structure

```text
Code_Only/
├── README.md
├── requirements.txt
├── Submit.sh
├── run_no_memory_batch.py
├── Tools/
│   ├── README.md
│   ├── diagnose_scale_responses.py
│   ├── example_single_turn_test.py
│   └── read_results_workbook.py
└── Downstream/
    ├── README.md
    ├── Submit_Downstream.sh
    ├── Run_Part2_Analysis.sh
    └── scripts/
        ├── build_downstream_tasks.py
        ├── run_downstream.py
        ├── parse_downstream.py
        ├── aggregate_downstream.py
        ├── prepare_scale_scores.py
        └── analyze_correlations.py
```

> Note: The actual directory name uses Chinese characters (`下游任务`) to match the original project layout. All documentation and comments are in English.

## Two Formal Pipelines

### 1. Scale Measurement

The formal design uses 9 models, 7 scales, Chinese and English, PromptA/PromptB tested separately, with 5 repeats per condition. The 7 scales contain 157 items, so the planned calls per prompt condition are:

```text
9 models × 157 items × 2 languages × 5 repeats = 14,130
```

Scale request behavior:

- Each item is an independent single-turn request with no context memory.
- The payload sends `model`, `messages`, and `max_tokens=32`.
- No `temperature`, `top_p`, `seed`, or reasoning parameters are passed.
- Only valid options, explicit answer markers, or an entire response of `N/A` are accepted.
- Up to 5 attempts by default; after a format error, a **format-only** reminder is appended to the same item.
- A checkpoint is written after every item; interrupted runs resume from incomplete items.
- Formal workbooks store parsed answers, not the full raw JSON of every valid request.
- To audit raw responses, token usage, finish/status information, use `Tools/diagnose_scale_responses.py`.

Main entry point:

```bash
sbatch Submit.sh
```

PromptA and PromptB must write to different result directories. For example:

```bash
PROMPT_VARIANT=A RESULTS_ROOT=Outputs/Results_9API_7Scale_PromptA sbatch Submit.sh
PROMPT_VARIANT=B RESULTS_ROOT=Outputs/Results_9API_7Scale_PromptB sbatch Submit.sh
```

Submit one repeat at a time via `REPEAT_START=1..5`; use `--skip-existing` to avoid overwriting completed workbooks.

Detailed scale-side instructions are in `Tools/README.md` and the module docstrings.

### 2. Downstream Tasks and Correlation Analysis

The downstream design uses 9 models, 24 tasks, Chinese and English, 5 repeats, for 2,160 formal requests. Tasks include 18 MFV scenario items, 3 OUS Instrumental Harm items, and 3 OUS Impartial Beneficence allocation items.

Downstream request behavior:

- Each item is an independent single-turn request with no context memory.
- The payload sends only `model` and `messages`; no generation parameters or `max_tokens` are explicitly passed.
- On format errors, empty answers, or request errors, the identical prompt is resent up to 5 times with no repair prompt.
- Only a response exactly equal to `N/A` is recorded as a valid N/A; valid N/A is not scored as zero and is not retried.
- Every raw record stores the full prompt, prompt SHA-256, raw answer, attempt count, elapsed time, and error.

Generation and parsing entry point:

```bash
cd 下游任务
sbatch Submit_Downstream.sh
```

Aggregation and correlation analysis entry point:

```bash
cd 下游任务
./Run_Part2_Analysis.sh
```

Correlation analysis uses PromptA and PromptB scale scores as primary inputs, and additionally computes the equal-weight `PromptMean=(PromptA+PromptB)/2` as a sensitivity analysis. The five repeats are aggregated within each model first; the unit of correlation analysis is always the model, never the repeat.

Per-script inputs/outputs and N/A aggregation rules for the downstream side are in `下游任务/README.md`.

## External Inputs

At runtime, the code needs at least the following files. They are excluded from this directory because they belong to item banks, configuration, or data:

```text
Test_File/Scale_16_Q_F.xlsx
Test_File/_AI_Protocol_C/_AI_Protocol.xlsx
Test_File/_AI_Protocol_E/_AI_Protocol.xlsx
Test_File/API_Selected_Final9.txt
Test_File/Sheet_Selected_Final7.txt
Downstream/data/model_list.json
统计/PromptA/最终结果_原始文献复核/final_scores_by_run_language_official_only.csv
统计/PromptB/最终结果_原始文献复核/final_scores_by_run_language_official_only.csv
```

API credentials are read from environment variables first:

```bash
export N1N_BASE_URL="https://example.com/v1"
export N1N_API_KEY="..."
```

`OPENAI_BASE_URL`/`OPENAI_API_KEY`, `API_BASE_URL`/`API_KEY`, or a local `Test_File/api_config.json` are also supported:

```json
{
  "base_url": "https://example.com/v1",
  "api_key": "..."
}
```

Do not put real keys into the code package or version control.

## Install Dependencies

```bash
python -m pip install -r requirements.txt
```

Main dependencies are `requests`, `pandas`, `openpyxl`, `numpy`, `scipy`, and `matplotlib`.

## No-API Checks

Scale dry-run:

```bash
python run_no_memory_batch.py --dry-run --prompt A --repeats 1
```

Downstream dry-run:

```bash
cd 下游任务
DOWNSTREAM_DRY_RUN=1 python scripts/run_downstream.py
```

Compile and shell syntax checks:

```bash
python -m compileall -q .
bash -n Submit.sh
bash -n 下游任务/Submit_Downstream.sh
bash -n 下游任务/Run_Part2_Analysis.sh
```

## Reproducibility Boundaries

- `build_downstream_tasks.py` is the code source for downstream items and frozen prompts; running it produces `Downstream/prompts/downstream_tasks.json`.
- Model names and scale selection are determined by external allowlists, not hard-coded in analysis scripts.
- Final scale score tables are statistical data products and are not in the code-only directory; downstream analysis only reads official-dimension CSVs.
- Results, figures, and historical scripts under `Outputs/`, `统计/`, and `Archive/` are not part of this code package.
- If formal prompts are modified, start a new independent result batch; do not mix with existing prompt hashes or workbooks.
