# Scale Helper Tools

Scripts in this directory are used for previewing, diagnosing, and reading scale results. They are not part of the formal batch-calling main path. The formal entry points are `Submit.sh` and `run_no_memory_batch.py` in the parent directory.

## `example_single_turn_test.py`

Purpose: select one scale item and print the final system/user messages; the API is called only when `--run` is explicitly added.

Preview example:

```bash
python Tools/example_single_turn_test.py \
  --model gpt-5.4-2026-03-05 \
  --sheet "IPIP BFFM-50" \
  --number 1 \
  --language zh \
  --prompt A
```

Send the request and display the full returned JSON:

```bash
python Tools/example_single_turn_test.py \
  --model gpt-5.4-2026-03-05 \
  --sheet "IPIP BFFM-50" \
  --number 1 \
  --language en \
  --prompt B \
  --run \
  --show-json
```

Note: this tool is for manual spot checks. It uses the same message-construction rules as the formal batch script, but it does not write into formal repeat directories.

## `diagnose_scale_responses.py`

Purpose: draw one random item from the formal 7 scales using a seed, call all final models with the same item, and save the full request and raw response. This tool is specifically for inspecting `max_tokens=32`, reasoning tokens, finish reason, empty content, and parse status.

```bash
python Tools/diagnose_scale_responses.py \
  --prompt A \
  --max-tokens 32 \
  --timeout 90 \
  --seed 20260716
```

Default output directory:

```text
Outputs/Diagnostics/scale_max_tokens32_9api_<timestamp>/
```

Main files:

- `raw_responses.jsonl`: per-model full request, HTTP status, response JSON, visible text, usage, reasoning/output tokens, and parse result.
- `summary.csv`: flat summary for easy screening.
- `SUMMARY.md`: human-readable summary.
- `test_metadata.json`: random seed, selected item, models, and call parameters.

The diagnostic script does not retry by default, so follow-up attempts are not mixed into the first raw response.

## `read_results_workbook.py`

Purpose: read a formal `*_results.xlsx` and ensure literal `N/A` answers are not automatically converted to `NaN` by pandas.

```bash
python Tools/read_results_workbook.py \
  Outputs/Results_9API_7Scale_PromptA/repeat_01/<model>_results.xlsx
```

This tool is read-only and does not modify workbooks.

## Credentials and Data

The three tools share credential and data sources with the formal runner:

- `Test_File/api_config.json` or API environment variables.
- `Test_File/Scale_16_Q_F.xlsx`.
- Chinese and English `_AI_Protocol.xlsx`.
- `API_Selected_Final9.txt` and `Sheet_Selected_Final7.txt`.

No diagnostic output may contain a reusable API key; response JSON may retain model returns and token usage, but the request Authorization header must not be written.
