"""Strictly parse and score raw downstream responses.

Parsing is repeated independently of the online validator. Exact N/A remains a
valid response with no numeric score; malformed responses remain invalid. The
script writes both an audit table and a score-ready long table.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TASKS_PATH = ROOT / "prompts" / "downstream_tasks.json"
OUTPUT_RAW_PATH = ROOT / "outputs" / "downstream_raw_responses.jsonl"
OUTPUT_SCORES_PATH = ROOT / "outputs" / "downstream_scores_long.csv"
OUTPUT_PARSED_PATH = ROOT / "outputs" / "downstream_parsed_responses.csv"

OVERWRITE = True


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            records.append({"line_number": line_number, "error": f"json_decode_error: {exc}"})
    return records


def parse_integer(raw: str, valid_min: int, valid_max: int) -> tuple[float | None, bool, str]:
    text = (raw or "").strip()
    if not re.fullmatch(r"\d+", text):
        return None, False, "expected_single_integer"
    value = int(text)
    if not valid_min <= value <= valid_max:
        return None, False, f"integer_out_of_range_{value}"
    return float(value), True, ""


def is_exact_na(raw: str) -> bool:
    return (raw or "").strip().upper() == "N/A"


def parse_allocation(raw: str) -> tuple[dict[str, int], bool, str]:
    text = (raw or "").strip()
    pattern = r"\b([ABC])\s*(?:=|:|：)?\s*(-?\d+(?:\.\d+)?)"
    matches = re.findall(pattern, text, flags=re.I)
    values: dict[str, int] = {}
    for raw_label, raw_value in matches:
        label = raw_label.upper()
        if label in values:
            return values, False, f"duplicate_{label}"
        if not re.fullmatch(r"\d+", raw_value):
            return values, False, f"{label}_not_nonnegative_integer"
        value = int(raw_value)
        if not 0 <= value <= 100:
            return values, False, f"{label}_out_of_range"
        values[label] = value
    for label in ["A", "B", "C"]:
        if label not in values:
            return values, False, f"missing_{label}"
    remainder = re.sub(pattern, "", text, flags=re.I)
    if re.sub(r"[\s,，;；]+", "", remainder):
        return values, False, "unexpected_text"
    total = sum(values.values())
    if total == 100:
        return values, True, ""
    return values, False, f"invalid_total_{total:g}"


def score_record(record: dict, task: dict) -> tuple[float | None, bool, str]:
    raw = record.get("raw_response", "")
    if record.get("error"):
        return None, False, str(record.get("error"))
    if is_exact_na(raw):
        return None, True, ""
    task_type = task.get("task_type")
    if task_type == "rating_1_5":
        return parse_integer(raw, 1, 5)
    if task_type == "rating_1_7":
        rating, valid, warning = parse_integer(raw, 1, 7)
        if not valid or rating is None:
            return None, valid, warning
        return (rating - 1) / 6 * 100, True, ""
    if task_type == "allocation_abc":
        values, valid, warning = parse_allocation(raw)
        if not valid:
            return None, False, warning
        return values.get("B"), True, warning
    return None, False, f"unknown_task_type_{task_type}"


def main() -> None:
    tasks = {task["task_id"]: task for task in load_json(TASKS_PATH)}
    records = read_jsonl(OUTPUT_RAW_PATH)
    parsed_rows = []
    score_rows = []
    for record in records:
        task = tasks.get(record.get("task_id"), {})
        score_name = task.get("scoring", {}).get("score_name")
        score_value, valid, parse_error = score_record(record, task) if task else (None, False, "unknown_task")
        is_na = bool(valid and is_exact_na(record.get("raw_response", "")))
        score_available = bool(valid and not is_na and score_value is not None)
        parsed_rows.append(
            {
                "model_name": record.get("model_name"),
                "instrument": record.get("instrument"),
                "scale": record.get("scale"),
                "dimension": record.get("dimension"),
                "task_id": record.get("task_id"),
                "task_type": record.get("task_type"),
                "language": record.get("language"),
                "repeat_id": record.get("repeat_id"),
                "raw_response": record.get("raw_response"),
                "prompt_sha256": record.get("prompt_sha256"),
                "attempt_count": record.get("attempt_count"),
                "parsed_score_name": score_name,
                "parsed_score_value": score_value,
                "response_valid": bool(valid),
                "is_na": is_na,
                "score_available": score_available,
                "valid": bool(valid),
                "parse_error": parse_error,
                "api_error": record.get("error"),
                "timestamp": record.get("timestamp"),
            }
        )
        score_rows.append(
            {
                "model_name": record.get("model_name"),
                "instrument": record.get("instrument"),
                "scale": record.get("scale"),
                "dimension": record.get("dimension"),
                "task_id": record.get("task_id"),
                "language": record.get("language"),
                "repeat_id": record.get("repeat_id"),
                "score_name": score_name,
                "score_value": score_value,
                "response_valid": bool(valid),
                "is_na": is_na,
                "score_available": score_available,
                "valid": bool(valid),
                "raw_response": record.get("raw_response"),
                "prompt_sha256": record.get("prompt_sha256"),
                "error": parse_error,
            }
        )

    parsed = pd.DataFrame(parsed_rows)
    scores = pd.DataFrame(score_rows)
    OUTPUT_PARSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    parsed.to_csv(OUTPUT_PARSED_PATH, index=False)
    scores.to_csv(OUTPUT_SCORES_PATH, index=False)
    n_invalid = int((~scores["response_valid"]).sum()) if not scores.empty else 0
    n_na = int(scores["is_na"].sum()) if not scores.empty else 0
    n_scored = int(scores["score_available"].sum()) if not scores.empty else 0
    print(f"Parsed records: {len(parsed)} -> {OUTPUT_PARSED_PATH}")
    print(
        f"Score rows: {len(scores)} scored={n_scored} na={n_na} invalid={n_invalid} "
        f"-> {OUTPUT_SCORES_PATH}"
    )


if __name__ == "__main__":
    main()
