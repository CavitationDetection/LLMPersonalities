"""Call all enabled models for every downstream task, language, and repeat.

The request payload intentionally contains only model and messages. Invalid
format and transport retries resend the identical frozen prompt. Every record
stores raw text and a prompt hash in outputs/downstream_raw_responses.jsonl;
existing unique keys are skipped unless overwrite is explicitly enabled.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
API_EXP_ROOT = ROOT.parent

MODEL_LIST_PATH = ROOT / "data" / "model_list.json"
TASKS_PATH = ROOT / "prompts" / "downstream_tasks.json"
API_CONFIG_PATH = API_EXP_ROOT / "Test_File" / "api_config.json"
OUTPUT_RAW_PATH = ROOT / "outputs" / "downstream_raw_responses.jsonl"

LANGUAGES = os.environ.get("DOWNSTREAM_LANGUAGES", "zh en").split()
REPEAT_START = int(os.environ.get("DOWNSTREAM_REPEAT_START", "1"))
N_REPEATS = int(os.environ.get("DOWNSTREAM_N_REPEATS", "5"))
TIMEOUT = int(os.environ.get("DOWNSTREAM_TIMEOUT", "90"))
MAX_RETRIES = int(os.environ.get("DOWNSTREAM_MAX_RETRIES", "5"))
RETRY_BACKOFF_BASE = float(os.environ.get("DOWNSTREAM_RETRY_BACKOFF_BASE", "2"))
RETRY_BACKOFF_MAX = float(os.environ.get("DOWNSTREAM_RETRY_BACKOFF_MAX", "20"))
OVERWRITE = os.environ.get("DOWNSTREAM_OVERWRITE", "0") == "1"
DRY_RUN = os.environ.get("DOWNSTREAM_DRY_RUN", "0") == "1"
DRY_RUN_LIMIT = int(os.environ.get("DOWNSTREAM_DRY_RUN_LIMIT", "8"))


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_api_config() -> tuple[str, str]:
    base_url = (
        os.environ.get("N1N_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("API_BASE_URL")
        or os.environ.get("BASE_URL")
    )
    api_key = (
        os.environ.get("N1N_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("API_KEY")
    )
    if base_url and api_key:
        return base_url.strip(), api_key.strip()
    payload = load_json(API_CONFIG_PATH)
    return str(payload["base_url"]).strip(), str(payload["api_key"]).strip()


def enabled_models() -> list[str]:
    models = []
    for item in load_json(MODEL_LIST_PATH):
        if isinstance(item, str):
            models.append(item)
        elif item.get("enabled", True):
            models.append(str(item["model_name"]))
    return models


def existing_keys() -> set[tuple[str, str, str, int]]:
    keys: set[tuple[str, str, str, int]] = set()
    if OVERWRITE or not OUTPUT_RAW_PATH.exists():
        return keys
    for line in OUTPUT_RAW_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        keys.add(
            (
                str(record.get("model_name")),
                str(record.get("task_id")),
                str(record.get("language")),
                int(record.get("repeat_id")),
            )
        )
    return keys


def parse_integer(raw: str, valid_min: int, valid_max: int) -> int | None:
    text = (raw or "").strip()
    if not re.fullmatch(r"\d+", text):
        return None
    value = int(text)
    if valid_min <= value <= valid_max:
        return value
    return None


def is_exact_na(raw: str) -> bool:
    return (raw or "").strip().upper() == "N/A"


def parse_allocation(raw: str) -> tuple[dict[str, int], str]:
    text = (raw or "").strip()
    pattern = r"\b([ABC])\s*(?:=|:|：)?\s*(-?\d+(?:\.\d+)?)"
    matches = re.findall(pattern, text, flags=re.I)
    values: dict[str, int] = {}
    for raw_label, raw_value in matches:
        label = raw_label.upper()
        if label in values:
            return values, f"duplicate_{label}"
        if not re.fullmatch(r"\d+", raw_value):
            return values, f"{label}_not_nonnegative_integer"
        value = int(raw_value)
        if not 0 <= value <= 100:
            return values, f"{label}_out_of_range"
        values[label] = value
    for label in ["A", "B", "C"]:
        if label not in values:
            return values, f"missing_{label}"
    remainder = re.sub(pattern, "", text, flags=re.I)
    if re.sub(r"[\s,，;；]+", "", remainder):
        return values, "unexpected_text"
    total = sum(values.values())
    if total == 100:
        return values, ""
    return values, f"invalid_total_{total:g}"


def output_is_valid(task: dict, raw_response: str) -> tuple[bool, str]:
    if is_exact_na(raw_response):
        return True, ""
    task_type = task["task_type"]
    if task_type == "rating_1_5":
        value = parse_integer(raw_response, 1, 5)
        return (value is not None, "expected_one_integer_1_to_5")
    if task_type == "rating_1_7":
        value = parse_integer(raw_response, 1, 7)
        return (value is not None, "expected_one_integer_1_to_7")
    if task_type == "allocation_abc":
        _, warning = parse_allocation(raw_response)
        return (warning == "", warning or "")
    return False, f"unknown_task_type_{task_type}"


def prompt_sha256(system_prompt: str, user_prompt: str) -> str:
    content = json.dumps(
        {"system_prompt": system_prompt, "user_prompt": user_prompt},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def validate_existing_prompt_versions(tasks: list[dict]) -> None:
    if not OUTPUT_RAW_PATH.exists():
        return
    expected = {
        (task["task_id"], language): prompt_sha256(
            prompt_pack["system_prompt"], prompt_pack["user_prompt"]
        )
        for task in tasks
        for language, prompt_pack in task["languages"].items()
    }
    for line_number, line in enumerate(
        OUTPUT_RAW_PATH.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        key = (str(record.get("task_id")), str(record.get("language")))
        if key not in expected:
            continue
        recorded_hash = str(
            record.get("prompt_sha256")
            or prompt_sha256(
                str(record.get("system_prompt", "")),
                str(record.get("user_prompt", "")),
            )
        )
        if recorded_hash != expected[key]:
            raise RuntimeError(
                f"Prompt version mismatch at {OUTPUT_RAW_PATH}:{line_number} for {key}. "
                "Archive or clear the old formal run before changing prompts."
            )


def request_completion(base_url: str, api_key: str, model: str, system_prompt: str, user_prompt: str) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    response = requests.post(
        base_url.rstrip("/") + "/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=TIMEOUT,
    )
    if not response.ok:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:400]}")
    data = response.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "\n".join(parts).strip()
    return str(content).strip()


def retry_delay(attempt: int) -> float:
    return min(RETRY_BACKOFF_BASE * (2 ** (attempt - 1)), RETRY_BACKOFF_MAX)


def run_one_request(base_url: str, api_key: str, model: str, task: dict, language: str, repeat_id: int) -> dict:
    prompt_pack = task["languages"][language]
    error = None
    raw_response = ""
    started = time.time()
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw_response = request_completion(
                base_url=base_url,
                api_key=api_key,
                model=model,
                system_prompt=prompt_pack["system_prompt"],
                user_prompt=prompt_pack["user_prompt"],
            )
            valid, validation_note = output_is_valid(task, raw_response)
            if valid:
                error = None
                break
            error = f"invalid_response_format: {validation_note}"
            if attempt < MAX_RETRIES:
                time.sleep(retry_delay(attempt))
        except Exception as exc:
            error = str(exc)
            if attempt < MAX_RETRIES:
                time.sleep(retry_delay(attempt))
    return {
        "model_name": model,
        "instrument": task["instrument"],
        "scale": task["scale"],
        "dimension": task["dimension"],
        "task_id": task["task_id"],
        "task_type": task["task_type"],
        "language": language,
        "repeat_id": repeat_id,
        "system_prompt": prompt_pack["system_prompt"],
        "user_prompt": prompt_pack["user_prompt"],
        "prompt_sha256": prompt_sha256(prompt_pack["system_prompt"], prompt_pack["user_prompt"]),
        "raw_response": raw_response,
        "attempt_count": attempt,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(time.time() - started, 2),
        "error": error,
    }


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def dry_run(models: list[str], tasks: list[dict], repeat_ids: list[int]) -> None:
    shown = 0
    for model in models:
        for task in tasks:
            for language in LANGUAGES:
                for repeat_id in repeat_ids:
                    print("=" * 80)
                    print(f"model={model} task={task['task_id']} language={language} repeat={repeat_id}")
                    print(task["languages"][language]["system_prompt"])
                    print("---")
                    print(task["languages"][language]["user_prompt"])
                    shown += 1
                    if shown >= DRY_RUN_LIMIT:
                        return


def main() -> None:
    models = enabled_models()
    tasks = load_json(TASKS_PATH)
    validate_existing_prompt_versions(tasks)
    repeat_ids = list(range(REPEAT_START, REPEAT_START + N_REPEATS))
    total = len(models) * len(tasks) * len(LANGUAGES) * len(repeat_ids)
    print(
        f"Models={len(models)} tasks={len(tasks)} languages={LANGUAGES} "
        f"repeat_ids={repeat_ids} total_requests={total}"
    )
    print("API payload fields: model, messages")
    if DRY_RUN:
        dry_run(models, tasks, repeat_ids)
        return

    base_url, api_key = load_api_config()
    done = existing_keys()
    completed = 0
    skipped = 0
    for repeat_id in repeat_ids:
        for model in models:
            for task in tasks:
                for language in LANGUAGES:
                    key = (model, task["task_id"], language, repeat_id)
                    if key in done:
                        skipped += 1
                        continue
                    record = run_one_request(base_url, api_key, model, task, language, repeat_id)
                    append_jsonl(OUTPUT_RAW_PATH, record)
                    completed += 1
                    done.add(key)
                    status = "error" if record["error"] else "ok"
                    print(
                        f"{completed + skipped}/{total} {status} {model} "
                        f"{task['task_id']} {language} repeat={repeat_id}"
                    )
    print(f"Finished. skipped={skipped} new_records={completed} output={OUTPUT_RAW_PATH}")


if __name__ == "__main__":
    main()
