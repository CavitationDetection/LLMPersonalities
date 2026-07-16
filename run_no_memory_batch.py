"""Run the formal bilingual no-memory questionnaire batch.

Inputs are the questionnaire/protocol workbooks and model/sheet allowlists
under Test_File. Each item is sent as an independent chat request. Parsed
answers are checkpointed after every item and exported as one workbook per
model and repeat. Literal N/A is a valid missing response, not a numeric zero.
The formal scale protocol sends max_tokens=32 and omits sampling parameters.
"""

import argparse
import csv
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parent
TEST_FILE_DIR = ROOT / "Test_File"
OUTPUTS_DIR = ROOT / "Outputs"
METADATA_DIR = ROOT / "Metadata"
QUESTIONNAIRE_FILE = TEST_FILE_DIR / "Scale_16_Q_F.xlsx"
PROTOCOL_ZH_FILE = TEST_FILE_DIR / "_AI_Protocol_C" / "_AI_Protocol.xlsx"
PROTOCOL_EN_FILE = TEST_FILE_DIR / "_AI_Protocol_E" / "_AI_Protocol.xlsx"
DEFAULT_SELECTED_MODELS_FILE = TEST_FILE_DIR / "API_Selected_Final9.txt"
DEFAULT_SELECTED_SHEETS_FILE = TEST_FILE_DIR / "Sheet_Selected_Final7.txt"
DEFAULT_RESULTS_ROOT = OUTPUTS_DIR / "Results_9API_7Scale_PromptB"
API_CONFIG_FILE = TEST_FILE_DIR / "api_config.json"
ITEM_KEY_MAP_FILE = METADATA_DIR / "questionnaire_item_key_map.csv"
REFERENCE_MAIN = ROOT.parent / "llm_emotion" / "main.py"

EXCLUDED_MODELS = {"grok-3-mini", "moonshot-v1-32k", "moonshot-v1-8k"}
DEFAULT_LANGUAGES = ("zh", "en")
DEFAULT_PROMPT = "B"
DEFAULT_REPEATS = 1
DEFAULT_REPEAT_START = 1
DEFAULT_MAX_RETRIES = 5
DEFAULT_TIMEOUT = 90
DEFAULT_RESPONSE_MAX_TOKENS = 32
DEFAULT_RETRY_BACKOFF_BASE = 2.0
DEFAULT_RETRY_BACKOFF_MAX = 20.0
DEFAULT_ORDER = "forward"
INVALID_ANSWER = "NO_VALID_ANSWER"


def os_env(name: str):
    value = __import__("os").environ.get(name)
    if value:
        value = value.strip()
    return value or None


def load_reference_config():
    base_url = (
        os_env("N1N_BASE_URL")
        or os_env("OPENAI_BASE_URL")
        or os_env("API_BASE_URL")
        or os_env("BASE_URL")
    )
    api_key = (
        os_env("N1N_API_KEY")
        or os_env("OPENAI_API_KEY")
        or os_env("API_KEY")
    )

    if base_url and api_key:
        return base_url, api_key, "environment"

    if not REFERENCE_MAIN.exists():
        raise RuntimeError("Cannot resolve API config from environment or reference file.")

    text = REFERENCE_MAIN.read_text(encoding="utf-8")
    base_match = re.search(r'BASE_URL\s*=\s*"([^"]+)"', text)
    key_match = re.search(r'API_KEY\s*=\s*"([^"]+)"', text)
    if not base_match or not key_match:
        raise RuntimeError("Failed to parse BASE_URL/API_KEY from reference file.")

    return base_match.group(1), key_match.group(1), str(REFERENCE_MAIN)


def bootstrap_local_config(base_url: str, api_key: str):
    if API_CONFIG_FILE.exists():
        return

    payload = {
        "base_url": base_url,
        "api_key": api_key,
    }
    API_CONFIG_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_api_config():
    base_url = (
        os_env("N1N_BASE_URL")
        or os_env("OPENAI_BASE_URL")
        or os_env("API_BASE_URL")
        or os_env("BASE_URL")
    )
    api_key = (
        os_env("N1N_API_KEY")
        or os_env("OPENAI_API_KEY")
        or os_env("API_KEY")
    )

    if base_url and api_key:
        return base_url, api_key, "environment"

    if API_CONFIG_FILE.exists():
        payload = json.loads(API_CONFIG_FILE.read_text(encoding="utf-8"))
        base_url = str(payload.get("base_url", "")).strip()
        api_key = str(payload.get("api_key", "")).strip()
        if base_url and api_key:
            return base_url, api_key, str(API_CONFIG_FILE)

    base_url, api_key, source = load_reference_config()
    bootstrap_local_config(base_url, api_key)
    return base_url, api_key, f"{source} -> {API_CONFIG_FILE}"


def parse_selection_file(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")

    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.search(r'"([^"]+)"', line)
        if match:
            items.append(match.group(1))
    return items


def parse_selected_models(models_file: Path):
    models = parse_selection_file(models_file, "Model selection")
    return [model for model in models if model not in EXCLUDED_MODELS]


def parse_selected_sheets(sheets_file: Path):
    return parse_selection_file(sheets_file, "Sheet selection")


def _legacy_normalize_language(language: str) -> str:
    value = language.strip().lower()
    if value in {"zh", "cn", "chinese", "中文", "中"}:
        return "zh"
    if value in {"en", "english", "英文", "英"}:
        return "en"
    raise ValueError(f"Unsupported language: {language}")


def load_protocol(language: str) -> str:
    protocol_file = PROTOCOL_ZH_FILE if language == "zh" else PROTOCOL_EN_FILE
    xl = pd.ExcelFile(protocol_file)
    chunks = []
    for sheet_name in xl.sheet_names:
        df = pd.read_excel(protocol_file, sheet_name=sheet_name, header=None)
        for value in df.fillna("").astype(str).to_numpy().flatten():
            value = value.strip()
            if value:
                chunks.append(value)
    if not chunks:
        raise RuntimeError(f"No protocol text found in {protocol_file}")
    return "\n".join(chunks)


def load_questionnaires():
    if not QUESTIONNAIRE_FILE.exists():
        raise FileNotFoundError(f"Questionnaire file not found: {QUESTIONNAIRE_FILE}")
    xl = pd.ExcelFile(QUESTIONNAIRE_FILE)
    data = {}
    for sheet_name in xl.sheet_names:
        data[sheet_name] = pd.read_excel(QUESTIONNAIRE_FILE, sheet_name=sheet_name).ffill()
    return data


def filter_questionnaires(questionnaires, allowed_sheets):
    allowed_lookup = set(allowed_sheets)
    missing = [sheet for sheet in allowed_sheets if sheet not in questionnaires]
    if missing:
        raise RuntimeError(f"Unknown questionnaire sheets requested: {missing}")
    return {
        sheet_name: df
        for sheet_name, df in questionnaires.items()
        if sheet_name in allowed_lookup
    }


def get_prompt_column(language: str, prompt_variant: str) -> str:
    return f"Prompt_{prompt_variant}_{'Chinese' if language == 'zh' else 'English'}"


def get_question_column(language: str) -> str:
    return "Question_Chinese" if language == "zh" else "Question_English"


def get_options_column(language: str) -> str:
    return "Options_Chinese" if language == "zh" else "Options_English"


def get_message_labels(language: str):
    if language == "zh":
        return {
            "questionnaire": "来源",
            "part": "部分",
            "item_number": "题号",
            "question": "题目",
            "options": "选项",
            "allowed_options": "合法选项",
            "na_part": "无",
            "sep": "：",
        }
    return {
        "questionnaire": "Source",
        "part": "Part",
        "item_number": "Item Number",
        "question": "Question",
        "options": "Options",
        "allowed_options": "Allowed Options",
        "na_part": "N/A",
        "sep": ":",
    }


def extract_allowed_options(options_text: str):
    if not options_text:
        return []

    lines = [line.strip() for line in str(options_text).splitlines() if line.strip()]
    numeric = []
    for line in lines:
        match = re.match(r"^(\d+)", line)
        if match:
            numeric.append(match.group(1))
    if numeric:
        return list(dict.fromkeys(numeric))

    alpha = []
    for line in lines:
        match = re.match(r"^([A-Z])\b", line, flags=re.I)
        if match:
            alpha.append(match.group(1).upper())
    return list(dict.fromkeys(alpha))


def _legacy_build_messages(sheet_name: str, row: pd.Series, language: str, prompt_variant: str, protocol_text: str):
    prompt_col = get_prompt_column(language, prompt_variant)
    question_col = get_question_column(language)
    options_col = get_options_column(language)

    prompt_text = str(row.get(prompt_col, "")).strip()
    question_text = str(row.get(question_col, "")).strip()
    options_text = str(row.get(options_col, "")).strip()
    option_range = str(row.get("Option_Range", "")).strip()

    if not prompt_text:
        raise RuntimeError(f"Prompt text is empty for sheet={sheet_name}, number={row.get('Number')}")
    if not question_text:
        raise RuntimeError(f"Question text is empty for sheet={sheet_name}, number={row.get('Number')}")

    system_message = protocol_text
    user_parts = [
        prompt_text,
        f"Questionnaire: {sheet_name}",
        f"Item Number: {int(row['Number'])}",
        f"Question:\n{question_text}",
    ]
    if options_text:
        user_parts.append(f"Options:\n{options_text}")
    if option_range:
        user_parts.append(f"Allowed Options: {option_range}")
    user_message = "\n\n".join(user_parts)
    return system_message, user_message, options_text


def _legacy_normalize_token(token: str):
    return token.strip().strip("`*[](){}<>\"'").strip(".,;:!?。；：！？ ").strip()


def _legacy_parse_answer(raw_answer: str, allowed_options):
    if raw_answer is None:
        return {"parsed": None, "status": "empty", "note": "answer is None"}

    answer = str(raw_answer).strip()
    if not answer:
        return {"parsed": None, "status": "empty", "note": "answer is empty"}

    normalized_answer = normalize_token(answer)
    if normalized_answer.upper() == "N/A":
        return {"parsed": "N/A", "status": "exact_na", "note": ""}

    if not allowed_options:
        return {"parsed": answer, "status": "no_validation", "note": ""}

    allowed_set = {str(option) for option in allowed_options}
    if normalized_answer in allowed_set:
        return {"parsed": normalized_answer, "status": "exact_option", "note": ""}

    explicit_candidates = []
    explicit_patterns = [
        r"(?i)(?:final answer|answer|option|答案|最终答案|选项)\s*[:：]?\s*(?:is\s*)?(N/A|[A-Z]|\d+)\b",
        r"(?i)\\boxed\{\s*(N/A|[A-Z]|\d+)\s*\}",
    ]
    for pattern in explicit_patterns:
        for match in re.finditer(pattern, answer):
            candidate = normalize_token(match.group(1)).upper()
            if candidate == "N/A":
                explicit_candidates.append("N/A")
            elif candidate in allowed_set:
                explicit_candidates.append(candidate)
    explicit_candidates = list(dict.fromkeys(explicit_candidates))
    if len(explicit_candidates) == 1:
        return {
            "parsed": explicit_candidates[0],
            "status": "explicit_marker",
            "note": "accepted via explicit answer marker",
        }
    if len(explicit_candidates) > 1:
        return {
            "parsed": None,
            "status": "ambiguous_marker",
            "note": f"multiple explicit candidates: {explicit_candidates}",
        }

    found = []
    for option in sorted(allowed_set, key=len, reverse=True):
        pattern = rf"(?<!\w){re.escape(option)}(?!\w)"
        if re.search(pattern, answer, flags=re.I):
            found.append(option)
    found = list(dict.fromkeys(found))
    if len(found) > 1:
        return {
            "parsed": None,
            "status": "ambiguous_multiple_options",
            "note": f"multiple options mentioned: {found}",
        }

    if len(found) == 1:
        return {
            "parsed": None,
            "status": "verbose_single_option_rejected",
            "note": f"single option {found[0]} mentioned in verbose text without explicit answer marker",
        }

    if "N/A" in answer.upper():
        return {
            "parsed": None,
            "status": "ambiguous_na",
            "note": "N/A mentioned inside longer response without exact match",
        }

    return {"parsed": None, "status": "unrecognized", "note": "no valid option recognized"}


def _legacy_make_retry_instruction(language: str, allowed_options):
    if not allowed_options:
        if language == "zh":
            return "你刚才的回答无效。请只输出一个合法答案，不要附加任何解释。"
        return "Your previous response was invalid. Reply with exactly one valid answer and nothing else."
    allowed = ", ".join(str(item) for item in allowed_options)
    if language == "zh":
        return (
            "你刚才的回答无效。"
            f"请只输出以下选项之一或 N/A：{allowed}。"
            "不要附加解释。"
        )
    return (
        "Your previous response was invalid. "
        f"Reply with exactly one of the following options or N/A: {allowed}. "
        "Do not add any explanation."
    )


def normalize_language(language: str) -> str:
    value = language.strip().lower()
    if value in {"zh", "cn", "chinese", "中文"}:
        return "zh"
    if value in {"en", "english", "英文"}:
        return "en"
    raise ValueError(f"Unsupported language: {language}")


def format_display_value(value) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            return str(int(value))
        return str(value).strip()

    text = str(value).strip()
    if not text:
        return ""
    try:
        numeric = float(text)
    except ValueError:
        return text
    if math.isfinite(numeric) and numeric.is_integer():
        return str(int(numeric))
    return text


def sanitize_key_fragment(value: str) -> str:
    text = re.sub(r"[^0-9A-Za-z]+", "_", str(value).strip())
    text = text.strip("_")
    return text or "X"


def build_item_specs(df: pd.DataFrame):
    number_labels = [format_display_value(value) for value in df["Number"].tolist()]
    number_counts = Counter(number_labels)
    seen_numbers = Counter()
    used_keys = set()
    specs = []

    for row_position, (_, row) in enumerate(df.iterrows()):
        number_label = format_display_value(row.get("Number"))
        part_label = format_display_value(row.get("Part"))
        seen_numbers[number_label] += 1

        if number_counts[number_label] > 1:
            part_fragment = sanitize_key_fragment(part_label or f"block_{seen_numbers[number_label]}")
            base_key = f"P{part_fragment}_Q{sanitize_key_fragment(number_label)}"
            if part_label:
                display_label = f"Part {part_label} | Q{number_label}"
            else:
                display_label = f"Q{number_label} ({seen_numbers[number_label]})"
        else:
            base_key = number_label
            display_label = f"Q{number_label}"

        item_key = base_key
        suffix = 2
        while item_key in used_keys:
            item_key = f"{base_key}_{suffix}"
            suffix += 1
        used_keys.add(item_key)

        specs.append(
            {
                "row_position": row_position,
                "item_key": item_key,
                "display_label": display_label,
                "number_label": number_label,
                "part_label": part_label,
            }
        )

    return specs


def write_item_key_map(questionnaires):
    rows = []
    for sheet_name, df in questionnaires.items():
        for spec in build_item_specs(df):
            row = df.iloc[spec["row_position"]]
            rows.append(
                {
                    "Questionnaire": sheet_name,
                    "Item_Key": spec["item_key"],
                    "Part": spec["part_label"],
                    "Number": spec["number_label"],
                    "Question_English": str(row.get("Question_English", "")).strip(),
                    "Question_Chinese": str(row.get("Question_Chinese", "")).strip(),
                }
            )

    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    output_file = ITEM_KEY_MAP_FILE
    pd.DataFrame(rows).to_csv(output_file, index=False, encoding="utf-8-sig")
    return output_file


def build_messages(sheet_name: str, row: pd.Series, language: str, prompt_variant: str, protocol_text: str):
    prompt_col = get_prompt_column(language, prompt_variant)
    question_col = get_question_column(language)
    options_col = get_options_column(language)
    labels = get_message_labels(language)

    prompt_text = str(row.get(prompt_col, "")).strip()
    question_text = str(row.get(question_col, "")).strip()
    options_text = str(row.get(options_col, "")).strip()
    option_range = str(row.get("Option_Range", "")).strip()

    if not prompt_text:
        raise RuntimeError(f"Prompt text is empty for sheet={sheet_name}, number={row.get('Number')}")
    if not question_text:
        raise RuntimeError(f"Question text is empty for sheet={sheet_name}, number={row.get('Number')}")

    system_message = protocol_text
    user_parts = [
        prompt_text,
        f"{labels['questionnaire']}{labels['sep']}{sheet_name}",
        f"{labels['part']}{labels['sep']}{format_display_value(row.get('Part')) or labels['na_part']}",
        f"{labels['item_number']}{labels['sep']}{format_display_value(row.get('Number'))}",
        f"{labels['question']}{labels['sep']}\n{question_text}",
    ]
    if options_text:
        user_parts.append(f"{labels['options']}{labels['sep']}\n{options_text}")
    if option_range:
        user_parts.append(f"{labels['allowed_options']}{labels['sep']}{option_range}")
    user_message = "\n\n".join(user_parts)
    return system_message, user_message, options_text


def normalize_token(token: str):
    return token.strip().strip("`*[](){}<>\"'").strip(".,;:!?。；：！？，").strip()


def parse_answer(raw_answer: str, allowed_options):
    if raw_answer is None:
        return {"parsed": None, "status": "empty", "note": "answer is None"}

    answer = str(raw_answer).strip()
    if not answer:
        return {"parsed": None, "status": "empty", "note": "answer is empty"}

    normalized_answer = normalize_token(answer)
    if normalized_answer.upper() == "N/A":
        return {"parsed": "N/A", "status": "exact_na", "note": ""}

    if not allowed_options:
        return {"parsed": answer, "status": "no_validation", "note": ""}

    allowed_set = {str(option) for option in allowed_options}
    if normalized_answer in allowed_set:
        return {"parsed": normalized_answer, "status": "exact_option", "note": ""}

    explicit_candidates = []
    explicit_patterns = [
        r"(?i)(?:final answer|answer|option|答案|最终答案|选项)\s*[:：]?\s*(?:is\s*)?(N/A|[A-Z]|\d+)\b",
        r"(?i)\\boxed\{\s*(N/A|[A-Z]|\d+)\s*\}",
    ]
    for pattern in explicit_patterns:
        for match in re.finditer(pattern, answer):
            candidate = normalize_token(match.group(1)).upper()
            if candidate == "N/A":
                explicit_candidates.append("N/A")
            elif candidate in allowed_set:
                explicit_candidates.append(candidate)

    explicit_candidates = list(dict.fromkeys(explicit_candidates))
    if len(explicit_candidates) == 1:
        return {
            "parsed": explicit_candidates[0],
            "status": "explicit_marker",
            "note": "accepted via explicit answer marker",
        }
    if len(explicit_candidates) > 1:
        return {
            "parsed": None,
            "status": "ambiguous_marker",
            "note": f"multiple explicit candidates: {explicit_candidates}",
        }

    found = []
    for option in sorted(allowed_set, key=len, reverse=True):
        pattern = rf"(?<!\w){re.escape(option)}(?!\w)"
        if re.search(pattern, answer, flags=re.I):
            found.append(option)
    found = list(dict.fromkeys(found))

    if len(found) > 1:
        return {
            "parsed": None,
            "status": "ambiguous_multiple_options",
            "note": f"multiple options mentioned: {found}",
        }
    if len(found) == 1:
        return {
            "parsed": None,
            "status": "verbose_single_option_rejected",
            "note": f"single option {found[0]} mentioned in verbose text without explicit answer marker",
        }
    if "N/A" in answer.upper():
        return {
            "parsed": None,
            "status": "ambiguous_na",
            "note": "N/A mentioned inside longer response without exact match",
        }
    return {"parsed": None, "status": "unrecognized", "note": "no valid option recognized"}


def make_retry_instruction(language: str, allowed_options):
    if not allowed_options:
        if language == "zh":
            return "你刚才的回答无效。请只输出一个合法答案值本身，例如 1、2、A 或 N/A。不要输出选项文本，不要附加任何解释。"
        return "Your previous response was invalid. Reply with only one valid answer value itself, such as 1, 2, A, or N/A. Do not include option text or explanation."

    allowed = ", ".join(str(item) for item in allowed_options)
    if language == "zh":
        return (
            "你刚才的回答无效。"
            f"请只输出以下选项之一或 N/A：{allowed}。"
            "请只输出选项值本身，不要输出选项文本，不要附加解释。"
        )
    return (
        "Your previous response was invalid. "
        f"Reply with exactly one of the following options or N/A: {allowed}. "
        "Reply with only the option value itself, not the option text, and do not add any explanation."
    )


def request_completion(
    base_url: str,
    api_key: str,
    model: str,
    system_message: str,
    user_message: str,
    timeout: int,
    response_max_tokens: int,
):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        "max_tokens": response_max_tokens,
    }
    response = requests.post(
        base_url.rstrip("/") + "/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
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


def retry_delay_seconds(attempt: int, backoff_base: float, backoff_max: float):
    return min(backoff_base * (2 ** (attempt - 1)), backoff_max)


def ask_with_retries(
    base_url: str,
    api_key: str,
    model: str,
    system_message: str,
    user_message: str,
    allowed_options,
    language: str,
    timeout: int,
    max_retries: int,
    response_max_tokens: int,
    retry_backoff_base: float,
    retry_backoff_max: float,
):
    attempts = []
    current_user_message = user_message
    for attempt in range(1, max_retries + 1):
        started = time.time()
        try:
            raw_answer = request_completion(
                base_url=base_url,
                api_key=api_key,
                model=model,
                system_message=system_message,
                user_message=current_user_message,
                timeout=timeout,
                response_max_tokens=response_max_tokens,
            )
            elapsed = round(time.time() - started, 2)
            parse_result = parse_answer(raw_answer, allowed_options)
            attempts.append(
                {
                    "attempt": attempt,
                    "raw_answer": raw_answer,
                    "parsed_answer": parse_result["parsed"],
                    "parse_status": parse_result["status"],
                    "parse_note": parse_result["note"],
                    "elapsed_seconds": elapsed,
                    "error": "",
                }
            )
            if parse_result["parsed"] is not None:
                return parse_result["parsed"], attempts
            current_user_message = user_message + "\n\n" + make_retry_instruction(language, allowed_options)
        except Exception as exc:
            elapsed = round(time.time() - started, 2)
            attempts.append(
                {
                    "attempt": attempt,
                    "raw_answer": "",
                    "parsed_answer": None,
                    "parse_status": "request_error",
                    "parse_note": "",
                    "elapsed_seconds": elapsed,
                    "error": str(exc),
                }
            )
            current_user_message = user_message + "\n\n" + make_retry_instruction(language, allowed_options)

        if attempt < max_retries and attempts[-1]["parse_status"] == "request_error":
            # Space out transport retries so transient gateway errors have time to clear.
            time.sleep(retry_delay_seconds(attempt, retry_backoff_base, retry_backoff_max))
    return INVALID_ANSWER, attempts


def clean_model_name(model_name: str):
    return model_name.replace("-", "_").replace("(", "").replace(")", "")


def repeat_results_dir(results_root: Path, repeat_index: int):
    return results_root / f"repeat_{repeat_index:02d}"


def checkpoint_path_for_model(model_name: str, results_dir: Path):
    return results_dir / f"{clean_model_name(model_name)}_checkpoint.json"


def output_path_for_model(model_name: str, results_dir: Path):
    return results_dir / f"{clean_model_name(model_name)}_results.xlsx"


def log_path_for_model(model_name: str, results_dir: Path):
    return results_dir / f"{clean_model_name(model_name)}_validation.log"


def initialize_checkpoint(model: str, questionnaires, languages, prompt_variant, repeat_index: int):
    sheets = {}
    for sheet_name in questionnaires.keys():
        sheets[sheet_name] = {}
        for language in languages:
            sheets[sheet_name][language] = {
                "answers": {},
                "completed": False,
            }
    return {
        "model": model,
        "prompt": prompt_variant,
        "repeat": repeat_index,
        "languages": list(languages),
        "sheets": sheets,
        "validation_lines": [f"Model: {model}", "=" * 60],
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def load_or_create_checkpoint(model: str, questionnaires, languages, prompt_variant, repeat_index: int, results_dir: Path):
    checkpoint_path = checkpoint_path_for_model(model, results_dir)
    if checkpoint_path.exists():
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    else:
        payload = initialize_checkpoint(model, questionnaires, languages, prompt_variant, repeat_index)

    if payload.get("model") != model:
        raise RuntimeError(f"Checkpoint model mismatch for {model}")
    if payload.get("prompt") != prompt_variant:
        raise RuntimeError(f"Checkpoint prompt mismatch for {model}")
    if payload.get("repeat") != repeat_index:
        raise RuntimeError(f"Checkpoint repeat mismatch for {model}: expected {repeat_index}, found {payload.get('repeat')}")

    payload.setdefault("validation_lines", [f"Model: {model}", "=" * 60])
    payload.setdefault("sheets", {})
    for sheet_name in questionnaires.keys():
        payload["sheets"].setdefault(sheet_name, {})
        for language in languages:
            payload["sheets"][sheet_name].setdefault(
                language,
                {"answers": {}, "completed": False},
            )
            payload["sheets"][sheet_name][language].setdefault("answers", {})
            payload["sheets"][sheet_name][language].setdefault("completed", False)
    return payload


def write_checkpoint(model_name: str, checkpoint, results_dir: Path):
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    checkpoint_path_for_model(model_name, results_dir).write_text(
        json.dumps(checkpoint, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def render_results_from_checkpoint(model_name: str, questionnaires, checkpoint, languages, prompt_variant, repeat_index: int):
    model_results = {}
    for sheet_name, df in questionnaires.items():
        item_specs = build_item_specs(df)
        sheet_rows = []
        for language in languages:
            row_state = checkpoint["sheets"][sheet_name][language]
            row_data = {
                "API": model_name,
                "Repeat": repeat_index,
                "Language": language,
                "Prompt": prompt_variant,
                "Memory": "none",
                "Order": DEFAULT_ORDER,
                "Completed": "yes" if row_state.get("completed") else "no",
            }
            for spec in item_specs:
                row_data[spec["item_key"]] = row_state["answers"].get(spec["item_key"], "")
            sheet_rows.append(row_data)
        model_results[sheet_name] = pd.DataFrame(sheet_rows)
    return model_results


def save_results(model_name: str, model_results, validation_lines, results_dir: Path):
    results_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_path_for_model(model_name, results_dir)
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        for sheet_name, df in model_results.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    log_file = log_path_for_model(model_name, results_dir)
    if len(validation_lines) > 2:
        log_file.write_text("\n".join(validation_lines), encoding="utf-8")
    elif log_file.exists():
        log_file.unlink()

    return output_file


def process_model(
    model: str,
    questionnaires,
    protocols,
    languages,
    prompt_variant,
    repeat_index: int,
    results_dir: Path,
    base_url,
    api_key,
    timeout,
    max_retries,
    response_max_tokens,
    retry_backoff_base,
    retry_backoff_max,
):
    checkpoint = load_or_create_checkpoint(
        model,
        questionnaires,
        languages,
        prompt_variant,
        repeat_index=repeat_index,
        results_dir=results_dir,
    )
    total_calls = 0

    for sheet_name, df in questionnaires.items():
        item_specs = build_item_specs(df)
        for language in languages:
            state = checkpoint["sheets"][sheet_name][language]
            if state.get("completed"):
                continue

            for spec in item_specs:
                row = df.iloc[spec["row_position"]]
                question_key = spec["item_key"]
                question_label = spec["display_label"]
                if question_key in state["answers"] and str(state["answers"][question_key]).strip():
                    continue

                system_message, user_message, options_text = build_messages(
                    sheet_name=sheet_name,
                    row=row,
                    language=language,
                    prompt_variant=prompt_variant,
                    protocol_text=protocols[language],
                )
                allowed_options = extract_allowed_options(options_text)
                answer, attempts = ask_with_retries(
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    system_message=system_message,
                    user_message=user_message,
                    allowed_options=allowed_options,
                    language=language,
                    timeout=timeout,
                    max_retries=max_retries,
                    response_max_tokens=response_max_tokens,
                    retry_backoff_base=retry_backoff_base,
                    retry_backoff_max=retry_backoff_max,
                )
                total_calls += len(attempts)
                state["answers"][question_key] = answer
                write_checkpoint(model, checkpoint, results_dir)

                if answer == INVALID_ANSWER:
                    attempt_summary = " | ".join(
                        f"attempt {item['attempt']}: status={item['parse_status']!r} raw={item['raw_answer']!r} error={item['error']!r} note={item['parse_note']!r}"
                        for item in attempts
                    )
                    checkpoint["validation_lines"].append(
                        f"{sheet_name} | {language} | {question_label}: {attempt_summary}"
                    )
                else:
                    final_attempt = attempts[-1]
                    if final_attempt["parse_status"] not in {"exact_option", "exact_na", "no_validation"}:
                        checkpoint["validation_lines"].append(
                            f"{sheet_name} | {language} | {question_label}: accepted via {final_attempt['parse_status']} raw={final_attempt['raw_answer']!r}"
                        )
                write_checkpoint(model, checkpoint, results_dir)

            state["completed"] = True
            write_checkpoint(model, checkpoint, results_dir)
            partial_results = render_results_from_checkpoint(
                model,
                questionnaires,
                checkpoint,
                languages,
                prompt_variant,
                repeat_index,
            )
            save_results(model, partial_results, checkpoint["validation_lines"], results_dir)

    model_results = render_results_from_checkpoint(
        model,
        questionnaires,
        checkpoint,
        languages,
        prompt_variant,
        repeat_index,
    )
    return model_results, checkpoint["validation_lines"], total_calls


def summarize_run(models, questionnaires, languages, repeats: int):
    item_total = sum(int(df["Number"].dropna().shape[0]) for df in questionnaires.values())
    per_model_requests = item_total * len(languages)
    return {
        "models": len(models),
        "sheets": len(questionnaires),
        "items": item_total,
        "repeats": repeats,
        "requests_per_model": per_model_requests,
        "requests_per_repeat": per_model_requests * len(models),
        "requests_total": per_model_requests * len(models) * repeats,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run the curated no-memory questionnaire batch with API defaults.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, choices=["A", "B"])
    parser.add_argument("--languages", nargs="+", default=list(DEFAULT_LANGUAGES))
    parser.add_argument("--models-file", default=str(DEFAULT_SELECTED_MODELS_FILE))
    parser.add_argument("--sheets-file", default=str(DEFAULT_SELECTED_SHEETS_FILE))
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--repeat-start", type=int, default=DEFAULT_REPEAT_START)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--max-models", type=int, default=None)
    parser.add_argument("--max-sheets", type=int, default=None)
    parser.add_argument("--sheets", nargs="*", default=None, help="Optional explicit questionnaire sheet allowlist.")
    parser.add_argument("--models", nargs="*", default=None, help="Optional explicit model allowlist.")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--response-max-tokens", type=int, default=DEFAULT_RESPONSE_MAX_TOKENS)
    parser.add_argument("--retry-backoff-base", type=float, default=DEFAULT_RETRY_BACKOFF_BASE)
    parser.add_argument("--retry-backoff-max", type=float, default=DEFAULT_RETRY_BACKOFF_MAX)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.repeats < 1:
        raise RuntimeError("--repeats must be at least 1.")
    if args.repeat_start < 1:
        raise RuntimeError("--repeat-start must be at least 1.")

    languages = [normalize_language(language) for language in args.languages]
    models_file = Path(args.models_file).resolve()
    sheets_file = Path(args.sheets_file).resolve()
    results_root = Path(args.results_root).resolve()

    models = parse_selected_models(models_file)
    if args.models:
        allowlist = set(args.models)
        models = [model for model in models if model in allowlist]
    if args.max_models is not None:
        models = models[: args.max_models]

    questionnaires = load_questionnaires()
    selected_sheets = parse_selected_sheets(sheets_file)
    questionnaires = filter_questionnaires(questionnaires, selected_sheets)
    if args.sheets:
        requested = set(args.sheets)
        questionnaires = {name: df for name, df in questionnaires.items() if name in requested}
    if args.max_sheets is not None:
        questionnaires = dict(list(questionnaires.items())[: args.max_sheets])

    protocols = {language: load_protocol(language) for language in languages}
    base_url, api_key, config_source = load_api_config()
    item_key_map_file = write_item_key_map(questionnaires)
    summary = summarize_run(models, questionnaires, languages, args.repeats)

    print(f"Config source: {config_source}")
    print(f"Models file: {models_file}")
    print(f"Sheets file: {sheets_file}")
    print(f"Prompt variant: {args.prompt}")
    print(f"Languages: {languages}")
    print(f"Order: {DEFAULT_ORDER}")
    print(f"Repeat start: {args.repeat_start}")
    print(f"Repeats: {args.repeats}")
    print(f"Timeout seconds: {args.timeout}")
    print(f"Max retries: {args.max_retries}")
    print(f"Response max tokens: {args.response_max_tokens}")
    print(
        "Request-error backoff: "
        f"base={args.retry_backoff_base}s max={args.retry_backoff_max}s"
    )
    print(f"Models: {summary['models']}")
    print(f"Sheets: {summary['sheets']}")
    print(f"Items total: {summary['items']}")
    print(f"Requests per model: {summary['requests_per_model']}")
    print(f"Requests per repeat: {summary['requests_per_repeat']}")
    print(f"Total requests (without retries): {summary['requests_total']}")
    print(f"Excluded models: {sorted(EXCLUDED_MODELS)}")
    print(f"Results root: {results_root}")
    print(f"Item key map: {item_key_map_file}")

    if args.dry_run:
        print("Dry run only. No API requests sent.")
        return

    if not models:
        raise RuntimeError("No models selected to run.")

    results_root.mkdir(parents=True, exist_ok=True)

    repeat_end = args.repeat_start + args.repeats - 1
    for repeat_index in range(args.repeat_start, repeat_end + 1):
        current_results_dir = repeat_results_dir(results_root, repeat_index)
        current_results_dir.mkdir(parents=True, exist_ok=True)
        print(f"=== Repeat {repeat_index} | Results dir: {current_results_dir} ===")

        for idx, model in enumerate(models, start=1):
            output_file = output_path_for_model(model, current_results_dir)
            checkpoint_file = checkpoint_path_for_model(model, current_results_dir)
            if args.skip_existing and output_file.exists():
                print(f"[repeat {repeat_index} | {idx}/{len(models)}] Skipping existing result for {model}")
                continue

            print(f"[repeat {repeat_index} | {idx}/{len(models)}] Running model: {model}")
            if checkpoint_file.exists():
                print(f"  resuming from checkpoint: {checkpoint_file}")
            started = time.time()
            model_results, validation_lines, total_calls = process_model(
                model=model,
                questionnaires=questionnaires,
                protocols=protocols,
                languages=languages,
                prompt_variant=args.prompt,
                repeat_index=repeat_index,
                results_dir=current_results_dir,
                base_url=base_url,
                api_key=api_key,
                timeout=args.timeout,
                max_retries=args.max_retries,
                response_max_tokens=args.response_max_tokens,
                retry_backoff_base=args.retry_backoff_base,
                retry_backoff_max=args.retry_backoff_max,
            )
            save_path = save_results(model, model_results, validation_lines, current_results_dir)
            elapsed = round(time.time() - started, 2)
            print(f"  saved: {save_path}")
            print(f"  elapsed: {elapsed}s | api calls including retries: {total_calls}")
            if checkpoint_file.exists():
                checkpoint_file.unlink()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
