"""Preview or send one questionnaire item as a single-turn request.

Preview mode is the default and performs no API call. Use this helper to inspect
the final system/user messages before a batch, not to create formal results.
"""

import argparse
import csv
import json
import math
import re
import sys
import textwrap
import time
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parent.parent
TEST_FILE_DIR = ROOT / "Test_File"
OUTPUTS_DIR = ROOT / "Outputs"
QUESTIONNAIRE_FILE = TEST_FILE_DIR / "Scale_16_Q_F.xlsx"
PROTOCOL_ZH_FILE = TEST_FILE_DIR / "_AI_Protocol_C" / "_AI_Protocol.xlsx"
PROTOCOL_EN_FILE = TEST_FILE_DIR / "_AI_Protocol_E" / "_AI_Protocol.xlsx"
AVAILABILITY_CSV = OUTPUTS_DIR / "Availability" / "models_availability_report.csv"
API_CONFIG_FILE = TEST_FILE_DIR / "api_config.json"
REFERENCE_MAIN = ROOT.parent / "llm_emotion" / "main.py"


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

    API_CONFIG_FILE.write_text(
        json.dumps({"base_url": base_url, "api_key": api_key}, ensure_ascii=False, indent=2),
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


def os_env(name: str):
    value = __import__("os").environ.get(name)
    if value:
        value = value.strip()
    return value or None


def load_questionnaire(sheet_name: str) -> pd.DataFrame:
    if not QUESTIONNAIRE_FILE.exists():
        raise FileNotFoundError(f"Questionnaire file not found: {QUESTIONNAIRE_FILE}")
    df = pd.read_excel(QUESTIONNAIRE_FILE, sheet_name=sheet_name)
    return df.ffill()


def load_protocol(language: str) -> str:
    protocol_file = PROTOCOL_ZH_FILE if language == "zh" else PROTOCOL_EN_FILE
    if not protocol_file.exists():
        raise FileNotFoundError(f"Protocol file not found: {protocol_file}")

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


def pick_default_model() -> str:
    if not AVAILABILITY_CSV.exists():
        return "gpt-4o-mini-2024-07-18"

    with AVAILABILITY_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("available") == "yes":
                model = (row.get("model_id") or "").strip()
                if model and model not in {"grok-3-mini", "moonshot-v1-32k", "moonshot-v1-8k"}:
                    return model

    return "gpt-4o-mini-2024-07-18"


def normalize_prompt_variant(prompt_variant: str) -> str:
    value = prompt_variant.strip().upper()
    if value not in {"A", "B"}:
        raise ValueError("prompt_variant must be A or B")
    return value


def get_prompt_column(language: str, prompt_variant: str) -> str:
    if language == "zh":
        return f"Prompt_{prompt_variant}_Chinese"
    return f"Prompt_{prompt_variant}_English"


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


def resolve_single_item(df: pd.DataFrame, number: int, part: str | None):
    number_mask = df["Number"].apply(format_display_value) == str(number)
    matched = df[number_mask]
    if matched.empty:
        raise RuntimeError(f"Item {number} not found.")

    if part is not None:
        part_value = str(part).strip()
        part_mask = matched["Part"].apply(format_display_value) == part_value
        matched = matched[part_mask]
        if matched.empty:
            raise RuntimeError(f"Item {number} with Part {part_value} not found.")
        return matched.iloc[0]

    if len(matched) > 1:
        available_parts = ", ".join(sorted({format_display_value(value) for value in matched["Part"].tolist()}))
        raise RuntimeError(
            f"Item {number} appears multiple times in this sheet. Please add --part. Available parts: {available_parts}"
        )

    return matched.iloc[0]


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
    return system_message, user_message


def call_model(model: str, system_message: str, user_message: str):
    base_url, api_key, config_source = load_api_config()
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        "max_tokens": 256,
    }

    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    started = time.time()
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    elapsed = time.time() - started

    raw_body = response.text
    if not response.ok:
        raise RuntimeError(f"HTTP {response.status_code}: {raw_body[:500]}")

    data = response.json()
    content = data["choices"][0]["message"].get("content", "")
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        answer = "\n".join(parts).strip()
    else:
        answer = str(content).strip()

    return {
        "answer": answer,
        "elapsed_seconds": round(elapsed, 2),
        "config_source": config_source,
        "raw_json": data,
    }


def preview_block(title: str, content: str):
    line = "=" * 20
    print(f"{line} {title} {line}")
    print(content)
    print()


def build_preview_text(model: str, sheet: str, row: pd.Series, language: str, prompt_variant: str, system_message: str, user_message: str) -> str:
    parts = [
        f"Model: {model}",
        f"Sheet: {sheet}",
        f"Part: {format_display_value(row.get('Part')) or 'N/A'}",
        f"Item: {format_display_value(row.get('Number'))}",
        f"Language: {language}",
        f"Prompt: {prompt_variant}",
        "Temperature: omitted (API default)",
        "Memory Mode: single-turn / no-memory",
        "",
        "==================== SYSTEM ====================",
        system_message,
        "",
        "==================== USER ====================",
        user_message,
        "",
    ]
    return "\n".join(parts)


def normalize_language(language: str) -> str:
    value = language.strip().lower()
    if value in {"zh", "cn", "chinese", "中文"}:
        return "zh"
    if value in {"en", "english", "英文"}:
        return "en"
    raise ValueError(f"Unsupported language: {language}")


def main():
    parser = argparse.ArgumentParser(
        description="Minimal single-turn, no-memory questionnaire example."
    )
    parser.add_argument("--model", default=pick_default_model())
    parser.add_argument("--sheet", default="HEXACO-PI-R-100")
    parser.add_argument("--number", type=int, default=1)
    parser.add_argument("--part", default=None, help="Required when a sheet reuses item numbers across parts.")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--prompt", default="A")
    parser.add_argument(
        "--save-preview",
        default=None,
        help="Optional path to save the preview text as a UTF-8 file.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Actually call the model. Without this flag, only preview the messages.",
    )
    parser.add_argument(
        "--show-json",
        action="store_true",
        help="When used with --run, also print the raw response JSON.",
    )
    args = parser.parse_args()

    language = normalize_language(args.language)
    prompt_variant = normalize_prompt_variant(args.prompt)

    df = load_questionnaire(args.sheet)
    row = resolve_single_item(df, args.number, args.part)

    protocol_text = load_protocol(language)
    system_message, user_message = build_messages(
        sheet_name=args.sheet,
        row=row,
        language=language,
        prompt_variant=prompt_variant,
        protocol_text=protocol_text,
    )
    preview_text = build_preview_text(
        model=args.model,
        sheet=args.sheet,
        row=row,
        language=language,
        prompt_variant=prompt_variant,
        system_message=system_message,
        user_message=user_message,
    )

    if args.save_preview:
        preview_path = Path(args.save_preview).resolve()
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        preview_path.write_text(preview_text, encoding="utf-8")

    print(f"Model: {args.model}")
    print(f"Sheet: {args.sheet}")
    print(f"Part: {format_display_value(row.get('Part')) or 'N/A'}")
    print(f"Item: {args.number}")
    print(f"Language: {language}")
    print(f"Prompt: {prompt_variant}")
    print("Temperature: omitted (API default)")
    print(f"Memory Mode: single-turn / no-memory")
    print()

    preview_block("SYSTEM", system_message)
    preview_block("USER", user_message)

    if not args.run:
        print("Preview only. Add --run to send this single-turn request.")
        return

    result = call_model(args.model, system_message, user_message)
    preview_block("ANSWER", result["answer"] or "<empty>")
    print(f"Elapsed: {result['elapsed_seconds']}s")
    print(f"Config Source: {result['config_source']}")

    if args.show_json:
        preview_block(
            "RAW JSON",
            json.dumps(result["raw_json"], ensure_ascii=False, indent=2),
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
