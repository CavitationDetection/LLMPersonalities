"""Print a result workbook while preserving literal N/A cell values.

Pandas normally treats N/A as missing; this reader disables that conversion.
"""

import argparse
from pathlib import Path

import pandas as pd


def read_workbook(workbook_path: Path):
    if not workbook_path.exists():
        raise FileNotFoundError(f"Workbook not found: {workbook_path}")

    excel = pd.ExcelFile(workbook_path)
    return {
        sheet_name: pd.read_excel(
            workbook_path,
            sheet_name=sheet_name,
            keep_default_na=False,
        )
        for sheet_name in excel.sheet_names
    }


def main():
    parser = argparse.ArgumentParser(
        description="Read a results workbook without converting literal N/A answers into NaN."
    )
    parser.add_argument("workbook")
    parser.add_argument("--sheet", default=None)
    parser.add_argument("--head", type=int, default=5)
    args = parser.parse_args()

    workbook_path = Path(args.workbook).resolve()
    sheets = read_workbook(workbook_path)

    if args.sheet:
        if args.sheet not in sheets:
            raise RuntimeError(f"Sheet not found: {args.sheet}")
        print(sheets[args.sheet].head(args.head).to_string(index=False))
        return

    print(f"Workbook: {workbook_path}")
    print("Sheets:")
    for sheet_name, df in sheets.items():
        print(f"- {sheet_name}: {df.shape[0]} rows x {df.shape[1]} columns")


if __name__ == "__main__":
    main()
