"""Utility to normalize Indonesian product text in Excel files."""
from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path
from typing import Iterable

import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMOJI_RE = re.compile(r"[\U00010000-\U0010ffff]", flags=re.UNICODE)
REPEATED_CHAR_RE = re.compile(r"(.)\1{2,}")
NON_ALPHA_RE = re.compile(r"[^0-9a-zA-Z\s]", flags=re.UNICODE)

SPELLING_MAP = {
    "nggak": "tidak",
    "gak": "tidak",
    "ga": "tidak",
    "ngga": "tidak",
    "gk": "tidak",
    "tdk": "tidak",
    "sy": "saya",
    "dr": "dari",
    "yg": "yang",
    "dlm": "dalam",
    "utk": "untuk",
    "bkn": "bukan",
    "krn": "karena",
    "karen": "karena",
    "sm": "sama",
    "sdh": "sudah",
    "udh": "sudah",
    "dgn": "dengan",
    "tp": "tapi",
    "ttg": "tentang",
    "lg": "lagi",
    "bbrp": "beberapa",
    "jk": "jika",
    "blm": "belum",
}

FACTORY = StemmerFactory()
STEMMER = FACTORY.create_stemmer()


def normalize_spelling(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = REPEATED_CHAR_RE.sub(r"\1\1", text)

    tokens = text.split()
    normalized_tokens = [SPELLING_MAP.get(token, token) for token in tokens]
    return " ".join(normalized_tokens)


def remove_noise(text: str) -> str:
    text = HTML_TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = EMOJI_RE.sub(" ", text)
    text = NON_ALPHA_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def light_stem(text: str) -> str:
    tokens = text.split()
    stemmed_tokens = []
    for token in tokens:
        if len(token) <= 3:
            stemmed_tokens.append(token)
        else:
            stemmed_tokens.append(STEMMER.stem(token))
    return " ".join(stemmed_tokens)


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)

    text = remove_noise(text)
    text = normalize_spelling(text)
    if not text:
        return text
    text = light_stem(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def infer_default_columns(df: pd.DataFrame) -> list[str]:
    candidates = [
        "product_name",
        "nama",
        "product_description",
        "deskripsi",
        "description",
    ]
    found = [col for col in candidates if col in df.columns]
    unique_cols: list[str] = []
    for col in found:
        if col not in unique_cols:
            unique_cols.append(col)
    return unique_cols


def process_excel(input_path: Path, output_path: Path, columns: Iterable[str] | None, sheet_name: str | int | None) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file {input_path} does not exist")

    df = pd.read_excel(input_path, sheet_name=sheet_name)

    if isinstance(df, dict):
        raise ValueError("Multiple sheets detected. Please specify a single sheet to process.")

    target_columns = list(columns) if columns else infer_default_columns(df)
    if not target_columns:
        raise ValueError(
            "No target columns supplied and default columns were not found. "
            "Pass --columns to specify which text columns to clean."
        )

    for column in target_columns:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in the Excel file.")
        cleaned_column_name = f"{column}_clean"
        df[cleaned_column_name] = df[column].map(clean_text)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, index=False)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize Indonesian product text in Excel files by cleaning HTML/emoji and applying "
            "light stemming with Sastrawi."
        )
    )
    parser.add_argument("input", type=Path, help="Path to the input Excel file (e.g., input.xlsx)")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=Path("cleaned_input.xlsx"),
        help="Path to write the cleaned Excel file. Defaults to cleaned_input.xlsx",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        help="Names of text columns to clean. If omitted, the script tries common defaults.",
    )
    parser.add_argument(
        "--sheet",
        help="Optional Excel sheet name or index to process. Defaults to the first sheet.",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    sheet_name = None
    if args.sheet is not None:
        sheet_name = int(args.sheet) if args.sheet.isdigit() else args.sheet
    process_excel(args.input, args.output, args.columns, sheet_name)


if __name__ == "__main__":
    main()
