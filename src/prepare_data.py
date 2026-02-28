import argparse
import pandas as pd
from src.utils import clean_text, normalize_label

def infer_columns(df: pd.DataFrame):
    text_candidates = ["text", "content", "article", "news", "body"]
    label_candidates = ["label", "class", "target", "category", "is_fake"]
    text_col = next((c for c in text_candidates if c in df.columns), None)
    label_col = next((c for c in label_candidates if c in df.columns), None)
    title_col = "title" if "title" in df.columns else None
    return text_col, title_col, label_col

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to raw Kaggle CSV")
    ap.add_argument("--output", required=True, help="Where to write cleaned CSV")
    ap.add_argument("--text-col", default=None, help="Override text column name")
    ap.add_argument("--label-col", default=None, help="Override label column name")
    ap.add_argument("--title-col", default=None, help="Optional title column name")
    ap.add_argument("--dropna", action="store_true", help="Drop rows with missing text/label")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    text_col, title_col, label_col = infer_columns(df)

    if args.text_col: text_col = args.text_col
    if args.label_col: label_col = args.label_col
    if args.title_col: title_col = args.title_col

    if text_col is None or label_col is None:
        raise SystemExit(
            f"Could not infer required columns. Found columns: {list(df.columns)}\n"
            "Provide --text-col and --label-col explicitly."
        )

    if title_col and title_col in df.columns:
        merged = (df[title_col].fillna("").astype(str) + ". " + df[text_col].fillna("").astype(str)).str.strip()
    else:
        merged = df[text_col].fillna("").astype(str)

    out = pd.DataFrame({
        "text": merged.map(clean_text),
        "label_raw": df[label_col]
    })

    out["label"] = out["label_raw"].map(normalize_label)

    unknown = out["label"].isna().mean()
    if unknown > 0.2:
        if pd.api.types.is_numeric_dtype(df[label_col]):
            out["label"] = df[label_col].astype(int)
            if set(out["label"].unique()) - {0, 1}:
                raise SystemExit(
                    "Numeric labels detected but not binary 0/1. "
                    "Please map labels to 0/1 or REAL/FAKE before training."
                )
        else:
            raise SystemExit(
                "Too many labels could not be normalized. "
                "Fix labels to REAL/FAKE or 0/1."
            )

    out = out[["text", "label"]]
    if args.dropna:
        out = out.dropna()

    out = out[out["text"].str.len() > 5].reset_index(drop=True)
    out.to_csv(args.output, index=False)

    print(f"Saved cleaned dataset: {args.output}  (rows={len(out)})")
    print("Label meaning: 1=REAL, 0=FAKE")

if __name__ == "__main__":
    main()
