"""Step 1: dataset profiling (missing values, duplicates, invalid formats)."""
import json

import pandas as pd

from src.config import (
    MISSING_REPORT_PATH, QUALITY_SUMMARY_PATH, RAW_PATH, REPORT_DIR,
    SPARSE_COLUMN_THRESHOLD,
)
from src.text_utils import normalize_domain, normalize_email, normalize_phone


def missing_report(df):
    """Per-column missing counts/percentages with a 'sparse' flag."""
    report = pd.DataFrame({
        "Column": df.columns,
        "Missing_Count": df.isnull().sum().values,
        "Missing_Percentage": (df.isnull().mean().values * 100).round(2),
    }).sort_values("Missing_Percentage", ascending=False).reset_index(drop=True)
    report["Sparse_Column"] = report["Missing_Percentage"] >= SPARSE_COLUMN_THRESHOLD
    return report


def invalid_format_counts(df):
    """How many present values fail basic format validation."""
    counts = {}
    checks = {
        "primary_email": normalize_email,
        "website_domain": lambda v: normalize_domain(v) if "." in normalize_domain(v) else "",
        "primary_phone": normalize_phone,
    }
    for column, normaliser in checks.items():
        if column not in df.columns:
            continue
        present = df[column].dropna()
        invalid = present.map(lambda v: normaliser(v) == "")
        counts[column] = {"present": int(len(present)), "invalid": int(invalid.sum())}
    return counts


def profile(df):
    """Return a JSON-serialisable summary of the dataset."""
    missing = missing_report(df)
    summary = {
        "records": int(len(df)),
        "columns": int(len(df.columns)),
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "sparse_columns": missing.loc[missing["Sparse_Column"], "Column"].tolist(),
        "empty_columns": df.columns[df.isnull().all()].tolist(),
        "invalid_formats": invalid_format_counts(df),
    }
    if "company_name" in df.columns:
        names = df["company_name"].dropna().str.lower().str.strip()
        summary["missing_company_names"] = int(df["company_name"].isnull().sum())
        summary["repeated_company_names"] = int(names.duplicated().sum())
    return summary, missing


def main():
    df = pd.read_csv(RAW_PATH, low_memory=False)
    summary, missing = profile(df)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    missing.to_csv(MISSING_REPORT_PATH, index=False)
    QUALITY_SUMMARY_PATH.write_text(json.dumps(summary, indent=2))

    print("\n========== DATATWIN DATA QUALITY REPORT ==========")
    print("Records:", f"{summary['records']:,}", "| Columns:", summary["columns"])
    print("Exact duplicate rows:", summary["exact_duplicate_rows"])
    print("Missing company names:", summary.get("missing_company_names"))
    print("Repeated company names:", summary.get("repeated_company_names"))
    print(f"Sparse columns (>= {SPARSE_COLUMN_THRESHOLD}% missing):",
          len(summary["sparse_columns"]))
    print("Invalid formats:", summary["invalid_formats"])
    print("\nTop missing columns:")
    print(missing.head(10).to_string(index=False))
    print(f"\nSaved: {MISSING_REPORT_PATH}\nSaved: {QUALITY_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
