"""
Data Cleaning & Preprocessing Pipeline
=======================================
Turn raw, messy data into clean ML-ready datasets.

Author  : Your Name
GitHub  : https://github.com/yourusername
License : MIT

Pipeline Steps:
  1. Load CSV data
  2. Inspect & report issues
  3. Handle missing values
  4. Remove duplicates
  5. Encode categorical columns (Label / One-Hot)
  6. Scale numerical columns (Min-Max / Standard)
  7. Export clean dataset
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR  = "cleaned_output"
LOG_FILE    = os.path.join(OUTPUT_DIR, "pipeline_log.txt")

# ── Utilities ─────────────────────────────────────────────────────────────────

def separator(char: str = "─", width: int = 55) -> None:
    print(char * width)

def header(title: str) -> None:
    separator("═")
    print(f"  {title}")
    separator("═")

def log(message: str) -> None:
    """Print and write to log file."""
    print(message)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(message + "\n")

# ── Step 1: Load Data ─────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV into DataFrame with basic validation."""
    header("  Step 1 — Load Data")

    if not os.path.exists(filepath):
        log(f"❌ File not found: {filepath}")
        sys.exit(1)

    df = pd.read_csv(filepath)
    log(f"✅ Loaded: {filepath}")
    log(f"   Shape  : {df.shape[0]} rows × {df.shape[1]} columns")
    log(f"   Columns: {list(df.columns)}")
    return df

# ── Step 2: Inspect & Report ──────────────────────────────────────────────────

def inspect_data(df: pd.DataFrame) -> dict:
    """Analyze and report data quality issues."""
    header("🔍  Step 2 — Data Inspection Report")

    total_cells = df.shape[0] * df.shape[1]

    # Missing values
    missing      = df.isnull().sum()
    missing_pct  = (missing / len(df) * 100).round(2)

    # Duplicates
    dup_count    = df.duplicated().sum()

    # Column types
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    log("\n  📊 Missing Values:")
    separator("-", 45)
    for col in df.columns:
        if missing[col] > 0:
            log(f"  {col:<25} {missing[col]:>5} missing  ({missing_pct[col]}%)")
        else:
            log(f"  {col:<25}     0 missing  (0.0%)")

    log(f"\n  🔁 Duplicate Rows  : {dup_count}")
    log(f"  🔢 Numeric Columns : {num_cols}")
    log(f"  🔤 Category Columns: {cat_cols}")
    log(f"  📦 Total Cells     : {total_cells}")

    return {
        "missing"  : missing,
        "dup_count": dup_count,
        "num_cols" : num_cols,
        "cat_cols" : cat_cols,
    }

# ── Step 3: Handle Missing Values ─────────────────────────────────────────────

def handle_missing(df: pd.DataFrame, strategy: str = "auto") -> pd.DataFrame:
    """
    Fill missing values.
    strategy:
        'auto'   → numeric=median, categorical=mode
        'mean'   → numeric=mean,   categorical=mode
        'drop'   → drop rows with any missing value
    """
    header("🧹  Step 3 — Handle Missing Values")

    before = df.isnull().sum().sum()

    if strategy == "drop":
        df = df.dropna().reset_index(drop=True)
        log(f"  ✅ Dropped rows with missing values.")

    else:
        num_cols = df.select_dtypes(include=[np.number]).columns
        cat_cols = df.select_dtypes(include=["object", "category"]).columns

        for col in num_cols:
            if df[col].isnull().any():
                fill_val = df[col].mean() if strategy == "mean" else df[col].median()
                df[col]  = df[col].fillna(round(fill_val, 4))
                log(f"  ✅ {col:<22} → filled with {strategy} ({round(fill_val,4)})")

        for col in cat_cols:
            if df[col].isnull().any():
                fill_val = df[col].mode()[0]
                df[col]  = df[col].fillna(fill_val)
                log(f"  ✅ {col:<22} → filled with mode ('{fill_val}')")

    after = df.isnull().sum().sum()
    log(f"\n  Missing before : {before}")
    log(f"  Missing after  : {after}")
    return df

# ── Step 4: Remove Duplicates ─────────────────────────────────────────────────

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows."""
    header("🗑️   Step 4 — Remove Duplicates")

    before = len(df)
    df     = df.drop_duplicates().reset_index(drop=True)
    after  = len(df)

    log(f"  Rows before : {before}")
    log(f"  Rows after  : {after}")
    log(f"  Removed     : {before - after} duplicate(s)")
    return df

# ── Step 5: Encode Categorical Columns ───────────────────────────────────────

def encode_categorical(df: pd.DataFrame, method: str = "label") -> pd.DataFrame:
    """
    Encode text/categorical columns.
    method:
        'label'   → LabelEncoder-style (pandas factorize)
        'onehot'  → pd.get_dummies (creates binary columns)
    """
    header("🔢  Step 5 — Encode Categorical Columns")

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if not cat_cols:
        log("  ℹ️  No categorical columns found. Skipping.")
        return df

    if method == "onehot":
        df = pd.get_dummies(df, columns=cat_cols, drop_first=False)
        new_cols = [c for c in df.columns if c not in cat_cols]
        log(f"  ✅ One-Hot Encoded: {cat_cols}")
        log(f"  📌 New columns    : {[c for c in df.columns if '_' in c][:10]}")

    else:  # label encoding
        for col in cat_cols:
            codes, uniques = pd.factorize(df[col])
            df[col] = codes
            log(f"  ✅ Label Encoded: {col:<20} → {list(uniques[:5])} → 0,1,2...")

    return df

# ── Step 6: Scale Numerical Columns ──────────────────────────────────────────

def scale_numerical(df: pd.DataFrame, method: str = "minmax") -> pd.DataFrame:
    """
    Scale numeric columns.
    method:
        'minmax'   → scales to [0, 1]
        'standard' → mean=0, std=1 (z-score)
    """
    header("📐  Step 6 — Scale Numerical Columns")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not num_cols:
        log("  ℹ️  No numeric columns found. Skipping.")
        return df

    if method == "standard":
        for col in num_cols:
            mean = df[col].mean()
            std  = df[col].std()
            if std == 0:
                df[col] = 0.0
            else:
                df[col] = ((df[col] - mean) / std).round(6)
            log(f"  ✅ Standard Scaled: {col:<20} mean={round(mean,2)}, std={round(std,2)}")

    else:  # minmax
        for col in num_cols:
            col_min = df[col].min()
            col_max = df[col].max()
            if col_max == col_min:
                df[col] = 0.0
            else:
                df[col] = ((df[col] - col_min) / (col_max - col_min)).round(6)
            log(f"  ✅ MinMax Scaled  : {col:<20} [{round(col_min,2)} → {round(col_max,2)}] → [0, 1]")

    return df

# ── Step 7: Export Clean Data ─────────────────────────────────────────────────

def export_data(df: pd.DataFrame, original_path: str) -> str:
    """Save cleaned DataFrame to CSV."""
    header("💾  Step 7 — Export Clean Dataset")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name  = os.path.splitext(os.path.basename(original_path))[0]
    out_path   = os.path.join(OUTPUT_DIR, f"{base_name}_cleaned_{timestamp}.csv")

    df.to_csv(out_path, index=False)
    log(f"  ✅ Clean data saved : {out_path}")
    log(f"  📦 Final shape      : {df.shape[0]} rows × {df.shape[1]} columns")
    return out_path

# ── Pipeline Runner ───────────────────────────────────────────────────────────

def run_pipeline(
    filepath        : str,
    missing_strategy: str = "auto",    # auto | mean | drop
    encode_method   : str = "label",   # label | onehot
    scale_method    : str = "minmax",  # minmax | standard
) -> pd.DataFrame:
    """Run the full cleaning pipeline end-to-end."""

    # Clear old log
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    open(LOG_FILE, "w").close()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log(f"🚀 Pipeline started at {timestamp}")
    log(f"   File    : {filepath}")
    log(f"   Missing : {missing_strategy} | Encode: {encode_method} | Scale: {scale_method}\n")

    df = load_data(filepath)
    inspect_data(df)
    df = handle_missing(df, strategy=missing_strategy)
    df = remove_duplicates(df)
    df = encode_categorical(df, method=encode_method)
    df = scale_numerical(df, method=scale_method)
    out = export_data(df, filepath)

    separator("═")
    log(f"\n✅  Pipeline complete!  →  {out}")
    log(f"📋  Log saved          →  {LOG_FILE}")
    separator("═")

    return df

# ── Demo: Generate Sample Raw Data ───────────────────────────────────────────

def generate_sample_data(path: str = "sample_raw.csv") -> None:
    """Create a messy sample CSV for testing the pipeline."""
    np.random.seed(42)
    n = 120

    data = {
        "age"       : np.random.choice([np.nan, *np.random.randint(18, 60, n)], n),
        "salary"    : np.random.choice([np.nan, *np.random.randint(20000, 100000, n)], n),
        "department": np.random.choice(["HR", "Engineering", "Sales", np.nan, "Marketing"], n),
        "education" : np.random.choice(["Bachelor", "Master", "PhD", "Bachelor"], n),
        "score"     : np.random.uniform(0, 100, n).round(2),
        "experience": np.random.randint(0, 20, n),
    }

    df = pd.DataFrame(data)
    # Inject duplicates
    df = pd.concat([df, df.sample(10, random_state=1)], ignore_index=True)
    df.to_csv(path, index=False)
    print(f"📄 Sample raw data created: {path}  ({len(df)} rows)")

# ── CLI Entry Point ───────────────────────────────────────────────────────────

def main():
    header("🧠  Data Cleaning & Preprocessing Pipeline")

    print("""
  Options:
    1. Run pipeline on your own CSV file
    2. Run demo with auto-generated sample data
    0. Exit
""")
    choice = input("  Choose option: ").strip()

    if choice == "0":
        sys.exit(0)

    elif choice == "2":
        sample_path = "sample_raw.csv"
        generate_sample_data(sample_path)
        run_pipeline(
            filepath         = sample_path,
            missing_strategy = "auto",
            encode_method    = "label",
            scale_method     = "minmax",
        )

    elif choice == "1":
        filepath = input("  Enter CSV file path: ").strip()

        print("\n  Missing value strategy:")
        print("    1. auto   (numeric=median, categorical=mode)")
        print("    2. mean   (numeric=mean,   categorical=mode)")
        print("    3. drop   (drop rows with missing values)")
        ms = {"1": "auto", "2": "mean", "3": "drop"}.get(
            input("  Choose [1/2/3]: ").strip(), "auto"
        )

        print("\n  Encoding method:")
        print("    1. label  (0, 1, 2 per category)")
        print("    2. onehot (binary columns)")
        enc = {"1": "label", "2": "onehot"}.get(
            input("  Choose [1/2]: ").strip(), "label"
        )

        print("\n  Scaling method:")
        print("    1. minmax   (range 0 to 1)")
        print("    2. standard (mean=0, std=1)")
        scl = {"1": "minmax", "2": "standard"}.get(
            input("  Choose [1/2]: ").strip(), "minmax"
        )

        run_pipeline(
            filepath         = filepath,
            missing_strategy = ms,
            encode_method    = enc,
            scale_method     = scl,
        )

    else:
        print("❌ Invalid option.")

if __name__ == "__main__":
    main()

