"""
explore_data.py
---------------
Quick exploratory analysis of all warranty claim CSV files.

Run from the project root:
    python explore_data.py

Prints per-file summaries (shape, date ranges, part names) plus a combined
overview with claim counts and data-quality stats.
"""

import sys
import os

import numpy as np
import pandas as pd

# Use the canonical DATA_FILES list from config so paths are always correct.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forecasting.config import DATA_FILES

DATE_COLS = ["FCOK_DATE", "REGD_DATE", "REPAIR_DATE", "PROCESSING_DATE"]


# ---------------------------------------------------------------------------
# Per-file summary
# ---------------------------------------------------------------------------

print("=" * 68)
print("  WARRANTY CLAIMS — DATA EXPLORATION")
print("=" * 68)

all_dfs = []
for f in DATA_FILES:
    if not os.path.exists(f):
        print(f"\n[WARN] File not found: {f}")
        continue

    df = pd.read_csv(f)
    for col in DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")

    print(f"\n=== {os.path.basename(f)} ===")
    print(f"  Shape          : {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"  Columns        : {list(df.columns)}")

    if "PROCESSING_DATE" in df.columns:
        print(f"  PROCESSING_DATE: {df['PROCESSING_DATE'].min().date()} "
              f"-> {df['PROCESSING_DATE'].max().date()}")
    if "REPAIR_DATE" in df.columns:
        print(f"  REPAIR_DATE    : {df['REPAIR_DATE'].min().date()} "
              f"-> {df['REPAIR_DATE'].max().date()}")

    if "Part Name" in df.columns:
        n_parts  = df["Part Name"].nunique()
        parts    = sorted(df["Part Name"].dropna().unique())
        print(f"  Unique parts   : {n_parts}  -> {parts}")

    for col in ["Model", "Fuel Type", "PLANT_CODE"]:
        if col in df.columns:
            print(f"  {col:15s}: {sorted(df[col].dropna().unique())}")

    # Missing values
    miss = df.isnull().sum()
    miss = miss[miss > 0]
    if len(miss):
        print(f"  Missing values : {dict(miss)}")
    else:
        print("  Missing values : none")

    all_dfs.append(df)


# ---------------------------------------------------------------------------
# Combined summary
# ---------------------------------------------------------------------------

if not all_dfs:
    print("\n[ERROR] No data files found. Check forecasting/config.py DATA_FILES.")
    sys.exit(1)

combined = pd.concat(all_dfs, ignore_index=True)
for col in DATE_COLS:
    if col in combined.columns:
        combined[col] = pd.to_datetime(combined[col], dayfirst=True, errors="coerce")

print("\n" + "=" * 68)
print("  COMBINED SUMMARY")
print("=" * 68)
print(f"  Total records  : {len(combined):,}")
print(f"  Total parts    : {combined['Part Name'].nunique()}")
print(f"  All part names : {sorted(combined['Part Name'].dropna().unique())}")

if "PROCESSING_DATE" in combined.columns:
    print(f"  Date range     : {combined['PROCESSING_DATE'].min().date()} "
          f"-> {combined['PROCESSING_DATE'].max().date()}")

# Claims per part
print("\n  Claims per part (total across all files):")
part_counts = (
    combined.groupby("Part Name")["Part Name"]
    .count()
    .sort_values(ascending=False)
    .rename("count")
)
for part, count in part_counts.items():
    bar = "#" * int(count / part_counts.max() * 30)
    print(f"    {part:10s}  {count:6,}  {bar}")

# Odometer stats
if "ODOMETER" in combined.columns:
    od = combined["ODOMETER"].dropna()
    print(f"\n  Odometer (km)  : "
          f"min={od.min():.0f}  median={od.median():.0f}  "
          f"max={od.max():.0f}  missing={combined['ODOMETER'].isna().sum():,}")

print("\n" + "=" * 68)
