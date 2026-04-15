"""
NYC TLC Yellow Taxi -- Data Cleaning
======================================

Applies targeted NaN handling and basic data quality filters to the raw
validated parquet before it enters the feature engineering pipeline.

Design principles
-----------------
- Drop rows only when a column is critical and cannot be reasonably imputed.
- Fill NaNs with sensible domain defaults when the column is non-critical.
- Never silently discard data — every action is logged with a row count.

Null audit on yellow_tripdata_2024-01_clean.parquet
----------------------------------------------------
Column              Null count   Strategy
------------------  ----------   ----------------------------------------
RatecodeID          137,985      Fill → 1 (standard metered rate, the mode)
store_and_fwd_flag  137,985      Fill → "N" (trip was not stored offline)
Airport_fee         137,985      Fill → 0.0 (no airport fee charged)

All three nulls occur on the same 137,985 rows, suggesting a batch of trips
where supplementary fields were not recorded.  Dropping them would discard
~4.7% of the dataset unnecessarily; imputing with domain defaults is safe.
"""

import pandas as pd
from pathlib import Path


# ── Per-column fill strategies ────────────────────────────────────────────────
#
# Each entry maps a column name to the value used to fill its NaNs.
# To change a strategy, update the value here — the pipeline picks it up
# automatically.

NAN_FILL_STRATEGIES = {
    "RatecodeID":         1,      # 1 = standard metered rate (most common)
    "store_and_fwd_flag": "N",    # N = data transmitted in real time (most common)
    "Airport_fee":        0.0,    # no airport fee for non-airport trips
}

# Columns that are critical — rows with nulls here are dropped rather than filled
CRITICAL_COLUMNS = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "trip_distance",
    "PULocationID",
    "DOLocationID",
    "fare_amount",
    "total_amount",
]

# Columns to retain after cleaning — all others are dropped as irrelevant
RELEVANT_COLUMNS = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "PULocationID",
    "DOLocationID",
]


# ── Cleaning functions ────────────────────────────────────────────────────────

def drop_critical_nulls(df):
    """
    Drop any row where a critical column is null.
    Critical columns are those whose absence makes the row unusable
    (e.g., missing pickup time means we cannot compute trip duration).
    """
    before = len(df)
    df = df.dropna(subset=[c for c in CRITICAL_COLUMNS if c in df.columns])
    dropped = before - len(df)
    if dropped:
        print(f"  drop_critical_nulls  : dropped {dropped:,} rows")
    else:
        print(f"  drop_critical_nulls  : no rows dropped")
    return df


def fill_non_critical_nulls(df):
    """
    Fill NaNs in non-critical columns using domain-appropriate defaults
    defined in NAN_FILL_STRATEGIES.
    """
    filled_report = []
    for col, fill_value in NAN_FILL_STRATEGIES.items():
        if col not in df.columns:
            continue
        n_null = df[col].isna().sum()
        if n_null > 0:
            df[col] = df[col].fillna(fill_value)
            filled_report.append(f"{col} ({n_null:,} → {fill_value!r})")

    if filled_report:
        print(f"  fill_non_critical_nulls : filled — {', '.join(filled_report)}")
    else:
        print(f"  fill_non_critical_nulls : no nulls to fill")
    return df


def drop_pre_december_2023(df):
    """
    Drop any trip whose pickup datetime is before December 2023.
    The dataset should only contain trips from December 2023 onwards.
    """
    before = len(df)
    df = df[df["tpep_pickup_datetime"] >= "2023-12-01"]
    dropped = before - len(df)
    if dropped:
        print(f"  drop_pre_december_2023  : dropped {dropped:,} rows")
    else:
        print(f"  drop_pre_december_2023  : no rows dropped")
    return df


def select_relevant_columns(df):
    """
    Retain only the columns required for the ETA prediction problem.
    All other columns are dropped as irrelevant.
    """
    cols = [c for c in RELEVANT_COLUMNS if c in df.columns]
    dropped_cols = [c for c in df.columns if c not in cols]
    if dropped_cols:
        print(f"  select_relevant_columns : dropped {len(dropped_cols)} columns — {dropped_cols}")
    return df[cols]


def drop_remaining_nulls(df):
    """
    Safety net: drop any rows that still contain NaNs after the targeted
    fill step.  In a healthy dataset this should remove zero rows.
    """
    before = len(df)
    df = df.dropna()
    dropped = before - len(df)
    if dropped:
        print(f"  drop_remaining_nulls : dropped {dropped:,} unexpected null rows")
    else:
        print(f"  drop_remaining_nulls : no remaining nulls")
    return df


# ── Main entry point ──────────────────────────────────────────────────────────

def clean_dataframe(df):
    """
    Run the full cleaning sequence on a DataFrame and return the cleaned copy.

    Steps
    -----
    1. Drop rows where critical columns are null.
    2. Fill NaNs in non-critical columns with domain defaults.
    3. Drop any residual nulls (safety net).

    Args:
        df : pd.DataFrame  Raw or validated DataFrame.

    Returns:
        pd.DataFrame  Cleaned DataFrame with no remaining NaNs.
    """
    print(f"  Input  rows : {len(df):,}")
    df = drop_critical_nulls(df)
    df = fill_non_critical_nulls(df)
    df = drop_remaining_nulls(df)
    df = drop_pre_december_2023(df)
    df = select_relevant_columns(df)
    print(f"  Output rows : {len(df):,}")
    return df.reset_index(drop=True)


def clean_parquet(input_path, output_path=None):
    """
    Load a parquet file, clean it, and optionally save the result.

    Args:
        input_path  : str | Path  Path to the input .parquet file.
        output_path : str | Path  If provided, saves the cleaned DataFrame here.

    Returns:
        pd.DataFrame  Cleaned DataFrame.
    """
    df      = pd.read_parquet(input_path)
    df_clean = clean_dataframe(df)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_parquet(output_path, index=False)
        print(f"  Saved cleaned file → {output_path}")

    return df_clean
