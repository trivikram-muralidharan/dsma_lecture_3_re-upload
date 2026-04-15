"""
NYC TLC Yellow Taxi -- Data Validation
=======================================

Each  helper function returns a result dict:
  { "name": str, "column": str, "passed": bool, "detail": str }

validate_nyc_taxi_parquet() runs all checks and returns:
  { "success": bool, "results": list[dict] }
"""

import pandas as pd


# ---------------------------------------------------------------------------
#  check helper functions
# ---------------------------------------------------------------------------

def _check_required_columns(df, required):
    missing = [c for c in required if c not in df.columns]
    passed  = len(missing) == 0
    return {
        "name":   "required_columns_present",
        "column": "TABLE",
        "passed": passed,
        "detail": "OK" if passed else f"missing columns: {missing}",
    }


def _check_row_count(df, min_value, max_value):
    n      = len(df)
    passed = min_value <= n <= max_value
    return {
        "name":   f"row_count_between({min_value:,}, {max_value:,})",
        "column": "TABLE",
        "passed": passed,
        "detail": "OK" if passed else f"actual row count: {n:,}",
    }


def _check_not_null(df, col):
    n_null = df[col].isna().sum()
    passed = int(n_null) == 0
    return {
        "name":   "not_null",
        "column": col,
        "passed": passed,
        "detail": "OK" if passed else f"{n_null:,} null values ({n_null/len(df):.2%})",
    }


def _check_dtype_datetime(df, col):
    is_dt = pd.api.types.is_datetime64_any_dtype(df[col])
    return {
        "name":   "dtype_is_datetime",
        "column": col,
        "passed": is_dt,
        "detail": "OK" if is_dt else f"actual dtype: {df[col].dtype}",
    }


def _check_in_set(df, col, value_set, mostly=1.0):
    valid     = df[col].isin(value_set)
    fail_rate = (~valid & df[col].notna()).mean()
    passed    = fail_rate <= (1 - mostly)
    bad_vals  = df.loc[~valid & df[col].notna(), col].value_counts().head(5).to_dict()
    return {
        "name":   f"values_in_set(mostly={mostly})",
        "column": col,
        "passed": passed,
        "detail": "OK" if passed else (
            f"{fail_rate:.2%} outside {value_set}; top bad values: {bad_vals}"
        ),
    }


def _check_between(df, col, min_value=None, max_value=None, mostly=1.0):
    mask = pd.Series(True, index=df.index)
    if min_value is not None:
        mask &= df[col] >= min_value
    if max_value is not None:
        mask &= df[col] <= max_value
    fail_rate = (~mask & df[col].notna()).mean()
    passed    = fail_rate <= (1 - mostly)
    bounds    = f"[{min_value}, {max_value}]"
    return {
        "name":   f"values_between{bounds}(mostly={mostly})",
        "column": col,
        "passed": passed,
        "detail": "OK" if passed else (
            f"{fail_rate:.2%} outside {bounds}; "
            f"min={df[col].min():.2f}, max={df[col].max():.2f}"
        ),
    }


def _check_pair_A_gt_B(df, col_a, col_b, or_equal=False, mostly=1.0):
    valid     = (df[col_a] >= df[col_b]) if or_equal else (df[col_a] > df[col_b])
    both      = df[col_a].notna() & df[col_b].notna()
    fail_rate = (~valid & both).mean()
    passed    = fail_rate <= (1 - mostly)
    op        = ">=" if or_equal else ">"
    return {
        "name":   f"pair_{col_a}_{op}_{col_b}(mostly={mostly})",
        "column": f"{col_a}, {col_b}",
        "passed": passed,
        "detail": "OK" if passed else f"{fail_rate:.2%} rows violate {col_a} {op} {col_b}",
    }


def _check_compound_unique(df, cols):
    n_dupes = df.duplicated(subset=cols).sum()
    passed  = int(n_dupes) == 0
    return {
        "name":   f"compound_unique({', '.join(cols)})",
        "column": ", ".join(cols),
        "passed": passed,
        "detail": "OK" if passed else f"{n_dupes:,} duplicate rows on key columns",
    }


# ---------------------------------------------------------------------------
# Main validation entry point
# ---------------------------------------------------------------------------

def validate_nyc_taxi_parquet(parquet_path):
    """
    Run all validation checks against one month of NYC TLC Yellow Taxi data.

    Args:
        parquet_path: Path (str or Path) to the .parquet file.

    Returns:
        dict:
          "success" (bool)  -- True only if every check passed
          "results" (list)  -- one result dict per check
    """
    df      = pd.read_parquet(parquet_path)
    results = []

    required_columns = [
        "VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime",
        "passenger_count", "trip_distance", "RatecodeID", "store_and_fwd_flag",
        "PULocationID", "DOLocationID", "payment_type",
        "fare_amount", "extra", "mta_tax", "tip_amount",
        "tolls_amount", "improvement_surcharge", "total_amount",
        "congestion_surcharge",
    ]
    results.append(_check_required_columns(df, required_columns))
    results.append(_check_row_count(df, min_value=100_000, max_value=10_000_000))

    results.append(_check_not_null(df, "VendorID"))
    results.append(_check_in_set(df, "VendorID", value_set=[1, 2]))

    for col in ["tpep_pickup_datetime", "tpep_dropoff_datetime"]:
        results.append(_check_not_null(df, col))
        results.append(_check_dtype_datetime(df, col))

    results.append(_check_pair_A_gt_B(
        df, "tpep_dropoff_datetime", "tpep_pickup_datetime", or_equal=False
    ))

    results.append(_check_between(df, "passenger_count", min_value=0, max_value=9))
    results.append(_check_not_null(df, "trip_distance"))
    results.append(_check_between(df, "trip_distance", min_value=0))
    results.append(_check_in_set(df, "RatecodeID", value_set=[1, 2, 3, 4, 5, 6]))
    results.append(_check_in_set(df, "store_and_fwd_flag", value_set=["Y", "N"]))

    for col in ["PULocationID", "DOLocationID"]:
        results.append(_check_not_null(df, col))
        results.append(_check_between(df, col, min_value=1, max_value=265))

    results.append(_check_in_set(df, "payment_type", value_set=[1, 2, 3, 4, 5, 6]))

    for col in ["fare_amount", "tip_amount", "tolls_amount", "total_amount"]:
        results.append(_check_not_null(df, col))
        results.append(_check_between(df, col, min_value=0))

    results.append(_check_between(df, "fare_amount", min_value=3.0, mostly=0.98))
    results.append(_check_in_set(df, "mta_tax", value_set=[0.0, 0.5]))
    results.append(_check_in_set(
        df, "improvement_surcharge", value_set=[0.0, 0.3], mostly=0.99
    ))
    results.append(_check_in_set(
        df, "congestion_surcharge", value_set=[0.0, 2.50, 2.75], mostly=0.99
    ))
    results.append(_check_pair_A_gt_B(
        df, "total_amount", "fare_amount", or_equal=True, mostly=0.99
    ))
    results.append(_check_compound_unique(
        df, cols=["VendorID", "tpep_pickup_datetime",
                  "tpep_dropoff_datetime", "PULocationID"]
    ))

    success = all(r["passed"] for r in results)
    return {"success": success, "results": results}
