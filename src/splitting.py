"""
NYC TLC Yellow Taxi -- Data Splitting
======================================

Splits the cleaned dataset into train and test sets using a TIME-BASED split.

Why time-based and not random?
  A random split would allow the model to see trips from "the future" during
  training, which inflates performance metrics. In the real world, a model is
  always trained on past data and evaluated on future data. 
"""

import pandas as pd
from pathlib import Path


TRAIN_RATIO      = 0.8
TRAIN_SAMPLE_SIZE = 20_000
TEST_SAMPLE_SIZE  =  2_000


def split_train_test(parquet_path, output_dir, train_ratio=TRAIN_RATIO):
    """
    Sort all trips by pickup time, then assign the first train_ratio
    fraction to train and the remainder to test.

    Args:
        parquet_path : str | Path  Path to the cleaned .parquet file.
        output_dir   : str | Path  Directory where train/test files will be saved.
        train_ratio  : float       Fraction of data used for training (default 0.8).

    Returns:
        (train_path, test_path) as strings.
    """
    df = pd.read_parquet(parquet_path)

    # Sort chronologically so the split boundary is a point in time
    df = df.sort_values("tpep_pickup_datetime").reset_index(drop=True)
    split_idx = int(len(df) * train_ratio)
    train_df  = df.iloc[:split_idx]
    test_df   = df.iloc[split_idx:]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.parquet"
    test_path  = output_dir / "test.parquet"

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path,  index=False)

    print(f"  Total rows : {len(df):>10,}")
    print(f"  Train rows : {len(train_df):>10,}  "
          f"({train_df['tpep_pickup_datetime'].min().date()} → "
          f"{train_df['tpep_pickup_datetime'].max().date()})")
    print(f"  Test rows  : {len(test_df):>10,}  "
          f"({test_df['tpep_pickup_datetime'].min().date()} → "
          f"{test_df['tpep_pickup_datetime'].max().date()})")

    return str(train_path), str(test_path)


def subsample_splits(train_path, test_path,
                     train_n=TRAIN_SAMPLE_SIZE, test_n=TEST_SAMPLE_SIZE,
                     random_state=42):
    """
    Draw a random subsample from each split to keep training runtime feasible.

    Why random (not time-based) for subsampling?
      The time-based split already enforces the correct temporal boundary
      between train and test.  Within each split, a random subsample gives
      a representative cross-section of the period without introducing bias.

    Args:
        train_path   : str | Path  Path to the full train .parquet file.
        test_path    : str | Path  Path to the full test  .parquet file.
        train_n      : int         Number of training rows to keep (default 20,000).
        test_n       : int         Number of test rows to keep     (default  2,000).
        random_state : int         Reproducibility seed.

    Returns:
        (train_sample_df, test_sample_df) as DataFrames.
    """
    train_df = pd.read_parquet(train_path)
    test_df  = pd.read_parquet(test_path)

    train_n = min(train_n, len(train_df))
    test_n  = min(test_n,  len(test_df))

    train_sample = train_df.sample(n=train_n, random_state=random_state).reset_index(drop=True)
    test_sample  = test_df.sample( n=test_n,  random_state=random_state).reset_index(drop=True)

    print(f"  Train sample : {len(train_sample):>6,}  (from {len(train_df):,})")
    print(f"  Test sample  : {len(test_sample):>6,}  (from {len(test_df):,})")

    return train_sample, test_sample
