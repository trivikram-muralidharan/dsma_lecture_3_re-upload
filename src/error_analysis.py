"""
NYC TLC — Per-Sample Error Analysis
=====================================

Breaks down model errors by meaningful data segments so you can see
*where* the model struggles before asking *why* it struggles over time
(which leads naturally into drift).

Three functions the pipeline calls in order:

  build_error_df()         Build a per-row DataFrame with actual, predicted,
                           abs_error, pct_error, plus segmentation columns
                           derived from the feature matrix.

  plot_error_by_segment()  Horizontal bar chart: mean MAE per group value
                           for a single categorical column.

  run_error_analysis()     Orchestrator: calls the two above for every
                           pre-defined segment and returns the error DataFrame
                           for downstream W&B logging.

Segment columns built automatically
-------------------------------------
  is_rush_hour     → "Rush Hour" vs "Off-Peak"
  day_of_week      → Mon … Sun
  is_airport_trip  → "Airport" vs "Non-Airport"  (PULocationID ∈ {1, 132, 138})
  distance_bucket  → quartile-based bins labelled Q1 (short) … Q4 (long)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# JFK = 132, LGA = 138, EWR = 1
AIRPORT_ZONES = {1, 132, 138}

DAY_NAMES = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}


# ── Error DataFrame ───────────────────────────────────────────────────────────

def build_error_df(y_test, y_pred, feature_df=None):
    """
    Build a per-sample error DataFrame.

    Args:
        y_test      : array-like of ground-truth trip durations (minutes)
        y_pred      : array-like of model predictions
        feature_df  : optional — the processed feature matrix (X_test).
                      When provided, segmentation columns are added so the
                      DataFrame can be sliced and filtered in W&B Tables.

    Returns:
        pd.DataFrame with columns:
            actual | predicted | abs_error | pct_error
            [+ segment columns when feature_df is supplied]

    Notes on pct_error:
        Trips shorter than 1 minute are excluded from percentage error to
        avoid division-near-zero artefacts (same guard used in evaluation.py).
    """
    y_test = np.asarray(y_test, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    error_df = pd.DataFrame({
        "actual":    y_test,
        "predicted": y_pred,
    })
    error_df["abs_error"] = np.abs(error_df["actual"] - error_df["predicted"])

    # Percentage error — guard against near-zero denominators
    mask = error_df["actual"] >= 1.0
    error_df["pct_error"] = np.nan
    error_df.loc[mask, "pct_error"] = (
        error_df.loc[mask, "abs_error"] / error_df.loc[mask, "actual"] * 100
    )

    if feature_df is None:
        return error_df

    feat = feature_df.reset_index(drop=True)

    # ── Rush hour ─────────────────────────────────────────────────────────────
    if "is_rush_hour" in feat.columns:
        error_df["rush_hour_label"] = (
            feat["is_rush_hour"]
            .map({1: "Rush Hour", 0: "Off-Peak"})
            .fillna("Unknown")
        )

    # ── Day of week ───────────────────────────────────────────────────────────
    if "day_of_week" in feat.columns:
        error_df["day_name"] = feat["day_of_week"].map(DAY_NAMES).fillna("Unknown")

    # ── Airport vs non-airport ────────────────────────────────────────────────
    if "PULocationID" in feat.columns:
        error_df["trip_type"] = (
            feat["PULocationID"]
            .isin(AIRPORT_ZONES)
            .map({True: "Airport", False: "Non-Airport"})
        )

    # ── Distance bucket (quartile-based — works on scaled values too) ─────────
    if "trip_distance" in feat.columns:
        try:
            error_df["distance_bucket"] = pd.qcut(
                feat["trip_distance"],
                q      = 4,
                labels = ["Q1 (short)", "Q2", "Q3", "Q4 (long)"],
                duplicates = "drop",
            )
        except Exception:
            pass  # too few unique values to bin — skip silently

    # ── Time-of-day bucket (already engineered) ───────────────────────────────
    if "time_of_day_bucket" in feat.columns:
        error_df["time_of_day"] = feat["time_of_day_bucket"].map(
            {0: "Overnight", 1: "Off-Peak", 2: "Rush Hour"}
        ).fillna("Unknown")

    return error_df


# ── Segment plot ──────────────────────────────────────────────────────────────

def plot_error_by_segment(error_df, group_col, title=None, output_dir=None):
    """
    Horizontal bar chart: mean absolute error per value of `group_col`.

    Each bar also shows the sample count so that you can distinguish
    "high error" from "high error but only 3 trips".

    Args:
        error_df   : DataFrame from build_error_df()
        group_col  : column name to group by (e.g. "rush_hour_label")
        title      : plot title; defaults to "MAE by <group_col>"
        output_dir : if provided, saves plot as <group_col>.png

    Returns:
        matplotlib Figure
    """
    if group_col not in error_df.columns:
        print(f"  Skipping segment plot — '{group_col}' not in error_df.")
        return None

    grouped = (
        error_df.groupby(group_col, observed=True)["abs_error"]
        .agg(mean_mae="mean", n_trips="count")
        .sort_values("mean_mae", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(9, max(3, len(grouped) * 0.65)))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(grouped)))
    bars   = ax.barh(grouped.index.astype(str), grouped["mean_mae"], color=colors)

    # Value labels on bars
    ax.bar_label(bars, fmt="%.2f min", padding=5, fontsize=9)

    # Trip count inside each bar (white text, left-aligned)
    for i, (idx, row) in enumerate(grouped.iterrows()):
        ax.text(
            grouped["mean_mae"].min() * 0.05, i,
            f"n={row['n_trips']:,}",
            va="center", ha="left", fontsize=8,
            color="white", fontweight="bold",
        )

    ax.set_xlabel("Mean Absolute Error (minutes)")
    ax.set_title(title or f"MAE by {group_col}", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        path = Path(output_dir) / f"error_by_{group_col}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved → {path}")

    return fig


# ── Layered histogram ────────────────────────────────────────────────────────

def plot_error_histogram_by_segment(error_df, group_col, title=None, output_dir=None, bins=40):
    """
    Layered (overlapping) histogram of abs_error, one semi-transparent layer
    per group value in `group_col`.

    Args:
        error_df   : DataFrame from build_error_df()
        group_col  : column name to split by (e.g. "rush_hour_label")
        title      : plot title; defaults to "Error Distribution by <group_col>"
        output_dir : if provided, saves plot as <group_col>_hist.png
        bins       : number of histogram bins (default 40)

    Returns:
        matplotlib Figure
    """
    if group_col not in error_df.columns:
        print(f"  Skipping histogram — '{group_col}' not in error_df.")
        return None

    groups = error_df[group_col].dropna().unique()

    # Clip to the visible x range before computing bin edges so all 40 bins
    # are spread across 0–100 min rather than the full outlier-extended range.
    all_vals = error_df["abs_error"].dropna().clip(upper=100)
    bin_edges = np.histogram_bin_edges(all_vals, bins=bins)

    fig, ax = plt.subplots(figsize=(9, 4))
    colors = plt.cm.tab10.colors

    for i, grp in enumerate(sorted(groups, key=str)):
        vals = error_df.loc[error_df[group_col] == grp, "abs_error"].dropna().clip(upper=100)
        ax.hist(
            vals,
            bins    = bin_edges,
            alpha   = 0.45,
            label   = f"{grp} (n={len(vals):,})",
            color   = colors[i % len(colors)],
            edgecolor = "none",
            density = True,
        )

    ax.set_xlim(0, 100)
    ax.set_xlabel("Absolute Error (minutes)")
    ax.set_ylabel("Density")
    ax.set_title(title or f"Error Distribution by {group_col}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        path = Path(output_dir) / f"error_hist_{group_col}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved → {path}")

    return fig


# ── Orchestrator ──────────────────────────────────────────────────────────────

# Segments to analyse: (column_in_error_df, human-readable title)
_SEGMENTS = [
    ("rush_hour_label",  "Rush Hour vs Off-Peak"),
    ("time_of_day",      "Time of Day Bucket"),
    ("distance_bucket",  "Trip Distance Quartile"),
    ("trip_type",        "Airport vs Non-Airport"),
    ("day_name",         "Day of Week"),
]


def run_error_analysis(X_test, y_test, model, output_dir=None):
    """
    Full error analysis pipeline.

    Steps
    -----
    1. Generate predictions from the model.
    2. Build the per-sample error DataFrame (with segment columns).
    3. Print a summary statistics table.
    4. For each pre-defined segment, print group-level MAE and save a plot.

    Args:
        X_test     : processed feature DataFrame (from run_feature_pipeline)
        y_test     : ground-truth target Series
        model      : fitted champion model
        output_dir : directory for saving plots; None = show only

    Returns:
        error_df (pd.DataFrame) — use this for W&B Table logging
        figs     (dict)         — {segment_col: matplotlib Figure}
    """
    y_pred   = model.predict(X_test)
    error_df = build_error_df(y_test, y_pred, feature_df=X_test)

    # ── Summary statistics ────────────────────────────────────────────────────
    print(f"\n  {'Metric':<22} {'Value':>10}")
    print("  " + "-" * 35)
    print(f"  {'Mean Abs Error':<22} {error_df['abs_error'].mean():>9.2f} min")
    print(f"  {'Median Abs Error':<22} {error_df['abs_error'].median():>9.2f} min")
    print(f"  {'90th pct Error':<22} {error_df['abs_error'].quantile(0.9):>9.2f} min")
    print(f"  {'Max Error':<22} {error_df['abs_error'].max():>9.2f} min")
    valid_pct = error_df["pct_error"].dropna()
    if len(valid_pct):
        print(f"  {'Mean Pct Error':<22} {valid_pct.mean():>9.1f} %")

    # ── Per-segment breakdown ─────────────────────────────────────────────────
    figs = {}
    for col, title in _SEGMENTS:
        if col not in error_df.columns:
            continue
        if error_df[col].isna().all():
            continue

        print(f"\n  Error by {title}:")
        grouped = (
            error_df.groupby(col, observed=True)["abs_error"]
            .mean()
            .sort_values(ascending=False)
        )
        for group, mae in grouped.items():
            print(f"    {str(group):<25}  MAE = {mae:.2f} min")

        fig = plot_error_by_segment(
            error_df, col, title=f"MAE by {title}", output_dir=output_dir
        )
        if fig is not None:
            figs[col] = fig

        hist_fig = plot_error_histogram_by_segment(
            error_df, col, title=f"Error Distribution by {title}", output_dir=output_dir
        )
        if hist_fig is not None:
            figs[f"{col}_hist"] = hist_fig

    return error_df, figs