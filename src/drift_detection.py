"""
NYC TLC — Drift Detection
==========================

Detects two types of drift using the monthly 2024 evaluation dataset:

  Data drift 
    P(X) changes — the input feature distributions shift over time.
    Measured per feature using:
      • PSI  (Population Stability Index)  — industry standard
      • KS   (Kolmogorov-Smirnov test)     — statistical significance

  Concept drift
    P(Y|X) changes — same features, different target relationship.
    Measured by applying the January-trained model to each month's data
    and tracking how MAE degrades over time.

PSI interpretation (industry rule of thumb)
-------------------------------------------
  PSI < 0.10  : no significant drift
  PSI 0.1–0.2 : moderate drift — worth monitoring
  PSI > 0.20  : significant drift — investigate / mitigate

Public API
----------
  load_monthly_eval()           Load & minimally clean the multi-month parquet.
  compute_psi()                 PSI for a single feature array.
  build_drift_report()          PSI + KS report DataFrame for a feature set.
  detect_label_drift()          PSI + KS on the target distribution (P(Y) shift).
  detect_concept_drift()        MAE delta between reference and current data.
  plot_feature_distributions()    Side-by-side distribution plots for one feature.
  plot_label_drift_distribution() Layered histogram of P(Y) shift across months.
  plot_monthly_mae_curve()        Line chart of MAE vs. month.
  run_monthly_drift_analysis()  Full loop: every month → drift report + MAE.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

from src.cleaning   import (drop_critical_nulls, fill_non_critical_nulls,
                             drop_remaining_nulls, select_relevant_columns)
from src.features   import run_feature_pipeline, TARGET_COL
from src.evaluation import compute_metrics


# ── Monthly eval loader ───────────────────────────────────────────────────────

def load_monthly_eval(parquet_path):
    """
    Load the multi-month evaluation parquet and apply minimal cleaning.

    Uses the same column-selection and null-handling as cleaning.py but
    deliberately skips the date filter (drop_pre_december_2023) because
    this dataset IS 2024 data.

    Returns:
        pd.DataFrame with the six RELEVANT_COLUMNS, sorted by pickup time.
    """
    df = pd.read_parquet(parquet_path)

    df = drop_critical_nulls(df)
    df = fill_non_critical_nulls(df)
    df = drop_remaining_nulls(df)
    df = select_relevant_columns(df)
    df = df.reset_index(drop=True)

    if not pd.api.types.is_datetime64_any_dtype(df["tpep_pickup_datetime"]):
        df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])

    df = df.sort_values("tpep_pickup_datetime").reset_index(drop=True)

    n_months = df["tpep_pickup_datetime"].dt.month.nunique()
    print(f"  Loaded monthly eval : {len(df):,} rows across {n_months} months "
          f"({df['tpep_pickup_datetime'].dt.strftime('%b %Y').min()} → "
          f"{df['tpep_pickup_datetime'].dt.strftime('%b %Y').max()})")
    return df


# ── PSI ───────────────────────────────────────────────────────────────────────

def compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index for a single feature.

    Bins are derived from the *reference* distribution (percentile-based)
    so the same boundaries apply to both splits.

    Args:
        reference : 1-D array of reference (training) values
        current   : 1-D array of current (evaluation) values
        bins      : number of bins

    Returns:
        PSI score (float)
    """
    breakpoints = np.unique(np.percentile(reference, np.linspace(0, 100, bins + 1)))
    if len(breakpoints) < 3:
        return 0.0  # feature has too few unique values — skip

    eps = 1e-8
    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    cur_counts, _ = np.histogram(current,   bins=breakpoints)

    ref_pct = ref_counts / (len(reference) + eps) + eps
    cur_pct = cur_counts / (len(current)   + eps) + eps

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return round(psi, 6)


# ── Drift report ──────────────────────────────────────────────────────────────

def build_drift_report(reference_df: pd.DataFrame,
                       current_df:   pd.DataFrame,
                       feature_cols: list) -> pd.DataFrame:
    """
    Compute PSI and KS test p-value for each numeric feature.

    Args:
        reference_df : raw reference DataFrame (training or January test)
        current_df   : raw current DataFrame (e.g. one month of evaluation data)
        feature_cols : list of column names to check

    Returns:
        pd.DataFrame sorted by PSI descending, columns:
            feature | psi | ks_statistic | ks_pvalue | drifted

        drifted = True when PSI > 0.2 OR KS p-value < 0.05
    """
    records = []

    for col in feature_cols:
        if col not in reference_df.columns or col not in current_df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(reference_df[col]):
            continue

        ref_vals = reference_df[col].dropna().values
        cur_vals = current_df[col].dropna().values

        if len(ref_vals) == 0 or len(cur_vals) == 0:
            continue

        psi              = compute_psi(ref_vals, cur_vals)
        ks_stat, ks_pval = stats.ks_2samp(ref_vals, cur_vals)

        records.append({
            "feature":      col,
            "psi":          round(psi, 4),
            "ks_statistic": round(float(ks_stat), 4),
            "ks_pvalue":    round(float(ks_pval), 6),
            "drifted":      bool(psi > 0.20 or ks_pval < 0.05),
        })

    return (
        pd.DataFrame(records)
        .sort_values("psi", ascending=False)
        .reset_index(drop=True)
    )


# ── Label drift ───────────────────────────────────────────────────────────────

def detect_label_drift(reference_raw_df: pd.DataFrame,
                       current_raw_df:   pd.DataFrame) -> dict:
    """
    Quantify label drift: has P(Y) shifted between reference and current data?

    Computes trip_duration_minutes from the raw pickup/dropoff datetimes
    (same formula as the feature pipeline) and runs PSI + KS on the resulting
    target distributions.

    Args:
        reference_raw_df : raw reference DataFrame (e.g. January training data)
        current_raw_df   : raw current DataFrame (e.g. one evaluation month)

    Returns:
        dict with keys:
            psi            — Population Stability Index on the target
            ks_statistic   — KS test statistic
            ks_pvalue      — KS test p-value
            drifted        — True when PSI > 0.2 OR ks_pvalue < 0.05
            ref_mean       — mean trip duration in reference data (minutes)
            cur_mean       — mean trip duration in current data (minutes)
    """
    def _compute_target(df):
        delta = df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
        return (delta.dt.total_seconds() / 60).dropna().values

    ref_vals = _compute_target(reference_raw_df)
    cur_vals = _compute_target(current_raw_df)

    psi              = compute_psi(ref_vals, cur_vals)
    ks_stat, ks_pval = stats.ks_2samp(ref_vals, cur_vals)

    return {
        "psi":          round(psi, 6),
        "ks_statistic": round(float(ks_stat), 4),
        "ks_pvalue":    round(float(ks_pval), 6),
        "drifted":      bool(psi > 0.20 or ks_pval < 0.05),
        "ref_mean":     round(float(ref_vals.mean()), 4),
        "cur_mean":     round(float(cur_vals.mean()), 4),
    }


# ── Concept drift ─────────────────────────────────────────────────────────────

def detect_concept_drift(reference_raw_df, current_raw_df,
                         model, scaler) -> dict:
    """
    Quantify concept drift: how much worse does the model perform on
    current data vs. reference data?

    Both DataFrames are passed through the same feature pipeline using
    the *reference* scaler (no re-fitting) so any performance difference
    is purely due to the data distribution shift, not preprocessing
    differences.

    Args:
        reference_raw_df : raw reference DataFrame (e.g. January training data)
        current_raw_df   : raw current DataFrame (e.g. one evaluation month)
        model            : fitted champion model
        scaler           : StandardScaler fitted on reference training data

    Returns:
        dict with keys: reference_mae, current_mae, mae_delta, mae_pct_increase
    """
    ref_features, _ = run_feature_pipeline(
        reference_raw_df, scaler=scaler, is_training=False
    )
    cur_features, _ = run_feature_pipeline(
        current_raw_df, scaler=scaler, is_training=False
    )

    ref_metrics = compute_metrics(
        ref_features[TARGET_COL],
        model.predict(ref_features.drop(columns=[TARGET_COL])),
    )
    cur_metrics = compute_metrics(
        cur_features[TARGET_COL],
        model.predict(cur_features.drop(columns=[TARGET_COL])),
    )

    delta   = cur_metrics["mae"] - ref_metrics["mae"]
    pct_inc = delta / ref_metrics["mae"] * 100 if ref_metrics["mae"] > 0 else 0.0

    return {
        "reference_mae":    ref_metrics["mae"],
        "current_mae":      cur_metrics["mae"],
        "mae_delta":        round(delta, 4),
        "mae_pct_increase": round(pct_inc, 2),
    }


# ── Distribution plots ────────────────────────────────────────────────────────

def plot_feature_distributions(reference_df, current_df, feature_col,
                                ref_label="Reference (Jan)",
                                cur_label="Current",
                                output_dir=None):
    """
    Side-by-side KDE overlay + histogram for one feature.

    The left panel overlays both distributions for direct shape comparison.
    The right panel shows histograms to make count differences visible.

    Returns:
        matplotlib Figure
    """
    ref_vals = reference_df[feature_col].dropna()
    cur_vals = current_df[feature_col].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # KDE overlay
    ax = axes[0]
    ref_vals.plot.kde(ax=ax, label=ref_label, color="steelblue", linewidth=2)
    cur_vals.plot.kde(ax=ax, label=cur_label, color="tomato",    linewidth=2)
    ax.set_title(f"{feature_col} — Distribution Overlay")
    ax.set_xlabel(feature_col)
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)

    # Histogram
    ax2 = axes[1]
    ax2.hist(ref_vals, bins=40, alpha=0.55, label=ref_label,
             color="steelblue", density=True)
    ax2.hist(cur_vals, bins=40, alpha=0.55, label=cur_label,
             color="tomato",    density=True)
    ax2.set_title(f"{feature_col} — Histogram (density)")
    ax2.set_xlabel(feature_col)
    ax2.legend()
    ax2.spines[["top", "right"]].set_visible(False)

    plt.suptitle(f"Data Drift: {feature_col}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        path = Path(output_dir) / f"drift_dist_{feature_col}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved → {path}")

    return fig


def plot_label_drift_distribution(reference_raw_df, current_raw_df,
                                   ref_label="Reference (Jan)",
                                   cur_label="Current",
                                   clip_percentile=99,
                                   output_dir=None):
    """
    Layered histogram + KDE overlay showing how P(trip_duration_minutes)
    has shifted between the reference and a future month.

    The target is derived from raw datetimes so no feature pipeline is needed.
    Values beyond `clip_percentile` of the reference distribution are clipped
    to keep the x-axis readable despite outlier trips.

    Args:
        reference_raw_df : raw reference DataFrame (e.g. January training data)
        current_raw_df   : raw current DataFrame (e.g. one evaluation month)
        ref_label        : legend label for reference distribution
        cur_label        : legend label for current distribution
        clip_percentile  : upper percentile used to clip both series (default 99)
        output_dir       : directory for saving the plot (None = show only)

    Returns:
        matplotlib Figure
    """
    def _target(df):
        delta = df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
        return (delta.dt.total_seconds() / 60).dropna()

    ref_vals = _target(reference_raw_df)
    cur_vals = _target(current_raw_df)

    clip_max = float(np.percentile(ref_vals, clip_percentile))
    ref_vals = ref_vals.clip(upper=clip_max)
    cur_vals = cur_vals.clip(upper=clip_max)

    ref_mean = ref_vals.mean()
    cur_mean = cur_vals.mean()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # ── Left: KDE overlay ─────────────────────────────────────────────────────
    ax = axes[0]
    ref_vals.plot.kde(ax=ax, label=f"{ref_label}  (μ={ref_mean:.1f} min)",
                      color="steelblue", linewidth=2)
    cur_vals.plot.kde(ax=ax, label=f"{cur_label}  (μ={cur_mean:.1f} min)",
                      color="tomato", linewidth=2)
    ax.axvline(ref_mean, color="steelblue", linestyle="--", linewidth=1, alpha=0.7)
    ax.axvline(cur_mean, color="tomato",    linestyle="--", linewidth=1, alpha=0.7)
    ax.set_title("trip_duration_minutes — KDE Overlay")
    ax.set_xlabel("Trip duration (minutes)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    # ── Right: layered histogram ───────────────────────────────────────────────
    ax2 = axes[1]
    bins = np.linspace(0, clip_max, 60)
    ax2.hist(ref_vals, bins=bins, alpha=0.55, label=ref_label,
             color="steelblue", density=True)
    ax2.hist(cur_vals, bins=bins, alpha=0.55, label=cur_label,
             color="tomato",    density=True)
    ax2.axvline(ref_mean, color="steelblue", linestyle="--", linewidth=1, alpha=0.7)
    ax2.axvline(cur_mean, color="tomato",    linestyle="--", linewidth=1, alpha=0.7)
    ax2.set_title("trip_duration_minutes — Layered Histogram (density)")
    ax2.set_xlabel("Trip duration (minutes)")
    ax2.legend(fontsize=9)
    ax2.spines[["top", "right"]].set_visible(False)

    delta_mean = cur_mean - ref_mean
    plt.suptitle(
        f"Label Drift: trip_duration_minutes  "
        f"(Δμ = {delta_mean:+.1f} min,  {ref_label} vs {cur_label})",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        path = Path(output_dir) / "label_drift_trip_duration.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved → {path}")

    return fig


def plot_monthly_mae_curve(monthly_summary: pd.DataFrame, output_dir=None):
    """
    Line chart: MAE vs. evaluation month.

    A dashed reference line marks the January baseline MAE.
    Each point is annotated with the count of drifted features for that month.

    This is the central concept-drift visualisation for the lecture.

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    months = monthly_summary["month_num"].values
    maes   = monthly_summary["mae"].values

    # Baseline reference line (January)
    jan_rows = monthly_summary[monthly_summary["month_num"] == 1]
    if len(jan_rows):
        jan_mae = jan_rows["mae"].iloc[0]
        ax.axhline(
            jan_mae, color="steelblue", linestyle="--", linewidth=1.5, alpha=0.7,
            label=f"Jan baseline  MAE = {jan_mae:.2f} min",
        )

    # MAE curve
    ax.plot(months, maes, "o-", color="tomato", linewidth=2.5,
            markersize=8, label="Monthly eval MAE")

    # Annotate with n_drifted_features
    if "n_drifted_features" in monthly_summary.columns:
        for _, row in monthly_summary.iterrows():
            ax.annotate(
                f"{int(row['n_drifted_features'])} drifted",
                xy     = (row["month_num"], row["mae"]),
                xytext = (0, 12),
                textcoords = "offset points",
                ha="center", fontsize=8, color="dimgray",
            )

    ax.set_xticks(months)
    ax.set_xticklabels(monthly_summary["month"].values, rotation=45)
    ax.set_xlabel("Evaluation Month")
    ax.set_ylabel("MAE (minutes)")
    ax.set_title(
        "Concept Drift — Model Performance Degradation Across 2024",
        fontsize=13, fontweight="bold",
    )
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        path = Path(output_dir) / "monthly_mae_curve.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved → {path}")

    return fig


# ── Full monthly analysis loop ────────────────────────────────────────────────

# Raw feature columns to check for data drift (pre-pipeline columns)
_DRIFT_FEATURE_COLS = [
    "trip_distance",
    "passenger_count",
    "PULocationID",
    "DOLocationID",
]


def run_monthly_drift_analysis(monthly_eval_df, reference_raw_df,
                                model, scaler, output_dir=None):
    """
    Run data drift, label drift, and concept drift analysis for every month
    present in the monthly evaluation dataset.

    Args:
        monthly_eval_df  : full evaluation DataFrame from load_monthly_eval()
        reference_raw_df : raw reference DataFrame (January training data)
        model            : January-trained champion model
        scaler           : StandardScaler fitted on January training data
        output_dir       : directory for saving plots (None = show only)

    Returns:
        monthly_summary  : pd.DataFrame — one row per month with MAE + drift stats
                           (includes label_psi, label_ks_pvalue, label_drifted)
        drift_reports    : dict mapping month_label → drift report DataFrame
    """
    monthly_eval_df = monthly_eval_df.copy()
    monthly_eval_df["_month"]       = monthly_eval_df["tpep_pickup_datetime"].dt.month
    monthly_eval_df["_month_label"] = monthly_eval_df["tpep_pickup_datetime"].dt.strftime("%b")

    months = sorted(monthly_eval_df["_month"].unique())

    print(f"\n  {'Month':<8} {'MAE':>8} {'Δ MAE':>8} {'Δ %':>7}  {'Drifted':>8}")
    print("  " + "-" * 46)

    records      = []
    drift_reports = {}

    for month_num in months:
        mask        = monthly_eval_df["_month"] == month_num
        month_label = monthly_eval_df.loc[mask, "_month_label"].iloc[0]
        month_data  = (
            monthly_eval_df[mask]
            .drop(columns=["_month", "_month_label"])
            .reset_index(drop=True)
        )

        # ── Concept drift ─────────────────────────────────────────────────────
        try:
            concept = detect_concept_drift(
                reference_raw_df, month_data, model, scaler
            )
        except Exception as exc:
            print(f"  {month_label:<8}  concept drift failed: {exc}")
            continue

        # ── Data drift ────────────────────────────────────────────────────────
        drift_report    = build_drift_report(reference_raw_df, month_data,
                                             _DRIFT_FEATURE_COLS)
        n_drifted        = int(drift_report["drifted"].sum())
        drift_reports[month_label] = drift_report

        # ── Label drift ───────────────────────────────────────────────────────
        label_drift = detect_label_drift(reference_raw_df, month_data)

        print(
            f"  {month_label:<8} {concept['current_mae']:>8.2f} "
            f"{concept['mae_delta']:>+8.2f} {concept['mae_pct_increase']:>6.1f}%"
            f"  {n_drifted:>8}"
        )

        records.append({
            "month":              month_label,
            "month_num":          month_num,
            "mae":                concept["current_mae"],
            "reference_mae":      concept["reference_mae"],
            "mae_delta":          concept["mae_delta"],
            "mae_pct_increase":   concept["mae_pct_increase"],
            "n_drifted_features": n_drifted,
            "n_trips":            len(month_data),
            "label_psi":          label_drift["psi"],
            "label_ks_pvalue":    label_drift["ks_pvalue"],
            "label_drifted":      label_drift["drifted"],
            "label_ref_mean":     label_drift["ref_mean"],
            "label_cur_mean":     label_drift["cur_mean"],
        })

    monthly_summary = pd.DataFrame(records)

    # ── Plot feature distributions for drifted features (worst month) ─────────
    if len(monthly_summary) and output_dir:
        worst_month_label = monthly_summary.loc[
            monthly_summary["mae_delta"].idxmax(), "month"
        ]
        worst_report = drift_reports.get(worst_month_label, pd.DataFrame())
        drifted_feats = worst_report.loc[
            worst_report["drifted"], "feature"
        ].tolist() if len(worst_report) else []

        worst_mask  = monthly_eval_df["_month_label"] == worst_month_label
        worst_data  = (
            monthly_eval_df[worst_mask]
            .drop(columns=["_month", "_month_label"])
            .reset_index(drop=True)
        )
        for feat in drifted_feats[:2]:   # plot top 2 drifted features
            plot_feature_distributions(
                reference_raw_df, worst_data, feat,
                cur_label  = f"{worst_month_label} eval",
                output_dir = output_dir,
            )

    return monthly_summary, drift_reports
