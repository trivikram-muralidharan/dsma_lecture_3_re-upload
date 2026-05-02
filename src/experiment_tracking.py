"""
NYC TLC — Experiment Tracking
==============================

A thin, reusable wrapper around W&B that keeps all tracking logic out of
pipeline.py.  Every pipeline step that needs to talk to W&B goes through
this module instead of calling wandb directly.

Design goals
------------
- One class (ExperimentTracker) for a single training / evaluation run.
- One standalone function (log_monthly_drift_run) for the per-month drift
  monitoring loop, where each month is its own W&B run.
- Wow-factor helpers: log_table, log_plot, log_artifact, alert — each
  demonstrating a different W&B capability in a few lines of code.
"""

import wandb
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ── Per-run tracker ───────────────────────────────────────────────────────────

class ExperimentTracker:
    """
    Wraps a single W&B run.

    Usage
    -----
        tracker = ExperimentTracker(
            project  = "dsma-lecture3",
            run_name = "tuned-rf-champion",
            tags     = ["tuned", "engineered-features"],
            config   = {"model": "random_forest", "n_estimators": 200},
        )
        tracker.log_summary({"mae": 3.21, "rmse": 5.44})
        tracker.log_table(error_df, "error_analysis")
        tracker.log_plot(fig, "feature_importance")
        url = tracker.finish()
    """

    def __init__(self, project, run_name=None, tags=None, config=None):
        self.run = wandb.init(
            project = project,
            id      = wandb.util.generate_id(),
            name    = run_name,
            tags    = tags or [],
            config  = config or {},
            reinit  = True,
        )

    # ── Metrics ───────────────────────────────────────────────────────────────

    def log_metrics(self, metrics: dict):
        """
        Step-level metrics (training curves, per-epoch loss, etc.).
        Appears as a time-series chart in the W&B run page.
        """
        self.run.log(metrics)

    def log_summary(self, metrics: dict):
        """
        Final aggregate metrics (MAE, RMSE after full evaluation).
        Appears in the run table for easy cross-run comparison.
        """
        self.run.summary.update(metrics)

    # ── Interactive table ─────────────────────────────────────────────────────

    def log_table(self, df: pd.DataFrame, table_name: str):
        """
        Log a DataFrame as a W&B Table.

        The table is fully interactive in the W&B UI — you can filter
        by any column, sort by error magnitude, and slice by categorical
        features without writing any additional code.

        Ideal use: the per-sample error DataFrame from error_analysis.py.
        """
        table = wandb.Table(dataframe=df.reset_index(drop=True))
        self.run.log({table_name: table})

    # ── Plots ─────────────────────────────────────────────────────────────────

    def log_plot(self, fig, name: str):
        """
        Log a matplotlib Figure as a W&B Image.

        Keeps all charts in one place (the W&B run page) rather than
        scattered across an outputs/ folder.
        """
        self.run.log({name: wandb.Image(fig)})
        plt.close(fig)

    def log_image_file(self, image_path, name: str):
        """Log an already-saved image file to the run."""
        self.run.log({name: wandb.Image(str(image_path))})

    # ── Artifacts (model versioning) ──────────────────────────────────────────

    def log_artifact(self, file_path, artifact_name: str,
                     artifact_type: str = "model", metadata: dict = None):
        """
        Version a file (model .pkl, scaler, feature store) as a W&B Artifact.

        Each call creates a new version (v0, v1, v2 …) automatically.
        The lineage graph in W&B shows exactly which run produced each version.

        Ideal use: log every saved model so that you can compare
        model:v1 (baseline) → model:v2 (tuned) → model:v3 (post-mitigation).
        """
        artifact = wandb.Artifact(
            name     = artifact_name,
            type     = artifact_type,
            metadata = metadata or {},
        )
        artifact.add_file(str(file_path))
        self.run.log_artifact(artifact)

    # ── Alerts ────────────────────────────────────────────────────────────────

    def alert(self, title: str, text: str, level: str = "WARN"):
        """
        Fire a W&B alert.

        You will receive an email/Slack notification when this is called.
        Use it when drift is detected or MAE exceeds a threshold — a
        production monitoring moment in two lines of code.

        level options: "INFO", "WARN", "ERROR"
        """
        wandb.alert(title=title, text=text, level=level)

    # ── Code snapshot ─────────────────────────────────────────────────────────

    def log_code(self, root: str = "."):
        """Snapshot the current source files into the run."""
        self.run.log_code(root)

    # ── Finish ────────────────────────────────────────────────────────────────

    def finish(self) -> str:
        """Close the run and return its URL."""
        url = self.run.url
        self.run.finish()
        return url


# ── Per-month drift monitoring run ────────────────────────────────────────────

def log_monthly_drift_run(
    month_label:   str,
    month_num:     int,
    mae:           float,
    drift_report:  pd.DataFrame,
    project:       str,
    mae_delta:     float = None,
    n_trips:       int   = None,
    label_drift:   dict  = None,
):
    """
    Log one evaluation month as its own W&B run.

    Running this for every month in the evaluation set produces a set of
    runs that share the same project.  In W&B, select all of them and plot
    MAE vs. month_num to get the model-degradation curve automatically —
    no extra code required.

    Args:
        month_label  : human-readable label, e.g. "Feb" or "2024-02"
        month_num    : integer month (1–12) — used as the x-axis in comparisons
        mae          : model MAE on this month's data
        drift_report : DataFrame from build_drift_report()
        project      : W&B project name
        mae_delta    : MAE increase vs. reference month (optional)
        n_trips      : number of trips evaluated (optional)
        label_drift  : dict from detect_label_drift() (optional)
    """
    run = wandb.init(
        project = project,
        id      = wandb.util.generate_id(),
        name    = f"drift-eval-{month_label}",
        tags    = ["drift-monitoring", month_label],
        config  = {
            "evaluation_month": month_label,
            "month_num":        month_num,
        },
        reinit  = True,
    )

    run.summary["mae"]       = mae
    run.summary["month_num"] = month_num

    if mae_delta is not None:
        run.summary["mae_delta"] = mae_delta
    if n_trips is not None:
        run.summary["n_trips"] = n_trips

    if label_drift is not None:
        run.summary["label_psi"]        = label_drift["psi"]
        run.summary["label_ks_pvalue"]  = label_drift["ks_pvalue"]
        run.summary["label_drifted"]    = label_drift["drifted"]
        run.summary["label_ref_mean"]   = label_drift["ref_mean"]
        run.summary["label_cur_mean"]   = label_drift["cur_mean"]

    # Log per-feature PSI and KS p-value as individual summary scalars so
    # they appear as comparable columns in the W&B runs table.
    for _, row in drift_report.iterrows():
        feat = row["feature"]
        run.summary[f"psi_{feat}"]      = row["psi"]
        run.summary[f"ks_pvalue_{feat}"] = row["ks_pvalue"]
        run.summary[f"drifted_{feat}"]  = bool(row["drifted"])

    run.summary["n_drifted_features"] = int(drift_report["drifted"].sum())

    # Full drift report as an interactive table
    run.log({"drift_report": wandb.Table(dataframe=drift_report)})

    run.finish()