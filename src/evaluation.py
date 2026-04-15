"""
NYC TLC Yellow Taxi -- Model Evaluation & Champion Selection
=============================================================

This module demonstrates two key payoffs of a modular pipeline:

  1. Feature re-use
     The same run_feature_pipeline() call used during training is applied to
     the test set — but this time we pass in the pre-built feature store and
     fitted scaler instead of rebuilding them.  This proves that the feature
     engineering is reproducible and not entangled with the training data.

  2. Model comparison & champion selection
     Every model saved by models.py is loaded, scored against the test set,
     and ranked by MAE.  The best model is named the champion.

  3. Feature importance
     After selecting the champion, plot_feature_importance() visualises which
     engineered features actually drove the model's predictions — giving
     you direct feedback on whether your feature work paid off.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.models import load_all_models, load_model


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred):
    """
    Compute three complementary regression metrics.

    RMSE  – root mean squared error (penalises large errors heavily)
    MAE   – mean absolute error (interpretable: "off by X minutes on average")
    MAPE  – mean absolute percentage error (relative, excludes near-zero trips)

    Returns a dict: {"rmse": float, "mae": float, "mape": float}
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae  = np.mean(np.abs(y_true - y_pred))

    # Exclude trips shorter than 1 minute to avoid division near zero
    mask = y_true >= 1.0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    return {"rmse": round(rmse, 4), "mae": round(mae, 4), "mape": round(mape, 4)}


# ── Evaluation loop ───────────────────────────────────────────────────────────

def evaluate_all_models(X_test, y_test, model_dir):
    """
    Load every saved model, generate predictions on the test set,
    compute metrics, and return a ranked results DataFrame.

    Args:
        X_test    : pd.DataFrame of test features
        y_test    : pd.Series   of ground-truth trip durations
        model_dir : str | Path  directory containing .pkl model files

    Returns:
        pd.DataFrame sorted by MAE (ascending), columns:
            model | rmse | mae | mape
    """
    models  = load_all_models(model_dir)
    records = []

    print(f"\n  {'Model':<25} {'RMSE':>8} {'MAE':>8} {'MAPE':>8}")
    print("  " + "-" * 53)

    for name, model in models.items():
        y_pred  = model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)
        records.append({"model": name, **metrics})
        print(
            f"  {name:<25} {metrics['rmse']:>8.2f} "
            f"{metrics['mae']:>8.2f} {metrics['mape']:>7.1f}%"
        )

    results_df = pd.DataFrame(records).sort_values("mae").reset_index(drop=True)
    return results_df


# ── Champion selection ────────────────────────────────────────────────────────

def select_champion(results_df, metric="mae"):
    """
    Identify the model with the lowest value of `metric`.

    Why MAE as the default?
    Riders and dispatchers think in minutes, not squared minutes.
    MAE translates directly to "this model is wrong by X minutes on average" —
    an intuitive business metric that RMSE cannot claim.

    Args:
        results_df : DataFrame returned by evaluate_all_models()
        metric     : column name to rank by (default "mae")

    Returns:
        champion_name (str)
    """
    best_row = results_df.loc[results_df[metric].idxmin()]
    champion = best_row["model"]
    print(f"\n  Champion model : {champion}")
    print(f"  {metric.upper()}            : {best_row[metric]:.4f}")
    return champion


# ── Feature importance ────────────────────────────────────────────────────────

def plot_feature_importance(model, feature_names, model_name, output_dir=None, top_n=20):
    """
    Visualise which features the champion model relies on most.

    Handles three importance sources:
      - Tree-based models (Random Forest, Gradient Boosting, XGBoost):
            model.feature_importances_
      - Linear models (Linear Regression):
            absolute value of model.coef_  (magnitude of coefficients)

    You can see directly whether your engineered features
    (cyclical encoding, interaction terms, feature store values)
    ranked above the raw columns.

    Args:
        model        : fitted champion model
        feature_names: list of feature column names (X_test.columns.tolist())
        model_name   : string label for the plot title
        output_dir   : if provided, saves the plot as a .png file
        top_n        : show only the top N features (default 20)
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        importance_label = "Feature Importance (impurity-based)"
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
        importance_label = "Absolute Coefficient Value"
    else:
        print(f"  {model_name} does not expose feature importances — skipping plot.")
        return

    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .sort_values("importance", ascending=True)   # ascending for horizontal bar
    )

    fig, ax = plt.subplots(figsize=(10, max(5, len(importance_df) * 0.45)))
    bars = ax.barh(importance_df["feature"], importance_df["importance"], color="steelblue")
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    ax.set_xlabel(importance_label)
    ax.set_title(f"Feature Importance — {model_name}", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        save_path = Path(output_dir) / f"feature_importance_{model_name}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved → {save_path}")

    plt.show()
    return importance_df
