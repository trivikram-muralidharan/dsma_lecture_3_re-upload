"""
NYC TLC — Hyperparameter Tuning via W&B Sweeps
===============================================

Two-phase tuning strategy that mirrors the lecture theory:

  Phase 1 — Random Search (model family selection)
    Wide parameter grid, both Random Forest and Gradient Boosting.
    Purpose: quickly identify which model family fits this dataset.
    W&B sweep method: "random"

  Phase 2 — Grid Search (fine-tuning the winner)
    Narrow, exhaustive grid around the best region found in Phase 1.
    Purpose: rigorously squeeze out the last performance from the winner.
    W&B sweep method: "grid"

Adding a new model family
-------------------------
1. Add a MODEL_TYPE entry to RANDOM_SEARCH_CONFIG["parameters"]["model_type"].
2. Add its hyperparameters to RANDOM_SEARCH_CONFIG["parameters"].
3. Add a matching entry to GRID_SEARCH_CONFIGS.
4. Handle it in _build_model_from_config().
"""

import wandb
import joblib
import numpy as np
from pathlib import Path
from sklearn.ensemble         import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection  import cross_val_score
from sklearn.base             import clone

from src.models import save_model, CANDIDATE_MODELS


# ── Phase 1: Random search config ─────────────────────────────────────────────
#
# Sweeps across BOTH model families in a single sweep run.
# learning_rate is ignored when model_type == "random_forest".
# max_features  is ignored when model_type == "gradient_boosting".
# W&B samples parameter combinations at random — fast and effective for
# the "weed out the field" phase.

RANDOM_SEARCH_CONFIG = {
    "method": "random",
    "metric": {"name": "mae", "goal": "minimize"},
    "parameters": {
        "model_type":       {"values": ["random_forest", "gradient_boosting"]},
        "n_estimators":     {"values": [50, 100, 150, 200, 300]},
        "max_depth":        {"values": [5, 10, 15, 20, 30]},
        "min_samples_leaf": {"values": [1, 3, 5, 10, 20]},
        "learning_rate":    {"values": [0.01, 0.05, 0.1, 0.2, 0.3]},
        "max_features":     {"values": ["sqrt", "log2"]},
    },
}


# ── Phase 2: Grid search configs (one per model family) ───────────────────────
#
# Exhaustive search over a tight grid centred on the region the random
# search identified as promising.  Run only the winning family here.
# Grid sizes are kept small (≤ 12 combinations) for classroom runtime.

GRID_SEARCH_CONFIGS = {
    "random_forest": {
        "method": "grid",
        "metric": {"name": "mae", "goal": "minimize"},
        "parameters": {
            "model_type":       {"value": "random_forest"},
            "n_estimators":     {"values": [100, 200]},
            "max_depth":        {"values": [10, 15]},
            "min_samples_leaf": {"values": [3, 5]},
            "max_features":     {"values": ["sqrt"]},
        },
    },
    "gradient_boosting": {
        "method": "grid",
        "metric": {"name": "mae", "goal": "minimize"},
        "parameters": {
            "model_type":       {"value": "gradient_boosting"},
            "n_estimators":     {"values": [100, 200]},
            "max_depth":        {"values": [3, 5]},
            "learning_rate":    {"values": [0.05, 0.1]},
            "min_samples_leaf": {"values": [5]},
        },
    },
}


# ── Model builder ─────────────────────────────────────────────────────────────

def _build_model_from_config(cfg):
    """
    Instantiate an unfitted sklearn model from a W&B run config dict.
    Unused parameters (e.g. learning_rate for RF) are silently ignored.
    """
    model_type = cfg.get("model_type", "random_forest")

    if model_type == "random_forest":
        return RandomForestRegressor(
            n_estimators     = int(cfg.get("n_estimators", 100)),
            max_depth        = cfg.get("max_depth", None),
            min_samples_leaf = int(cfg.get("min_samples_leaf", 5)),
            max_features     = cfg.get("max_features", "sqrt"),
            n_jobs           = -1,
            random_state     = 42,
        )
    elif model_type == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators     = int(cfg.get("n_estimators", 100)),
            max_depth        = int(cfg.get("max_depth", 3)),
            learning_rate    = float(cfg.get("learning_rate", 0.1)),
            min_samples_leaf = int(cfg.get("min_samples_leaf", 5)),
            random_state     = 42,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")


# ── Sweep training closure ────────────────────────────────────────────────────

def _make_train_fn(X_train, y_train):
    """
    Return a zero-argument callable suitable for wandb.agent().

    The closure captures X_train / y_train so the agent can call it
    without arguments.  Each invocation:
      1. Reads hyperparameters from wandb.config
      2. Runs 3-fold CV and computes mean MAE
      3. Logs mae and rmse to W&B
    """
    def train_fn():
        with wandb.init() as run:
            cfg     = run.config
            model   = _build_model_from_config(cfg)

            mae_scores  = cross_val_score(
                model, X_train, y_train,
                cv      = 3,
                scoring = "neg_mean_absolute_error",
                n_jobs  = -1,
            )
            mse_scores  = cross_val_score(
                model, X_train, y_train,
                cv      = 3,
                scoring = "neg_mean_squared_error",
                n_jobs  = -1,
            )
            mae  = float(-mae_scores.mean())
            rmse = float((-mse_scores.mean()) ** 0.5)

            run.log({"mae": mae, "rmse": rmse})

    return train_fn


# ── Public API ────────────────────────────────────────────────────────────────

def run_wandb_sweep(X_train, y_train, sweep_config: dict,
                    project: str, n_runs: int = 15):
    """
    Register a W&B sweep, run `n_runs` trials, and return the best config.

    Args:
        X_train      : training feature DataFrame
        y_train      : training target Series
        sweep_config : RANDOM_SEARCH_CONFIG or GRID_SEARCH_CONFIGS[model]
        project      : W&B project name
        n_runs       : number of trials to run (ignored for grid sweeps,
                       which always run all combinations)

    Returns:
        sweep_id  (str)  : W&B sweep ID — paste into the UI to see results
        best_config (dict): hyperparameter dict of the best trial
        best_mae    (float): CV MAE of the best trial
    """
    sweep_id = wandb.sweep(sweep_config, project=project)
    train_fn = _make_train_fn(X_train, y_train)
    wandb.agent(sweep_id, function=train_fn, count=n_runs)

    api  = wandb.Api()
    runs = api.runs(project, filters={"sweep": sweep_id})
    completed = [r for r in runs if "mae" in r.summary]

    if not completed:
        raise RuntimeError("Sweep produced no results — check W&B connection.")

    best = min(completed, key=lambda r: r.summary["mae"])
    return sweep_id, best.config, best.summary["mae"]


def retrain_best_model(best_config: dict, X_train, y_train,
                       model_dir: str = "models/tuned"):
    """
    Build the winning model from its config, retrain on the full training
    set, and save it to disk.

    This is separate from the sweep because the sweep only runs CV —
    it never trains on the full dataset.  This step does the final fit.

    Returns:
        fitted model object
    """
    model = _build_model_from_config(best_config)
    model_type = best_config.get("model_type", "random_forest")

    print(f"  Retraining best {model_type} on full training set ...")
    model.fit(X_train, y_train)

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    save_path = save_model(model, f"tuned_{model_type}", model_dir)
    print(f"  Tuned model saved → {save_path}")

    return model
