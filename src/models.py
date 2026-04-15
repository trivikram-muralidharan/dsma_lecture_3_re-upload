"""
NYC TLC Yellow Taxi -- Model Creation & Training
=================================================

Defines a registry of candidate regression models for the ETA prediction task.
Each model is trained on the feature DataFrame produced by features.py and
saved to disk so that evaluation.py can load and compare them independently.

Adding a new model
------------------
1. Import the class at the top of this file.
2. Add an entry to CANDIDATE_MODELS with a descriptive string key.
That's it — train_all_models() will pick it up automatically.
"""

import joblib
from pathlib import Path

from sklearn.linear_model    import LinearRegression
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor  # noqa: F401
from sklearn.base            import clone

try:
    from xgboost import XGBRegressor
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False


# ── Candidate model registry ──────────────────────────────────────────────────
#
# This dictionary drives the entire training loop.
# Keys become the model's file name on disk (e.g. "random_forest.pkl").
#
# Hyperparameter note:
#   The values here are sensible starting points.  Lecture 3 will introduce
#   systematic hyperparameter tuning — treat these as the baseline to beat.

CANDIDATE_MODELS = {
    "linear_regression": LinearRegression(),

    "random_forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
    ),

    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        min_samples_leaf=5,
        random_state=42,
    ),
}




# ── Training functions ────────────────────────────────────────────────────────

def get_candidate_models():
    """
    Return fresh (unfitted) copies of every model in the registry.
    Using clone() ensures a clean slate even if this is called multiple times.
    """
    return {name: clone(model) for name, model in CANDIDATE_MODELS.items()}


def train_all_models(X_train, y_train, model_dir):
    """
    Train every candidate model and save each to disk.

    Args:
        X_train   : pd.DataFrame of training features
        y_train   : pd.Series   of training labels (trip_duration_minutes)
        model_dir : str | Path  directory where .pkl files will be written

    Returns:
        dict mapping model name → fitted model object
    """
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    models  = get_candidate_models()
    trained = {}

    for name, model in models.items():
        print(f"  Training {name} ...", end=" ", flush=True)
        model.fit(X_train, y_train)
        save_model(model, name, model_dir)
        trained[name] = model
        print("done")

    return trained


# ── Persistence helpers ───────────────────────────────────────────────────────

def save_model(model, name, model_dir):
    """Serialise a fitted model to <model_dir>/<name>.pkl."""
    path = Path(model_dir) / f"{name}.pkl"
    joblib.dump(model, path)
    return path


def load_model(name, model_dir):
    """Load a previously saved model by name."""
    path = Path(model_dir) / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No saved model found at {path}")
    return joblib.load(path)


def load_all_models(model_dir):
    """
    Load every .pkl file in model_dir and return a dict of
    { model_name: fitted_model }.
    """
    model_dir = Path(model_dir)
    models    = {}
    for pkl_file in sorted(model_dir.glob("*.pkl")):
        models[pkl_file.stem] = joblib.load(pkl_file)
    if not models:
        raise FileNotFoundError(f"No model .pkl files found in {model_dir}")
    return models
