import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd  # noqa: F401
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


@dataclass
class RunInfo:
    started_at_unix: float
    python: str
    platform: str
    numpy: str
    sklearn: str
    mlflow: str
    git_commit: str | None
    tracking_uri: str
    experiment_name: str
    seed: int
    n_estimators: int
    max_depth: int | None
    test_size: float


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def try_cmd(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def configure_mlflow() -> tuple[str, str]:
    """
    Enforce remote HTTP tracking. Prevents silent fallback to local file stores
    like file:/mlflow or ./mlruns inside the container.
    """
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise RuntimeError("MLFLOW_TRACKING_URI is not set")

    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "ml-docker-lab")

    # Important: set tracking URI BEFORE any other MLflow call.
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    active = mlflow.get_tracking_uri()
    if not active.startswith("http"):
        raise RuntimeError(f"MLflow tracking did not resolve to HTTP. active={active}")

    return tracking_uri, experiment_name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts-dir", default="/artifacts")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--max-depth", type=int, default=20)
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    # MLflow must be configured before any start_run/logging calls.
    tracking_uri, experiment_name = configure_mlflow()

    # Filesystem artifacts (volume)
    artifacts_dir = Path(args.artifacts_dir)
    ensure_dir(artifacts_dir)

    # Reproducibility knobs
    np.random.seed(args.seed)

    # Data
    ds = fetch_california_housing(download_if_missing=True, as_frame=True)
    X = ds.data
    y = ds.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )

    # Model
    max_depth = None if args.max_depth <= 0 else args.max_depth
    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=max_depth,
        random_state=args.seed,
        n_jobs=-1,
    )

    # Unique run id for your raw artifacts folder and MLflow run name
    run_id = f"rf_seed{args.seed}_n{args.n_estimators}_d{args.max_depth}_{int(time.time())}"
    run_dir = artifacts_dir / run_id
    ensure_dir(run_dir)

    # Train
    t0 = time.time()
    model.fit(X_train, y_train)
    train_seconds = time.time() - t0

    # Eval
    preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "train_seconds": float(train_seconds),
        "n_test": int(len(y_test)),
    }

    # Save raw artifacts to volume
    joblib.dump(model, run_dir / "model.joblib")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    # Environment snapshot
    import numpy
    import sklearn

    run_info = RunInfo(
        started_at_unix=time.time(),
        python=sys.version.replace("\n", " "),
        platform=f"{platform.platform()} ({platform.machine()})",
        numpy=numpy.__version__,
        sklearn=sklearn.__version__,
        mlflow=mlflow.__version__,
        git_commit=try_cmd(["git", "rev-parse", "HEAD"]),
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        seed=args.seed,
        n_estimators=args.n_estimators,
        max_depth=max_depth,
        test_size=args.test_size,
    )
    (run_dir / "run_info.json").write_text(json.dumps(asdict(run_info), indent=2))

    # Log to MLflow
    with mlflow.start_run(run_name=run_id):
        mlflow.log_params(
            {
                "seed": args.seed,
                "n_estimators": args.n_estimators,
                "max_depth": -1 if max_depth is None else max_depth,
                "test_size": args.test_size,
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Convenience: store your JSONs in MLflow artifacts too
        mlflow.log_artifact(str(run_dir / "metrics.json"))
        mlflow.log_artifact(str(run_dir / "args.json"))
        mlflow.log_artifact(str(run_dir / "run_info.json"))

    print("RUN_ID:", run_id)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
