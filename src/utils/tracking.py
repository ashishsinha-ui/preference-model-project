"""Local experiment tracking with MLflow (replaces W&B; everything stays on this server).

Run names follow the plan: {model_tier}_{label_type}_{dataset}_seed{seed}
  e.g. t2_bert_hard_hh_seed42, t3_deberta_soft_shp_seed0
Runs trained on UltraFeedback (AI labels) get the [AI-Label] prefix and a label_source tag.

View the dashboard:
  mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5050
"""
from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parents[2]
TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
EXPERIMENT = "pairwise-preference-model"


def start_run(name, config, ai_label=False, enabled=True):
    """Start a run and log its config. enabled=False turns tracking into a no-op."""
    if not enabled:
        return None
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=(ROOT / "mlruns").as_uri())
    mlflow.set_experiment(EXPERIMENT)
    if ai_label and not name.startswith("[AI-Label]"):
        name = f"[AI-Label] {name}"
    run = mlflow.start_run(run_name=name, log_system_metrics=True,
                           tags={"label_source": "ai" if ai_label else "human"})
    mlflow.log_params({k: str(v) for k, v in config.items()})
    return run


def log(metrics, step=None):
    if mlflow.active_run():
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)


def log_file(path):
    """Attach a small file (plot, results json) to the run. Never use for checkpoints."""
    if mlflow.active_run():
        mlflow.log_artifact(str(path))


def finish():
    if mlflow.active_run():
        mlflow.end_run()
