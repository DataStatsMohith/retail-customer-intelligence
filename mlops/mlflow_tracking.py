"""Helper utilities for MLflow experiment management."""
import mlflow

def get_best_run(experiment_name: str, metric: str = "silhouette_score"):
    client = mlflow.tracking.MlflowClient()
    exp    = client.get_experiment_by_name(experiment_name)
    if not exp:
        print(f"Experiment '{experiment_name}' not found.")
        return None
    runs = client.search_runs(exp.experiment_id, order_by=[f"metrics.{metric} DESC"])
    return runs[0] if runs else None

def register_model(run_id: str, model_name: str, artifact_path: str = "model"):
    mlflow.register_model(f"runs:/{run_id}/{artifact_path}", model_name)
