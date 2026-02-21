"""
End-to-end training pipeline with MLflow experiment tracking.
Demonstrates MLOps practices: versioning, logging, model registry.
"""
import mlflow
import mlflow.sklearn
import yaml
import joblib
from pathlib import Path
import sys; sys.path.insert(0, ".")

from data.generate_synthetic_data import generate_data
from src.data.loader import load_transactions, load_products, load_customers
from src.data.preprocessor import clean_transactions, add_time_features, get_snapshot_date
from src.features.rfm_features import compute_rfm, score_rfm, add_behavioural_features
from src.models.segmentation import CustomerSegmentation
from src.models.recommender import HybridRecommender
from src.evaluation.metrics import evaluate_segmentation
from src.utils.logger import get_logger

logger = get_logger("train_pipeline")

def load_config():
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)

def run_training():
    cfg = load_config()

    # ── 1. Data Generation (if not exists) ─────────────────
    if not Path("data/raw/transactions.csv").exists():
        logger.info("Generating synthetic data...")
        generate_data()

    # ── 2. Load & Preprocess ────────────────────────────────
    transactions = load_transactions()
    products     = load_products()
    transactions = clean_transactions(transactions)
    transactions = add_time_features(transactions)

    # ── 3. Feature Engineering ──────────────────────────────
    snapshot    = get_snapshot_date(transactions)
    rfm         = compute_rfm(transactions, snapshot)
    rfm         = score_rfm(rfm)
    rfm         = add_behavioural_features(transactions, rfm)

    # ── 4. MLflow Tracking ──────────────────────────────────
    mlflow.set_tracking_uri(cfg["mlops"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlops"]["experiment_name"])

    with mlflow.start_run(run_name="segmentation_run"):
        # ── 5. Segmentation ─────────────────────────────────
        seg_model = CustomerSegmentation(
            n_clusters=cfg["segmentation"]["n_clusters"],
            random_state=cfg["segmentation"]["random_state"]
        )
        rfm_segmented, seg_metrics = seg_model.fit_kmeans(rfm)
        rfm_segmented = seg_model.label_segments(rfm_segmented)

        mlflow.log_params({
            "n_clusters":    cfg["segmentation"]["n_clusters"],
            "algorithm":     "kmeans",
            "n_customers":   len(rfm),
        })
        mlflow.log_metrics(seg_metrics)
        logger.info(f"Segmentation metrics: {seg_metrics}")

        # Save segmented customers
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        rfm_segmented.to_csv("data/processed/segmented_customers.csv", index=False)
        mlflow.log_artifact("data/processed/segmented_customers.csv")

        # ── 6. Recommendation Engine ────────────────────────
    with mlflow.start_run(run_name="recommender_run"):
        recommender = HybridRecommender(
            n_recommendations=cfg["recommender"]["n_recommendations"]
        )
        recommender.fit(transactions, products)

        mlflow.log_params({
            "model_type":          "hybrid",
            "cf_weight":           0.7,
            "n_recommendations":   cfg["recommender"]["n_recommendations"],
        })

        # Save models
        Path("mlops/model_registry").mkdir(parents=True, exist_ok=True)
        joblib.dump(seg_model,   "mlops/model_registry/segmentation_model.pkl")
        joblib.dump(recommender, "mlops/model_registry/recommender_model.pkl")
        mlflow.log_artifact("mlops/model_registry/segmentation_model.pkl")
        mlflow.log_artifact("mlops/model_registry/recommender_model.pkl")

        logger.info("Training complete. Models saved to mlops/model_registry/")

if __name__ == "__main__":
    run_training()
