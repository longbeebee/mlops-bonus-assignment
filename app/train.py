"""Task functions for the Airflow Iris training pipeline."""
import os
from datetime import datetime, timezone

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


def _configure_mlflow() -> tuple[str, str, str]:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    model_name = os.getenv("MLFLOW_MODEL_NAME", "sklearn-iris-classifier")
    alias = os.getenv("MLFLOW_MODEL_ALIAS", "champion")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("iris-training")
    return tracking_uri, model_name, alias


def load_data() -> dict:
    dataset = load_iris(as_frame=True)
    return {
        "dataset": "sklearn.datasets.load_iris",
        "rows": len(dataset.data),
        "features": list(dataset.feature_names),
        "target_classes": len(dataset.target.unique()),
    }


def train_model(data_info: dict) -> dict:
    tracking_uri, _, _ = _configure_mlflow()
    dataset = load_iris(as_frame=True)
    x_train, x_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.2, random_state=42, stratify=dataset.target
    )
    model = LogisticRegression(max_iter=300, random_state=42)
    with mlflow.start_run(run_name=f"iris-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}") as run:
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average="macro")
        mlflow.log_params({"model_type": "LogisticRegression", "max_iter": 300, "random_state": 42})
        mlflow.log_metrics({"accuracy": accuracy, "f1_macro": f1})
        mlflow.set_tags({"dataset": data_info["dataset"], "pipeline": "airflow", "stage": "candidate"})
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            metadata={"features": list(dataset.feature_names), "target": "iris species"},
        )
        result = {"run_id": run.info.run_id, "accuracy": accuracy, "f1_macro": f1}
    print(f"run_id={result['run_id']} accuracy={accuracy:.4f} tracking_uri={tracking_uri}")
    return result


def evaluate_model(training: dict) -> dict:
    minimum_accuracy = 0.90
    if training["accuracy"] < minimum_accuracy:
        raise ValueError(f"model accuracy {training['accuracy']:.4f} < {minimum_accuracy:.2f}")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    MlflowClient().set_tag(training["run_id"], "validation", "passed")
    return training


def register_model(training: dict) -> dict:
    tracking_uri, model_name, _ = _configure_mlflow()
    model_version = mlflow.register_model(f"runs:/{training['run_id']}/model", model_name)
    client = MlflowClient(tracking_uri=tracking_uri)
    client.set_model_version_tag(model_name, model_version.version, "validation", "passed")
    client.set_model_version_tag(model_name, model_version.version, "accuracy", str(round(training["accuracy"], 6)))
    result = {**training, "model_name": model_name, "version": model_version.version}
    print(f"registered model={model_name} version={model_version.version}")
    return result


def promote_alias(registered: dict) -> None:
    tracking_uri, model_name, alias = _configure_mlflow()
    client = MlflowClient(tracking_uri=tracking_uri)
    client.set_registered_model_alias(model_name, alias, registered["version"])
    print(f"promoted model={model_name} version={registered['version']} alias={alias}")
