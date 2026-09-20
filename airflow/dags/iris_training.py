from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator


def load_data_callable():
    from app.train import load_data
    return load_data()


def train_model_callable(data_info):
    from app.train import train_model
    return train_model(data_info)


def evaluate_model_callable(training):
    from app.train import evaluate_model
    return evaluate_model(training)


def register_model_callable(training):
    from app.train import register_model
    return register_model(training)


def promote_alias_callable(registered):
    from app.train import promote_alias
    return promote_alias(registered)


with DAG(
    dag_id="iris_mlops_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False,
    default_args={"owner": "mlops", "retries": 2, "retry_delay": timedelta(minutes=1)},
    tags=["training", "mlflow", "sklearn"],
) as dag:
    load_data_task = PythonOperator(task_id="load_data", python_callable=load_data_callable)
    train_model_task = PythonOperator(
        task_id="train_model", python_callable=train_model_callable, op_args=[load_data_task.output]
    )
    evaluate_model_task = PythonOperator(
        task_id="evaluate_model", python_callable=evaluate_model_callable, op_args=[train_model_task.output]
    )
    register_model_task = PythonOperator(
        task_id="register_model", python_callable=register_model_callable, op_args=[evaluate_model_task.output]
    )
    promote_alias_task = PythonOperator(
        task_id="promote_alias", python_callable=promote_alias_callable, op_args=[register_model_task.output]
    )

    load_data_task >> train_model_task >> evaluate_model_task >> register_model_task >> promote_alias_task
