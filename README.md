# MLOps bonus assignment

Stack chạy trong một Docker network `mlops-network`:

`Airflow -> MLflow -> PostgreSQL (metadata) + MinIO (artifact)`, và `FastAPI -> MLflow/MinIO` để phục vụ model theo alias `champion`.

## Khởi động

```bash
cp .env.example .env
docker compose build
docker compose up -d
```

Mở các giao diện:

- Airflow: http://localhost:8080 (`admin` / `admin`)
- MLflow: http://localhost:5000
- MinIO: http://localhost:9001 (`minioadmin` / `minioadmin`)
- FastAPI docs: http://localhost:8000/docs

Trong Airflow, bật DAG `iris_mlops_pipeline` và trigger một run. Graph gồm 5 task: `load_data` → `train_model` → `evaluate_model` → `register_model` → `promote_alias`. DAG dùng dataset `sklearn.datasets.load_iris`, train Logistic Regression, log parameters/metrics/tags, chỉ register khi accuracy đạt ngưỡng, rồi gắn tag `validation=passed` và alias `champion`. Artifact model đi vào bucket MinIO `mlflow`; metadata tracking đi vào PostgreSQL.

Sau khi DAG thành công:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'content-type: application/json' \
  -d '{"features":[5.1,3.5,1.4,0.2]}'
```

Kiểm tra trạng thái:

```bash
docker compose ps
docker compose logs -f airflow-scheduler
```

Dừng stack nhưng giữ dữ liệu: `docker compose down`. Xóa cả volumes chỉ khi muốn reset toàn bộ metadata/artifact: `docker compose down -v`.
