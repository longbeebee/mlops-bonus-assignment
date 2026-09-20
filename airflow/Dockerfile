FROM apache/airflow:2.10.3-python3.11
COPY airflow/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY airflow/dags /opt/airflow/dags
COPY app /opt/project/app
ENV PYTHONPATH=/opt/project
