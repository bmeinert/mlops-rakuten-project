# Monitoring Setup

This document provides an end-to-end guide for setting up model monitoring using Evidently, Prometheus and Grafana. 
It covers everything from the existing exporter in the FastAPI service (the exporter lives in: plugins/cd4ml/inference/predict_service.py) 
to alerting and dashboard provisioning.


# Prerequisites
1. Python Exporter (FastAPI Service in predict_service.py)

2. Prometheus/Alertmanager Setup  
  2.1 prometheus.yml  
  2.2 alert_rules.yml  
  2.3 alertmanager.yml  

3. Grafana Setup

    3.1 dashboards.yml
    3.2 Dashboard JSON Files

# Project Structure
```text
monitoring/
├── grafana/
│   └── provisioning/
│       └── dashboards/
│           ├── authentication_dashboard.json
│           ├── drift_and_f1_dashboard.json
│           ├── prediction_performance_dashboard.json
│           └── dashboards.yml
└── prometheus/
    ├── prometheus.yml
    ├── alert_rules.yml
    └── alertmanager.yml
plugins/
└── cd4ml/
    └── inference/
        └── predict_service.py  # FastAPI exporter
```

# Full explanation of the monitoring service

The exporter in plugins/cd4ml/inference/predict_service.py uses the Prometheus Python client to expose metrics computed by Evidently on the /metrics endpoint of two services:

- Auth service on port 8001
- Predict service on port 8002

In monitoring/prometheus/prometheus.yml, the scrape_configs section defines pull jobs that Prometheus runs every 15 seconds. In our setup, Prometheus scrapes the Evidently‐derived metrics directly from both the Auth and Predict services, and it also uses the Blackbox Exporter to probe the Predict service’s health-check endpoint.

The monitoring/grafana/provisioning/dashboards/dashboards.yml file tells Grafana (running on port 3000) where to find the JSON dashboard definitions. Grafana then automatically imports each .json file under that directory as a separate dashboard, and every panel within them is defined by a PromQL query against the Prometheus data source.

The monitoring/prometheus/alert_rules.yml file defines Prometheus alerting rules—most importantly, it fires a ModelF1Drop alert whenever the prediction_f1_score metric falls by more than 10 % compared to one hour ago.

In monitoring/prometheus/alertmanager.yml, a webhook receiver is configured so that any alert fired by Prometheus triggers an HTTP POST to the Airflow REST API endpoint for the ml_pipeline_mixed_experiment_dvc DAG:

```bash
POST http://airflow-webserver:8080/api/v1/dags/ml_pipeline_mixed_experiment_dvc/dagRuns
```

This POST automatically kicks off the retraining pipeline in Airflow.

# Step-by-Step File Contents

## plugins/cd4ml/inference/predict_service.py

- Imported and defined Gauge, Counter, Histogram metrics.

- Exposed /metrics endpoint using generate_latest().

## monitoring/prometheus/prometheus.yml

- Configure global scrape interval and include rule_files.

- Add scrape_configs for prediction_service and api_probe (Blackbox Exporter).

## monitoring/prometheus/alert_rules.yml

- Define the ModelF1Drop alert rule (10% drop, 5m).

## monitoring/prometheus/alertmanager.yml

- Set the global resolve_timeout and route to airflow_retrain.

- Configure webhook_configs to post to Airflow's DAG-run endpoint.

## monitoring/grafana/provisioning/dashboards/dashboards.yml

- Declare the file-based provider pointing to the dashboards/ folder.

# monitoring/grafana/provisioning/dashboards/*.json

- JSON dashboard files authentication, drift & F1, prediction performance are visible in Grafana as Dashboard panels.


# Running the Stack

- Run the docker-compose files 

Attention: They need to be started simultanously to create a common network.
```bash
docker-compose -f docker-compose.yml -f docker-compose.monitoring.yml up --build -d
``` 

- First compose starts Airflow, Auth, Predict, DVC ...
- Second compose starts Prometheus, Alertmanager, Grafana, Blackbox

- Check Status
```bash
docker-compose -f docker-compose.yml -f docker-compose.monitoring.yml ps
```


# Verify running services on localhost:

Auth metrics: http://localhost:8001/metrics  

Predict metrics: http://localhost:8002/metrics  

Prometheus UI: http://localhost:9090

Alertmanager: http://localhost:9093

Grafana UI: http://localhost:3000 -> (admin/admin123) -> go to Dashboards

Airflow: http://localhost:8080 (airflow/airflow)