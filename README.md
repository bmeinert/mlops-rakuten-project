# Rakuten Product Category Classification – MLOps Reference Implementation

A production‑grade, **end‑to‑end MLOps pipeline** for automatic product‑type classification on the Rakuten retail dataset.

> **Project Scope (2025):** This fork focuses *exclusively* on **text‑based classification**. Unlike the original data‑science prototype that blended CNN image features with TF‑IDF, we rely solely on an SGD + TF‑IDF model (\~82 % weighted F1 across 27 classes). All accompanying MLOps infrastructure—data versioning, CI/CD, secure inference, monitoring—targets this single‑modal pipeline.

---

## Table of Contents

1. [High‑Level Architecture](#high‑level-architecture)
2. [Repository Layout](#repository-layout)
3. [Quick Start](#quick-start)
4. [Pipeline Walk‑Through](#pipeline-walk‑through)
5. [Versioning & Experiment Tracking](#versioning--experiment-tracking)
6. [Orchestration & Automation](#orchestration--automation)
7. [Monitoring & Observability](#monitoring--observability)
8. [Security & Access Control](#security--access-control)
9. [Testing & CI/CD](#testing--cicd)
10. [Project Roadmap](#project-roadmap)
11. [Contributing](#contributing)
12. [License](#license)

*For detailed instructions on the monitoring components, see* [READ\_ME\_FOR\_MONITORING.md](READ_ME_FOR_MONITORING.md).

---

## High‑Level Architecture


```mermaid
graph LR;
  %% Environment setup
  subgraph "Environment setup"
    direction LR
    GH["Code Version Control<br>GitHub"]
    DS_env["Data Versioning & Storage<br>Dagshub + S3"]
    GA["CI/CD<br>GitHub Actions"]
    DOCK["Virtualization<br>docker"]
    AF["Automation<br>Airflow"]
    %% invisible connectors (still count as links 0–3)
    GH -.-> DS_env -.-> GA -.-> DOCK -.-> AF
  end
  %% hide those first four links
  linkStyle 0,1,2,3 opacity:0

  %% Data Management
  subgraph "Data Management"
    direction LR
    RAW[/Raw Product<br>Data/]
    FB[/Human<br>Feedback/]
    PRE[Preprocessing<br>Service]
    CONS[/Consolidated<br>Dataset/]
  end

  %% ML Pipeline Steps
  subgraph "Model Training & Validation"
    direction LR
    TRN[Model Training<br>Service]
    VAL[Model Validation<br>Service]
  end

  %% Model Tracking & Registry
  subgraph "Model Tracking & Registry"
    direction LR
    ML[Model Tracking<br>MLflow]
    REG[Model Registry<br>MLflow]
    MODEL_ART[Model Artifacts<br>MLflow]
  end


  %% Deployment & Security
  subgraph "Deployment & Security"
    direction LR
    UI["User Interface<br>Streamlit App"]
    AUTH[Authentication Service<br>FastAPI]
    API[Prediction Service<br>FastAPI]
  end

  %% Monitoring
  subgraph "Monitoring"
    direction LR
    EV[Model Monitoring<br>Evidently AI]
    PR[Performance Metrics<br>Prometheus]
    GF[Dashboards<br>Grafana]
    ALERT[Performance Alerts<br>Alert Manager]
  end


  %% Control & Data Flows
  RAW & FB --> PRE --> CONS --> TRN --> VAL

  TRN --> ML --> MODEL_ART & REG
  VAL --> ML

  MODEL_ART & REG--> API
  MODEL_ART --> EV
  AUTH --> API

  UI --> API
  UI --> AUTH

  Users((End Users)) --> UI

  API --> FB
  API --> EV --> PR 
  API --> PR 
  AUTH --> PR 
  PR --> ALERT & GF
```

*The diagram shows the complete lifecycle from raw data ingestion through to monitored deployment in production.*

---

## Repository Layout

```text
mlops-rakuten-project/
├── docker-compose.yml              # Micro‑services
├── dockerfiles/                    # One Dockerfile per micro‑service
├── dags/                           # Airflow DAGs
├── plugins/                        # Domain logic
│   └── cd4ml/                      #  ├─ alertmanager_helper/
                                    #  ├─ data_processing/
│                                   #  ├─ model_training/
│                                   #  ├─ model_validation/
|                                   #  ├─ inference/
│                                   #  └─ tests/
├── monitoring/                     # Prometheus / Grafana configs
├── shared_volume/                  # **DVC‑tracked** data & models
├── ui_app.py                       # Streamlit front‑end
├── READ_ME_FOR_MONITORING.md       # Full monitoring guide
└── README.md                       # (this file)
```

> **Note:** All persistent artefacts (data, models, vectorizer, feedback CSV …) live under `shared_volume/` or in MLflow artifacts and are version‑controlled with **DVC**.

---

## Quick Start

### 1 – Prerequisites

* **Docker ≥ 23** – Runtime engine for all containerised services.
* **Git** – Required to clone / fork *this* repository; present inside the containers for internal commits.
* **DVC** – Already included into the Docker images for pipeline steps. Install *locally* only if you need to run `dvc pull/push` outside the containers (see [`READ_ME_FOR_DVC.md`](READ_ME_FOR_DVC.md)).

### 2 – Clone & configure

```bash
git clone https://github.com/BeritM/mlops-rakuten-project.git
cd mlops-rakuten-project
cp .env.example .env        # fill in GitHub & Dagshub creds
```

### 3 – Run the API and monitoring stack

```bash
docker compose up --build -d dvc-sync auth_service predict_service postgres airflow-init airflow-webserver airflow-scheduler blackbox-exporter prometheus grafana alertmanager webhook_proxy
# ⇒  http://localhost:8001  Auth Service
# ⇒  http://localhost:8002  Prediction API
# ⇒  http://localhost:8080  Airflow
# ⇒  http://localhost:9090  Prometheus
# ⇒  http://localhost:9093  Alertmanager
# ⇒  http://localhost:3000  Grafana
```


### 4 – (Optionally) start the core stack locally

```bash
# Pull raw data / models from Dagshub & execute the pipeline
docker compose up --build -d dvc-sync preprocessing model_training model_validation

```
Artifacts and metrics will be pushed automatically to GitHub / Dagshub.

### 5 – Launch the Streamlit demo

```bash
streamlit run ui_app.py
```

---

## Pipeline Walk‑Through

### Data Management

* **Ingestion & Pre‑processing** `plugins/cd4ml/data_processing/run_preprocessing.py` Ingests original training data set and human feedback provided through the prediction service, cleans multilingual text, splits sets, computes TF‑IDF, stores artifacts to DVC.
* **Versioning** Full lineage of raw/processed data, models and feedback stored on a Dagshub‑backed **S3 bucket**.

### Model Training

* **Algorithm** `SGDClassifier` (logistic loss) with custom class weights.
* **Experiment Tracking** All params, metrics & artefacts logged to **MLflow**; model registered in `experiments`.

### Validation & Auto‑Promotion

* Validation container compares new F1‑score against the currently deployed model and, if improved, *promotes* the run via MLflow aliasing (alias `production`).

### Deployment

* **Auth Service** – FastAPI app that implements OAuth2 login against an in-memory user store, issues and verifies HS256 JWTs (with a 30 min expiry), enforces role-based (admin) CRUD on `/users`, and exposes `/health` plus a Prometheus-compatible `/metrics` endpoint 
* **Predict Service** – FastAPI app that verifies HS256 JWTs, loads the TF-IDF vectorizer and SGD model from MLflow on startup, exposes `/predict` (logging each request’s output to a CSV with file-locking and asynchronous DVC/Git pushes), `/feedback` (updating that CSV and triggering pushes), `/model-info`, `/health`, and Prometheus-compatible `/metrics` endpoints, and continuously computes an F1-score gauge plus request-count and latency metrics as well as evidently drift metrics via background threads and middleware.
* **UI** – A Streamlit app that provides a graphical interface for end users to interact with the Authentication and Prediction services. **User role**: Get a product category prediction for a product (name & description)  & confirm the correctness of prediction. **Admin role**: Manage users, view production-model details, and manually trigger model retraining.  
---

## Versioning & Experiment Tracking

| Layer             | Tooling      | Remote                       |
| ----------------- | ------------ | ---------------------------- |
| **Code**          | Git + GitHub | `main` ➜ protected           |
| **Data / Models** | DVC          | S3 bucket (Dagshub)          |
| **Experiments**   | MLflow       | Dagshub‑hosted MLflow server |

The helper script `dvc_push_manager.py` automates *DVC ➜ Git* synchronisation.

---

## Orchestration & Automation

* **Local** – `docker‑compose.yml` wires the individual services and shared volumes.
* **Batch** – `ml_pipeline_mixed_experiment_dvc.py` Airflow DAG runs weekly and orchestrates the four Docker containers with `DockerOperator`.
* **CI / Tests** – GitHub Actions build images, execute linters + the PyTest suite located in `plugins/cd4ml/tests`.

---

## Monitoring & Observability

*For an in‑depth walkthrough, refer to* [READ\_ME\_FOR\_MONITORING.md](READ_ME_FOR_MONITORING.md).

| Aspect                 | Current State                                                                                         |
| ---------------------- | ----------------------------------------------------------------------------------------------------- |
| **Service health**     | FastAPI `/health` endpoints + Docker health‑checks; probed by Prometheus Blackbox Exporter            |
| **Metrics**            | Evidently‑derived and service‑level metrics exported via Prometheus Python client (`/metrics`)        |
| **Dashboards**         | Pre‑provisioned Grafana dashboards (auth, drift/F1, performance) under `monitoring/grafana/…`         |
| **Data / Model Drift** | Daily drift job (see `drift_monitor/`) pushes Evidently scores to Prometheus; alerts trigger Airflow  |
| **Alerting**           | Prometheus rules (e.g. *ModelF1Drop* > 10 %) routed to Alertmanager ➜ Airflow retraining webhook      |

---

## Security & Access Control

* Separate **auth** and **predict** micro‑services – principle of least privilege.
* JWT tokens (`/login`) include `sub` and `role`; default expiry 30 min.
* Admin endpoints protected by middleware (`Depends(admin_required)`).
* All inter‑service traffic remains inside the Docker network; only ports 8001/8002 exposed.

---

## CI/CD & Testing

A **single GitHub Actions workflow** (`.github/workflows/docker-publish.yml`) runs on each push to or pull from **`main`** or **`develop`**.

| Stage                   | What happens?                                                                                                          |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| **CodeQL Scan**         | Static code‑ & security‑analysis for Python.                                                                           |
| **Pytest**              | Runs unit tests for preprocessing & training *and* integration tests for API routes (via the dedicated `tests` image). |
| **Docker Build + Push** | Matrix builds *all* service images (dvc‑sync, preprocessing, training …) with Buildx and pushes them to Docker Hub.    |
| **Compose Build Check** | Verifies that the full `docker‑compose.yml` still builds end‑to‑end (no containers started).                           |

**Workflow Highlights**

* Layer‑caching and pip‑caching speed up builds.
* The `tests` container spins up **auth\_service + predict\_service**, waits for health checks and runs full request‑flow tests.
* Concurrency guard cancels redundant runs on the same branch.
* Docker Hub creds are injected via GitHub Secrets.

A green check‑mark in GitHub shows the pipeline passed.

---

### Local Endpoints

| Service        | URL                     | Notes                                                         |
| -------------- | ----------------------- | ------------------------------------------------------------- |
| Auth API       | `http://localhost:8001` | `/login`, `/users`, `/health`, `/metrics`                     |
| Predict API    | `http://localhost:8002` | `/predict`, `/model-info`, `/feedback`, `/health`, `/metrics` |
| Airflow Web UI | `http://localhost:8080` | DAG management                                                |
| Prometheus\*   | `http://localhost:9090` | Metrics explorer                                              |
| Alertmanager\* | `http://localhost:9093` | Alert routing                                                 |
| Grafana\*      | `http://localhost:3000` | Dashboards                                 |

*Endpoints marked with an asterisk are available only when the monitoring stack is running (**`docker-compose.monitoring.yml`**).*

**Streamlit UI** (launch locally):

```bash
streamlit run ui_app.py --server.port 8501
```

→ [http://localhost:8501](http://localhost:8501)

---

## Outlook / Next Steps

* **Data persistence** – move raw & feedback tables to PostgreSQL, store large artefacts (models, vectoriser, metrics) in object storage (S3/GCS) for cheaper, scalable retention.
* **Auto‑scaling deployment** – containerise inference stack on Kubernetes (e.g. K3s or EKS) with Horizontal Pod Autoscaler.
* **Feature Store** – introduce Feast to guarantee online/offline feature parity.
* **Continuous performance control** – hook Evidently alerts into Prometheus → Grafana; trigger auto‑retraining DAG on drift.
* **Rate-limiting and alerts** - detect unusually high request volumes (potential DDoS) and optionally disable or throttle the service if thresholds are exceeded.
* **Dashboard enhancement** - additional metrics such as resource utilization, auth attempts, ... to get more information and enhance systems and models.
* **Security hardening** – container image scanning (Trivy) & Dependabot for vuln alerts.
* **Resource Efficiency Improvements** - optimize container and service configurations to reduce computational load.
* **Image recognition models** - enable multimodal inference and enrich the system’s predictive capabilities.


---

## License

Licensed by the Rakuten Institute of Technology.

---

### Contributors

* BeritM
* Anomalisaa
* SebastianPoppitz
* Tubetraveller
