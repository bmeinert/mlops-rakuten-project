import os
import subprocess
import logging
import stat
import pwd
import grp

from datetime import timedelta
from dotenv import load_dotenv

from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

# Load .env 
load_dotenv('/opt/airflow/.env')

logger = logging.getLogger(__name__)


default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='ml_pipeline_mixed_experiment_dvc',
    default_args=default_args,
    description='ML Pipeline: all via DockerOperator',
    schedule_interval='@monthly',
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=['ml','training','docker','dvc'],
) as dag:

    PROJECT_DIR = os.environ["HOST_PROJECT_PATH"] # host path	
    logger.info(f"DEBUG: PROJECT_DIR resolved in DAG to: {PROJECT_DIR}")
    CONTAINER_PROJECT_PATH = '/app'
    CONTAINER_SHARED_VOLUME_PATH = os.path.join(CONTAINER_PROJECT_PATH, "shared-volume")
    CONTAINER_DVC_CACHE_PATH = os.path.join(CONTAINER_PROJECT_PATH, ".dvc", "_cache")

    DOCKER_SOCK = 'unix:///var/run/docker.sock'
    
    # 0. Env & Docker CLI sanity check
    def check_env(**context):
        required = [
            'GITHUB_TOKEN','GITHUB_REPO_OWNER','GITHUB_REPO_NAME',
            'DAGSHUB_USER_TOKEN','DAGSHUB_REPO_OWNER','DAGSHUB_REPO_NAME',
            'HOST_PROJECT_PATH' 
        ]
        missing = [v for v in required if not os.getenv(v)]
        if missing:
            raise ValueError(f"Missing env vars: {missing}")
        logger.info("Environment variables OK.")

    check_env_task = PythonOperator(
        task_id='check_environment',
        python_callable=check_env,
    )

    # 1. DVC Sync mit DockerOperator
    dvc_sync_task = DockerOperator(
        task_id='dvc-sync',
        image='mlops-rakuten-project-dvc-sync:latest', 
        working_dir=CONTAINER_PROJECT_PATH,
        command=['dvc', 'pull', 'shared_volume/data/raw', 'shared_volume/data/processed', 'shared_volume/data/feedback', 'shared_volume/models', '--force'],
        auto_remove=True,
        api_version='auto',
        docker_url=DOCKER_SOCK,
        network_mode='bridge',
        mounts=[
            Mount(
                source="dcv-cache-volume",
                target=CONTAINER_DVC_CACHE_PATH,
                type='volume'
            ),        
            # Mount of the shared volume as named volume
            Mount(
                source="shared-volume", 
                target=CONTAINER_SHARED_VOLUME_PATH,
                type='volume' 
            ),
        ],
        environment={
            'DAGSHUB_USER_TOKEN': os.getenv('DAGSHUB_USER_TOKEN'),
            'DAGSHUB_REPO_OWNER': os.getenv('DAGSHUB_REPO_OWNER'),
            'DAGSHUB_REPO_NAME': os.getenv('DAGSHUB_REPO_NAME'),
        },
        force_pull=False,
        mount_tmp_dir=False,
    )

    # Helper for the downstream Docker services
    def make_docker_task(task_id, image, command):
        return DockerOperator(
            task_id=task_id,
            image=image,
            api_version='auto',
            auto_remove=True,
            command=command,
            docker_url=DOCKER_SOCK,
            network_mode='bridge',
            mounts=[
                Mount(target='/app', source=PROJECT_DIR, type='bind'),
                Mount(source="dvc-cache", target=CONTAINER_DVC_CACHE_PATH),
                Mount(source="shared_volume", target=CONTAINER_SHARED_VOLUME_PATH, type='volume'),
            ],
            working_dir='/app',
            environment={
                'GITHUB_TOKEN': os.getenv('GITHUB_TOKEN'),
                'GITHUB_REPO_OWNER': os.getenv('GITHUB_REPO_OWNER'),
                'GITHUB_REPO_NAME': os.getenv('GITHUB_REPO_NAME'),
                'DAGSHUB_USER_NAME': os.getenv('DAGSHUB_USER_TOKEN'),
                'DAGSHUB_USER_TOKEN': os.getenv('DAGSHUB_USER_TOKEN'),
                'DAGSHUB_REPO_OWNER': os.getenv('DAGSHUB_REPO_OWNER'),
                'DAGSHUB_REPO_NAME': os.getenv('DAGSHUB_REPO_NAME'),
                'DATA_RAW_DIR': os.getenv('DATA_RAW_DIR'),
                'DATA_PROCESSED_DIR': os.getenv('DATA_PROCESSED_DIR'),
                'MODEL_DIR': os.getenv('MODEL_DIR'),
                'DATA_FEEDBACK_DIR': os.getenv('DATA_FEEDBACK_DIR'),
                'X_RAW': os.getenv('X_RAW'),
                'Y_RAW': os.getenv('Y_RAW'),
                'X_Y_RAW': os.getenv('X_Y_RAW'),
                'X_TRAIN_TFIDF': os.getenv('X_TRAIN_TFIDF'),
                'X_VALIDATE_TFIDF': os.getenv('X_VALIDATE_TFIDF'),
                'X_TEST' : os.getenv('X_TEST'),
                'X_TEST_TFIDF': os.getenv('X_TEST_TFIDF'),
                'Y_TRAIN': os.getenv('Y_TRAIN'),
                'Y_VALIDATE': os.getenv('Y_VALIDATE'),
                'Y_TEST': os.getenv('Y_TEST'),
                'TFIDF_VECTORIZER': os.getenv('TFIDF_VECTORIZER'),
                'MODEL': os.getenv('MODEL'),
                'PRODUCT_DICTIONARY': os.getenv('PRODUCT_DICTIONARY'),
                'CLASS_REPORT': os.getenv('CLASS_REPORT'),
                'CLASS_REPORT_VALIDATION': os.getenv('CLASS_REPORT_VALIDATION'),
                'CURRENT_RUN_ID': os.getenv('CURRENT_RUN_ID'),
                'PARAM_CONFIG': os.getenv('PARAM_CONFIG'),
                'FEEDBACK_CSV': os.getenv('FEEDBACK_CSV'),
                'RETRAIN_RAW': os.getenv('RETRAIN_RAW'),
                'REFERENCE_EVIDENTLY' : os.getenv('REFERENCE_EVIDENTLY'),
                'PYTHONPATH': '/app',
                'PYTHONUNBUFFERED': '1', 
            },
            force_pull=False,
            mount_tmp_dir=False,
        )

    preprocessing = make_docker_task(
        'preprocessing',
        'mlops-rakuten-project-preprocessing:latest',
        'python plugins/cd4ml/data_processing/run_preprocessing.py'
    )

    model_training = make_docker_task(
        'model_training',
        'mlops-rakuten-project-model_training:latest',
        'python plugins/cd4ml/model_training/run_model_training.py'
    )

    model_validation = make_docker_task(
        'model_validation',
        'mlops-rakuten-project-model_validation:latest',
        'python plugins/cd4ml/model_validation/run_model_validation.py'
    )

    # DAG ordering
    check_env_task >> dvc_sync_task >> preprocessing >> model_training >> model_validation
