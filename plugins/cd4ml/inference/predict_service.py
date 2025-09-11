import os
import csv
import pathlib
import joblib
import mlflow
from datetime import datetime
import time
from typing import Optional
from dvc_push_manager import track_and_push_with_retry
import threading
from prometheus_client import generate_latest, Gauge, Counter, Histogram
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from evidently import ColumnMapping
#from evidently.model_monitoring import PrometheusMetricWriter
import pandas as pd
from sklearn.metrics import f1_score
import threading
import traceback
import warnings
from sklearn.exceptions import UndefinedMetricWarning

from fastapi import FastAPI, Depends, HTTPException, status, Header, Request, Response
from pydantic import BaseModel
from jose import jwt, JWTError
from mlflow.tracking import MlflowClient
from plugins.cd4ml.data_processing.preprocessing_core import ProductTypePredictorMLflow
from plugins.cd4ml.inference.utils import generate_id
from filelock import FileLock
from collections import defaultdict
from dotenv import load_dotenv

warnings.filterwarnings('ignore', category=UndefinedMetricWarning)
load_dotenv()
# --------------------------------
# PROMETHEUS implementation
# --------------------------------

# Prometheus Gauge for F1-Score
PREDICTION_F1_SCORE = Gauge(
    'prediction_f1_score',
    'F1 score of the prediction service based on feedback data'
)

# Prometheus Counter for prediction requests
PREDICTION_REQUESTS_TOTAL = Counter(
    'prediction_requests_total',
    'Total number of prediction requests to the predict service',
    ['method', 'endpoint', 'status_code']  # Added status_code
)

# Prometheus Histogram for prediction latency
PREDICTION_LATENCY = Histogram(
    'prediction_latency_seconds',
    'Latency of prediction requests in seconds',
    ['method', 'endpoint']
)

# Data drift monitoring
DATA_DRIFT_DETECTED = Gauge(
    'data_drift_detected',
    'Indicates if data drift was detected (1 if detected, 0 otherwise)'
)

DATA_DRIFT_SCORE = Gauge(
    'data_drift_score',
    'Overall data drift score'
)

# Predictin drift monitoring
PREDICTION_DRIFT_DETECTED = Gauge(
    'prediction_drift_detected',
    'Indicates if prediction drift was detected (1 if detected, 0 otherwise)'
)

PREDICTION_DRIFT_SCORE = Gauge(
    'prediction_drift_score',
    'Jensen-Shannon drift score for the predicted_code column'
)

# Model performance monitoring
MODEL_PERFORMANCE_F1_SCORE_CURRENT = Gauge(
    'model_performance_f1_score_current',
    'F1 score of the current model on recent feedback data with ground truth'
)

MODEL_PERFORMANCE_PRECISION_CURRENT = Gauge(
    'model_performance_precision_current',
    'Precision of the current model on recent feedback data with ground truth'
)


def calculate_and_expose_f1():
    """Continuously calculate and expose the weighted F1 score based on user feedback."""
    while True:
        try:
            if not os.path.exists(FEEDBACK_CSV_PATH):
                print(f"Feedback file not found at {FEEDBACK_CSV_PATH}. Setting F1 to 0.")
                PREDICTION_F1_SCORE.set(0)
                time.sleep(60) 
                continue

            df = pd.read_csv(FEEDBACK_CSV_PATH)

            required_columns = ['correct_code', 'predicted_code']
            if not all(col in df.columns for col in required_columns):
                print("Feedback file is missing 'correct_code' or 'predicted_code' columns. Setting F1 to 0.")
                PREDICTION_F1_SCORE.set(0)
                continue

            df['correct_code'] = pd.to_numeric(df['correct_code'], errors='coerce')
            df['predicted_code'] = pd.to_numeric(df['predicted_code'], errors='coerce')
            df_filtered = df.dropna(subset=['correct_code', 'predicted_code'])

            if df_filtered.empty:
                print("No valid feedback data (correct_code/predicted_code) available after filtering for F1 calculation. Setting F1 to 0.")
                PREDICTION_F1_SCORE.set(0)
            else:
                true_labels = df_filtered['correct_code'].astype(int)
                predicted_labels = df_filtered['predicted_code'].astype(int)

                if len(true_labels.unique()) == 1 and len(predicted_labels.unique()) == 1 and true_labels.unique()[0] == predicted_labels.unique()[0]:
                     f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)
                     PREDICTION_F1_SCORE.set(0)
                     print(f"Calculated F1 Score: {f1}")
                elif len(true_labels.unique()) < 2 or len(predicted_labels.unique()) < 2:
                    print("Filtered feedback data contains less than two unique classes. F1 score might be misleading or invalid. Setting F1 to 0.")
                    PREDICTION_F1_SCORE.set(0)
                else:
                    f1 = f1_score(true_labels, predicted_labels, average='weighted')
                    PREDICTION_F1_SCORE.set(f1)
                    print(f"Calculated F1 Score: {f1}")

        except pd.errors.EmptyDataError:
            print(f"Feedback file {FEEDBACK_CSV_PATH} is empty. Setting F1 to 0.")
            PREDICTION_F1_SCORE.set(0)
        except Exception as e:
            print(f"Error calculating F1 score: {e}")
            print(f"Full traceback: {traceback.format_exc()}") 
            PREDICTION_F1_SCORE.set(0)

def calculate_and_expose_drift():
    """
    Continuously monitor data and prediction drift, plus current model performance,
    and expose the results through Prometheus gauges.
    """
    try:
        if not os.path.exists(REFERENCE_DF_PATH):
            print(f"[ERROR] Reference data not found at {REFERENCE_DF_PATH}. Drift monitoring will not function.")
            reference_df = None 
        else:
            reference_df = pd.read_csv(REFERENCE_DF_PATH)
            print(f"[INFO] Loaded reference data for Evidently from {REFERENCE_DF_PATH}.")
    except Exception as e:
        print(f"[ERROR] Failed to load reference data for Evidently: {e}")
        reference_df = None

    while True:
        try:
            if reference_df is None:
                print("Skipping drift calculation: Reference data not loaded.")
                DATA_DRIFT_DETECTED.set(0) 
                DATA_DRIFT_SCORE.set(0)
                PREDICTION_DRIFT_DETECTED.set(0)
                PREDICTION_DRIFT_SCORE.set(0)
                MODEL_PERFORMANCE_F1_SCORE_CURRENT.set(0)
                MODEL_PERFORMANCE_PRECISION_CURRENT.set(0)
                time.sleep(300)
                continue

            if not os.path.exists(FEEDBACK_CSV_PATH):
                print(f"Feedback file not found at {FEEDBACK_CSV_PATH}. Skipping drift calculation.")
                DATA_DRIFT_DETECTED.set(0) 
                DATA_DRIFT_SCORE.set(0)
                PREDICTION_DRIFT_DETECTED.set(0)
                PREDICTION_DRIFT_SCORE.set(0)
                MODEL_PERFORMANCE_F1_SCORE_CURRENT.set(0)
                MODEL_PERFORMANCE_PRECISION_CURRENT.set(0)
                time.sleep(300)
                continue

            feedback_df = pd.read_csv(FEEDBACK_CSV_PATH)
            current_df = feedback_df[["designation", "description", "correct_code", "predicted_code"]]

            required_cols_for_performance = ['predicted_code', 'correct_code']
            current_df_for_performance = current_df.dropna(subset=required_cols_for_performance)
            if current_df_for_performance.empty:
                print("No valid data with ground truth for performance metrics. Setting performance F1 to 0.")
                MODEL_PERFORMANCE_F1_SCORE_CURRENT.set(0)
            else:
                current_df_for_performance['correct_code'] = pd.to_numeric(current_df_for_performance['correct_code'], errors='coerce').astype(int)
                current_df_for_performance['predicted_code'] = pd.to_numeric(current_df_for_performance['predicted_code'], errors='coerce').astype(int)
                current_df_for_performance.dropna(subset=['correct_code', 'predicted_code'], inplace=True) 
                reference_df = reference_df.rename(columns={"prdtypecode": "correct_code", "y_pred": "predicted_code"})
                reference_df = reference_df[["designation", "description", "correct_code", "predicted_code"]]
                reference_df.dropna(subset=['correct_code', 'predicted_code'], inplace=True) 
                reference_df['correct_code'] = pd.to_numeric(reference_df['correct_code'], errors='coerce').astype(int)


            
            reference_df_for_evidently = reference_df.copy()
            current_df_for_evidently = current_df_for_performance.copy()
            
            column_mapping = ColumnMapping(
                target="correct_code",
                prediction="predicted_code",
                text_features=["designation", "description"],
                task="classification"
            )
            column_mapping_data_drift = ColumnMapping(
                target=None,
                prediction="predicted_code",
                text_features=["designation", "description"],
                task="classification"
            )
            
            data_drift_report = Report(metrics=[DataDriftPreset()])
            model_performance_report = Report(metrics=[ClassificationPreset()])

            model_performance_report.run(reference_data=reference_df_for_evidently, 
                       current_data=current_df_for_evidently,
                       column_mapping=column_mapping)
            data_drift_report.run(reference_data=reference_df_for_evidently, 
                       current_data=current_df_for_evidently,
                       column_mapping=column_mapping_data_drift)


            # --- Extract metrics and send to Prometheus ---

            # Data Drift
            overall_drift = data_drift_report.as_dict()['metrics'][0]['result']['dataset_drift']
            drift_table = data_drift_report.as_dict()['metrics'][1]['result']['drift_by_columns']
            
            print(f"Data Drift Detected: {overall_drift}")
            for col_name, col_info in drift_table.items():
                print(
                    f"{col_name}: Drift detected? {col_info['drift_detected']} "
                    f"(score={col_info['drift_score']:.3f})"
                )

            DATA_DRIFT_DETECTED.set(1 if overall_drift else 0)
            DATA_DRIFT_SCORE.set(data_drift_report.as_dict()['metrics'][0]['result']['drift_share'])
                                 
            prediction_drift_score = drift_table['predicted_code']['drift_score']
            PREDICTION_DRIFT_SCORE.set(prediction_drift_score)
            PREDICTION_DRIFT_DETECTED.set(1 if drift_table['predicted_code']['drift_detected'] else 0)              
            print(f"prediction drift score: {prediction_drift_score}")
            
            overall_performance = model_performance_report.as_dict()['metrics'][0]['result']['current']
            f1 = overall_performance['f1']
            precision = overall_performance['precision']
            print(f"F1: {f1:.3f}, Precision: {precision:.3f}")

            MODEL_PERFORMANCE_F1_SCORE_CURRENT.set(f1)
            MODEL_PERFORMANCE_PRECISION_CURRENT.set(precision)


        except Exception as e:
            print(f"[ERROR] Error calculating drift with Evidently: {e}")
            DATA_DRIFT_DETECTED.set(0)
            PREDICTION_DRIFT_DETECTED.set(0)
            MODEL_PERFORMANCE_F1_SCORE_CURRENT.set(0)
            traceback.print_exc() 

        time.sleep(900) 



# --------------------------------
# FastAPI App 
# --------------------------------
predict_app = FastAPI()

# --- Middleware for request counting and latency ---
@predict_app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """
    Middleware to track prediction request metrics.

    Measures request processing time and logs metrics to Prometheus:
      - Total request count (`PREDICTION_REQUESTS_TOTAL`)
      - Request latency (`PREDICTION_LATENCY`)

    Parameters:
        request (Request): The incoming FastAPI request.
        call_next (Callable): Function to process the request and return a response.

    Returns:
        Response: The HTTP response with Prometheus metrics recorded.
    """
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    PREDICTION_REQUESTS_TOTAL.labels(
        method=request.method,
        endpoint=request.url.path,
        status_code=response.status_code,
    ).inc()
    PREDICTION_LATENCY.labels(method=request.method, endpoint=request.url.path).observe(
        process_time
    )
    return response

# --- Auth Config ---
SECRET_KEY = "your_secret_key_here"
ALGORITHM = "HS256"

# --- Globals ---
predictor: Optional[ProductTypePredictorMLflow] = None
prod_model_version = None
model_params = {}
model_metrics = {}
label_to_code = {}
valid_labels = set()

# Global in-memory store to track last prediction per user
user_last_prediction = defaultdict(dict)
CSV_LOCK_PATH = None  # Will be initialized in startup()

# --- Auth Helper ---
def verify_token(token: str = Header(...)):
    """
    Validates a JWT token from the Authorization header.
    Decodes the token using the configured secret and algorithm, and checks for a valid `sub` claim.
    Parameters:
        token (str): JWT token from the HTTP header.
    Returns:
        dict: Decoded token payload if valid.
    Raises:
        HTTPException (401): If the token is missing, invalid, or lacks a `sub` claim.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        return payload
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

# --- DVC Push Helper ---
def _async_track_and_push(description: str) -> None:
    """
    Asynchronously run a DVC/Git tracking and push operation in a background thread.
    Starts a daemon thread that calls `track_and_push_with_retry()` with the given description.
    Logs success, warning, or error messages based on the outcome.
    Parameters:
        description (str): A message describing the change to track and push.
    """
    def _worker():
        try:
            ok = track_and_push_with_retry(description=description, max_retries=3)
            if ok:
                print("[INFO] DVC/Git push succeeded.")
            else:
                print("[WARNING] DVC/Git push completed with warnings.")
        except Exception as e:
            print(f"[ERROR] DVC/Git push failed: {e}")

    threading.Thread(target=_worker, daemon=True).start()

# --- Pydantic Models ---
class PredictionRequest(BaseModel):
    designation: str
    description: Optional[str] = None

class PredictionResponse(BaseModel):
    predicted_class: str

class FeedbackInput(BaseModel):
    correct_label: str

# --- Startup Hook ---
@predict_app.on_event("startup")
def startup():
    """
    Initializes the prediction service on application startup:
    - Loads environment variables for paths, DAGsHub credentials, and configuration.
    - Sets up paths for feedback tracking and file locks.
    - Connects to DAGsHub-hosted MLflow to:
        - Fetch and load the production model, vectorizer, reference data, and product dictionary.
    - Initializes the predictor and supporting objects (e.g. label mappings).
    - Launches background threads for:
        - Continuous F1 score calculation from feedback.
        - Data and prediction drift monitoring using Evidently.
    Raises:
        RuntimeError: If any step in the initialization fails.
    """
    global FEEDBACK_CSV_PATH, predictor, prod_model_version
    global model_params, model_metrics, label_to_code, valid_labels, CSV_LOCK_PATH
    global REFERENCE_DF_PATH, product_dictionary_path

    # Load environment variables
    feedback_dir = os.getenv("DATA_FEEDBACK_DIR")
    feedback_filename = os.getenv("FEEDBACK_CSV")
    dagshub_user = os.getenv("DAGSHUB_USER_NAME")
    dagshub_token = os.getenv("DAGSHUB_USER_TOKEN")
    repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
    repo_name = os.getenv("DAGSHUB_REPO_NAME")

    # Setup feedback path and file lock path
    if feedback_dir and feedback_filename:
        pathlib.Path(feedback_dir).mkdir(parents=True, exist_ok=True)
        FEEDBACK_CSV_PATH = os.path.join(feedback_dir, feedback_filename)
        CSV_LOCK_PATH = FEEDBACK_CSV_PATH + ".lock"
        print(f"[INFO] Feedback CSV path: {FEEDBACK_CSV_PATH}")
    else:
        print("[WARNING] Feedback vars not set. Feedback functionality disabled.")
        FEEDBACK_CSV_PATH = None
        CSV_LOCK_PATH = None

    # Setup MLflow
    model_name = "SGDClassifier_Model"
    try:
        tracking_uri = f"https://{dagshub_user}:{dagshub_token}@dagshub.com/{repo_owner}/{repo_name}.mlflow"
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("rakuten_final_model")
        print(f"[INFO] MLflow tracking URI set")

        client = MlflowClient()
        print(f"[DEBUG] Fetching model alias 'production'...")
        prod_model_version = client.get_model_version_by_alias(model_name, "production")
        run_id = prod_model_version.run_id
        print(f"[INFO] Got run ID: {run_id}")

        print(f"[DEBUG] Loading production model...")
        production_model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}@production")

        run_info = client.get_run(run_id)
        model_params = run_info.data.params
        model_metrics = run_info.data.metrics
        print(f"[INFO] Model loaded successfully.")

        print(f"[DEBUG] Downloading vectorizer...")
        vectorizer_path_dir = client.download_artifacts(run_id=run_id, path="vectorizer")
        vectorizer_path = os.path.join(vectorizer_path_dir, "tfidf_vectorizer.pkl")
        print(f"[INFO] Vectorizer path: {vectorizer_path}")
        print(f"[INFO] Using remote vectorizer: {vectorizer_path}")

        print(f"[INFO] Downloading reference data for evidently")
        reference_df_dir = client.download_artifacts(run_id=run_id, path="monitoring_data")
        REFERENCE_DF_PATH = os.path.join(reference_df_dir, "reference_df_evidently.csv")
        print(f"[INFO] Reference data path: {REFERENCE_DF_PATH}")

        print(f"[DEBUG] Downloading product dictionary...")
        product_dict_dir = client.download_artifacts(run_id=run_id, path="product_dictionary")
        product_dictionary_path = os.path.join(product_dict_dir, "product_dictionary.pkl")
        print(f"[INFO] Product dictionary path: {product_dictionary_path}")
        print(f"[INFO] Using remote product dictionary: {product_dictionary_path}")


        with open(product_dictionary_path, "rb") as f:
            product_dict = joblib.load(f)
        label_to_code = {v: k for k, v in product_dict.items()}
        valid_labels = set(label_to_code.keys())

        predictor = ProductTypePredictorMLflow(
            model=production_model,
            vectorizer_path=vectorizer_path,
            product_dictionary_path=product_dictionary_path
        )

        print("[INFO] Predict service startup complete.")

        # Start the F1 score calculation thread
        f1_thread = threading.Thread(target=calculate_and_expose_f1, daemon=True)
        f1_thread.start()
        print("[INFO] F1 score calculation thread started.")

        # Start the drift monitoring thread
        drift_thread = threading.Thread(target=calculate_and_expose_drift, daemon=True)
        drift_thread.start()
        print("Drift monitoring thread started.")

    except Exception as e:
        print(f"[ERROR] Failed during startup: {e}")
        raise RuntimeError("Predict service startup failed. See logs for details.")

# --- Endpoints ---

@predict_app.get("/health")
def health_check():
    """
    Health check endpoint for the prediction service.

    Returns a simple JSON response indicating the service is running.

    Returns:
        dict: {
            "status": "healthy",
            "service": "predict_service"
        }
    """
    return {"status": "healthy", "service": "predict_service"}

@predict_app.get("/model-info")
def get_model_info(user=Depends(verify_token)):
    """
    Returns metadata about the deployed production model.

    Requires authentication. Responds with model version, registration timestamp,
    selected hyperparameters, and key performance metrics.

    Returns:
        dict: {
            "model_version": str,
            "registered_at": ISO 8601 timestamp,
            "parameters": {
                "alpha": float or str,
                "loss": str,
                "max_iter": int
            },
            "metrics": {
                "f1_weighted": float
            }
        }
    """
    return {
        "model_version": prod_model_version.version,
        "registered_at": datetime.fromtimestamp(prod_model_version.creation_timestamp / 1000).isoformat(),
        "parameters": {
            "alpha": model_params.get("alpha"),
            "loss": model_params.get("loss"),
            "max_iter": model_params.get("max_iter"),
        },
        "metrics": {
            "f1_weighted": model_metrics.get("f1_weighted")
        }
    }

@predict_app.post("/predict", response_model=PredictionResponse)
def predict_product_type(request: PredictionRequest, user=Depends(verify_token), fastapi_request: Request = None):
    """
    Predict the product type from a given designation and description.
    
    Stores the prediction result along with metadata (e.g., model version, session ID)
    into a CSV file and tracks the session for feedback purposes.
    
    Requires authentication.
    
    Parameters:
        request (PredictionRequest): Product input with designation and description.
        user (dict): Authenticated user.
    
    Returns:
        dict: {
            "predicted_class": str
        }
    """
    username = user.get("sub")
    prediction = predictor.predict(request.designation, request.description)
    predicted_code = label_to_code.get(prediction, "UNKNOWN")

    # Create unique session_id for tracking this prediction
    session_id = f"{username}_{datetime.now().timestamp()}"

    prediction_entry = {
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "model_version": prod_model_version.version,
        "designation": request.designation,
        "description": request.description,
        "productid": generate_id(10),
        "imageid": generate_id(12),
        "predicted_code": predicted_code,
        "predicted_label": prediction
    }

    # Track session in memory
    user_last_prediction[username] = {"id": session_id}

    # Ensure safe write with file lock
    file_exists = pathlib.Path(FEEDBACK_CSV_PATH).is_file()
    with FileLock(CSV_LOCK_PATH):
        with open(FEEDBACK_CSV_PATH, mode="a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=prediction_entry.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(prediction_entry)

    return {"predicted_class": prediction}


@predict_app.post("/feedback")
def submit_feedback(input: FeedbackInput, user=Depends(verify_token)):
    """
    Submit user feedback to correct the last predicted product label.

    Looks up the user's most recent prediction (via session ID) and updates
    the corresponding row in the feedback CSV with the corrected label, code, and correctness flag.

    Triggers an asynchronous DVC/Git push to persist the updated feedback.

    Parameters:
        input (FeedbackInput): Object containing the corrected label.
        user (dict): Authenticated user from token.
    
    Returns:
        dict: Success message indicating feedback was recorded.
    
    Raises:
        HTTPException:
            - 400 if the provided label is invalid.
            - 403 if no prediction session exists for the user.
            - 404 if the session is not found in the CSV log.
    """
    correct_label = input.correct_label
    username = user.get("sub")

    if correct_label not in valid_labels:
        raise HTTPException(status_code=400, detail="Invalid correct_label provided.")

    # Check session-based tracking
    session_id = user_last_prediction.get(username, {}).get("id")
    if not session_id:
        raise HTTPException(status_code=403, detail="No prediction found for current session.")

    correct_code = label_to_code[correct_label]

    # Read, find, and update only the matching session_id row
    with FileLock(CSV_LOCK_PATH):
        with open(FEEDBACK_CSV_PATH, mode="r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        updated = False
        for row in rows:
            if row.get("session_id") == session_id:
                row["correct_code"] = correct_code
                row["correct_label"] = correct_label
                row["is_correct"] = str(row["predicted_label"] == correct_label)
                updated = True
                break

        if not updated:
            raise HTTPException(status_code=404, detail="Session entry not found in CSV.")

        # Overwrite file safely
        with open(FEEDBACK_CSV_PATH, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    _async_track_and_push(description="append feedback correction")

    return {"status": "success", "message": "Feedback recorded."}

# --- Metrics Endpoint ---
@predict_app.get("/metrics")
async def metrics():
    """
    Expose Prometheus metrics for the prediction service.

    Returns metrics in plain text format compatible with Prometheus scraping.

    Returns:
        Response: Prometheus-formatted metrics with media type:
        'text/plain; version=0.0.4; charset=utf-8'
    """
    return Response(content=generate_latest(), media_type="text/plain; version=0.0.4; charset=utf-8")

