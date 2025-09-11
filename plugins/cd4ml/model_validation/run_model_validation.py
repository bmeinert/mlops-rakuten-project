"""
run_model_validation.py
"""

import sys
import os
import mlflow
import pandas as pd
import pickle
from mlflow.tracking import MlflowClient
from dvc_push_manager import track_and_push_with_retry
from model_validation import prediction_and_metrics
from plugins.cd4ml.model_training.model_training import load_train_data

# ---------- Paths ----------
INPUT_DIR  = os.getenv("DATA_PROCESSED_DIR")
OUTPUT_DIR = os.getenv("MODEL_DIR")
X_TEST_PATH = os.path.join(INPUT_DIR, os.getenv("X_TEST"))
X_TEST_TFIDF_PATH = os.path.join(INPUT_DIR, os.getenv("X_TEST_TFIDF"))
Y_TEST_PATH       = os.path.join(INPUT_DIR, os.getenv("Y_TEST"))
MODEL_FILE        = os.getenv("MODEL")
CLASS_REPORT_PATH = os.path.join(OUTPUT_DIR, os.getenv("CLASS_REPORT_VALIDATION"))
REFERENCE_EVIDENTLY_PATH = os.path.join(OUTPUT_DIR, os.getenv("REFERENCE_EVIDENTLY"))

# ---------- Dagshub ----------
DAGSHUB_USER_NAME = os.getenv("DAGSHUB_USER_NAME")
DAGSHUB_USER_TOKEN = os.getenv("DAGSHUB_USER_TOKEN")
DAGSHUB_REPO_OWNER = os.getenv("DAGSHUB_REPO_OWNER")
DAGSHUB_REPO_NAME  = os.getenv("DAGSHUB_REPO_NAME")

# ---------- OTHER ----------
MODEL_NAME = "SGDClassifier_Model"


def validate_environment():
    """Validate that all required environment variables are set."""
    required_vars = [
        "DATA_PROCESSED_DIR", "MODEL_DIR", "MODEL", "CLASS_REPORT_VALIDATION",
        "X_TEST", "X_TEST_TFIDF", "Y_TEST", "REFERENCE_EVIDENTLY",
        "DAGSHUB_USER_NAME", "DAGSHUB_USER_TOKEN", "DAGSHUB_REPO_OWNER", "DAGSHUB_REPO_NAME"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {missing_vars}")
    
    print("All required environment variables are set.")

def validate_input_files():
    """Validate that all required input files exist."""
    required_files = {
        "X_test": X_TEST_PATH,
        "X_test_tfidf": X_TEST_TFIDF_PATH,
        "y_test": Y_TEST_PATH,
        "current_run_id": os.path.join(OUTPUT_DIR, "current_run_id.txt"),
    }
    
    missing_files = []
    for name, path in required_files.items():
        if not os.path.exists(path):
            missing_files.append(f"{name}: {path}")
    
    if missing_files:
        raise FileNotFoundError(f"Missing required input files:\n" + "\n".join(missing_files))
    
    print("All required input files found.")

def setup_mlflow():
    """Setup MLflow client and tracking."""
    tracking_uri = (
        f"https://{DAGSHUB_USER_NAME}:{DAGSHUB_USER_TOKEN}"
        f"@dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}.mlflow"
    )
    
    print(f"Setting up MLflow client...")
    print(f"Tracking URI: https://{DAGSHUB_USER_NAME}:***@dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}.mlflow")
    
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("rakuten_final_model")
    client = MlflowClient(tracking_uri=tracking_uri)
    
    return client, tracking_uri

def get_production_model_performance(client):
    """Get the performance of the current production model."""
    try:
        # Get latest production model

        prod_model_version = client.get_model_version_by_alias(MODEL_NAME, "production")
        prod_run_id = prod_model_version.run_id
        print(f"[INFO] Got production model version: {prod_model_version.version}")
        print(f"[INFO] Got run ID: {prod_run_id}")

        # Get production model performance
        print(f"[INFO] Fetching production model run...")
        prod_model_run = client.get_run(prod_run_id)
        prod_f1 = prod_model_run.data.metrics.get("f1_weighted")

        if prod_f1 is not None:
            print(f"[INFO] Production model F1 score: {prod_f1:.4f}")
        else:
            print(f"[ERROR] Production model F1 score not found in metrics")
        
        return prod_f1, prod_run_id
        
    except Exception as e:
        print(f"[ERROR] Error retrieving production model: {e}")
        return None

def get_new_model_run_id():
    """Get the run ID of the newly trained model."""
    run_id_path = os.path.join(OUTPUT_DIR, "current_run_id.txt")
    
    with open(run_id_path, "r") as fh:
        new_run_id = fh.read().strip()
    
    print(f"New model run ID: {new_run_id}")
    return new_run_id

def validate_and_test_model(client, new_run_id):
    """Load and test the newly trained model."""
    print("Loading newly trained model...")

    print(f"[DEBUG] Loading production model...")
    latest_model = mlflow.pyfunc.load_model(model_uri=f"runs:/{new_run_id}/model")
    #run_info = client.get_run(new_run_id)

    #model = load_model(OUTPUT_DIR, MODEL_FILE)
    if latest_model is None:
        raise ValueError(f"[ERROR] Failed to load latest model.")
    else:
        print(f"[INFO] Model loaded successfully")
              
    print("Loading test data...")
    X_test_tfidf, y_test = load_train_data(X_TEST_TFIDF_PATH, Y_TEST_PATH)
    print(f"Test data shape: {X_test_tfidf.shape}")
    print(f"Test labels shape: {y_test.shape}")
    print(f"Test labels distribution:\n{y_test.value_counts().head()}")
    
    print("Running model validation...")
    val_acc, val_f1, class_report, y_pred = prediction_and_metrics(X_test_tfidf, y_test, latest_model)
    
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Validation F1 (weighted): {val_f1:.4f}")
    
    return val_acc, val_f1, class_report, latest_model, y_pred

def promote_model_if_better(client, new_run_id, new_f1, prod_f1):
    """Promote the new model to production if it performs better."""
    try:
        # Find the model version corresponding to the new run
        model_versions = client.search_model_versions("name='SGDClassifier_Model'")
        new_version = next((mv for mv in model_versions if mv.run_id == new_run_id), None)
        
        if new_version is None:
            raise ValueError(f"No model version found for run_id {new_run_id}")
        
        print(f"New model version: {new_version.version}")
        
        # Decision logic for promotion
        should_promote = False
        promotion_reason = ""
        
        if prod_f1 is None:
            should_promote = True
            promotion_reason = "No existing production model"
        elif new_f1 > prod_f1:
            improvement = ((new_f1 - prod_f1) / prod_f1) * 100
            should_promote = True
            promotion_reason = f"Performance improvement: {improvement:.2f}% (F1: {prod_f1:.4f} → {new_f1:.4f})"
        else:
            decline = ((prod_f1 - new_f1) / prod_f1) * 100
            promotion_reason = f"Performance decline: {decline:.2f}% (F1: {prod_f1:.4f} → {new_f1:.4f})"
        
        if should_promote:
            print(f"PROMOTING MODEL: {promotion_reason}")
            client.set_registered_model_alias(
                name="SGDClassifier_Model",
                alias="production",
                version=new_version.version
            )
            print(f"Model version {new_version.version} is now in production.")
            
            try:
                client.transition_model_version_stage(
                    name="SGDClassifier_Model",
                    version=new_version.version,
                    stage="Production"
                )
                print(f"Model version {new_version.version} transitioned to Production stage.")
            except Exception as e:
                print(f"Warning: Could not transition to Production stage: {e}")
            
            return True, promotion_reason
        else:
            print(f"KEEPING CURRENT MODEL: {promotion_reason}")
            return False, promotion_reason
            
    except Exception as e:
        print(f"Error during model promotion: {e}")
        return False, f"Error: {e}"

def main():
    print("=" * 60)
    print("STARTING MODEL VALIDATION PIPELINE")
    print("=" * 60)
    
    try:
        # Validate environment and inputs
        validate_environment()
        validate_input_files()
        
        print(f"Input directory: {INPUT_DIR}")
        print(f"Output directory: {OUTPUT_DIR}")
        print(f"Model file: {MODEL_FILE}")

        # 1. Setup MLflow connection
        print("\n1. Setting up MLflow connection...")
        client, tracking_uri = setup_mlflow()

        # 2. Get current production model performance
        print("\n2. Checking current production model...")
        prod_f1, prod_run_id = get_production_model_performance(client)

        # 3. Get new model run ID
        print("\n3. Getting new model information...")
        new_run_id = get_new_model_run_id()

        # 4. Validate and test the new model
        print("\n4. Validating new model...")
        val_acc, val_f1, class_report, latest_model, y_pred = validate_and_test_model(client, new_run_id)

        # 4.1 Create or load reference data for evidently
        print("Load or create reference dataframe for evidently")
        try:
            reference_df_evidently = pd.read_csv(REFERENCE_EVIDENTLY_PATH)
            print("Found reference file for evidently")
        except FileNotFoundError:
            print("Reference data for evidently not found. Create dataframe")
            X_test = pd.read_csv(X_TEST_PATH).drop(["productid", "imageid"], axis=1).reset_index(drop=True)
            with open(Y_TEST_PATH, "rb") as f:
                y_test = pickle.load(f)
            y_test = y_test.reset_index(drop=True)
            y_pred = pd.Series(y_pred, name="y_pred").reset_index(drop=True)
            reference_df_evidently = pd.concat([X_test, y_test, y_pred], axis=1, ignore_index=False)
            reference_df_evidently = reference_df_evidently.rename(columns={"prdtypecode": "correct_code", "y_pred": "predicted_code"})
            

        # 5. Compare and decide on promotion
        print("\n5. Model promotion decision...")
        promoted, reason = promote_model_if_better(client, new_run_id, val_f1, prod_f1)

        # 6. Log validation metrics to MLflow
        print("\n6. Logging validation results to MLflow...")
        with mlflow.start_run(run_id=new_run_id):

            with open(CLASS_REPORT_PATH, "w") as f:
                f.write(class_report)
            print(f"Classification report saved to: {CLASS_REPORT_PATH}")

            reference_df_evidently.to_csv(REFERENCE_EVIDENTLY_PATH, index=False)

            mlflow.log_metric("test_accuracy", val_acc)
            mlflow.log_metric("test_f1_weighted", val_f1)
            mlflow.log_param("promoted_to_production", promoted)
            mlflow.log_param("promotion_reason", reason)
            
            # Log the test classification report and reference dataframe for evidently
            mlflow.log_artifact(CLASS_REPORT_PATH, artifact_path="classification_report")
            mlflow.log_artifact(REFERENCE_EVIDENTLY_PATH, artifact_path="monitoring_data")

        # Summary
        print("\n" + "=" * 60)
        print("MODEL VALIDATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Test accuracy: {val_acc:.4f}")
        print(f"Test F1 score (weighted): {val_f1:.4f}")
        if prod_f1 is not None:
            print(f"Production F1 score: {prod_f1:.4f}")
        print(f"Model promoted: {'YES' if promoted else 'NO'}")
        print(f"Reason: {reason}")

    except Exception as e:
        print(f"\nERROR in model validation pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
    
    # Track validation results with DVC
    print("\n" + "=" * 60)
    print("TRACKING VALIDATION RESULTS WITH DVC")
    print("=" * 60)
    
    success = track_and_push_with_retry(
        description="track model validation results and reports", 
        max_retries=3,
        force_all=False
    )
    
    if success:
        print("Successfully tracked and pushed validation results to DVC")
    else:
        print("Warning: DVC tracking failed, but model validation completed successfully")