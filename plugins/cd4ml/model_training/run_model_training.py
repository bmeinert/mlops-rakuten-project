"""
run_model_training.py
"""

import sys
import os
import joblib
import mlflow
import mlflow.sklearn
import yaml
from sklearn.metrics import f1_score, classification_report
from dvc_push_manager import track_and_push_with_retry
from model_training import load_train_data, calculate_class_weights, train_final_model

# ---------- Dagshub ENV ----------
DAGSHUB_USER_NAME = os.getenv("DAGSHUB_USER_NAME")
DAGSHUB_USER_TOKEN = os.getenv("DAGSHUB_USER_TOKEN")
DAGSHUB_REPO_OWNER = os.getenv("DAGSHUB_REPO_OWNER")
DAGSHUB_REPO_NAME  = os.getenv("DAGSHUB_REPO_NAME")

# ---------- Paths ----------
INPUT_DIR  = os.getenv("DATA_PROCESSED_DIR")
OUTPUT_DIR = os.getenv("MODEL_DIR")
X_TRAIN_TFIDF_PATH = os.path.join(INPUT_DIR, os.getenv("X_TRAIN_TFIDF")) 
Y_TRAIN_PATH = os.path.join(INPUT_DIR, os.getenv("Y_TRAIN")) 
X_VALIDATE_TFIDF_PATH = os.path.join(INPUT_DIR, os.getenv("X_VALIDATE_TFIDF")) 
Y_VALIDATE_PATH = os.path.join(INPUT_DIR, os.getenv("Y_VALIDATE")) 
MODEL_PATH = os.path.join(OUTPUT_DIR, os.getenv("MODEL")) 
CLASS_REPORT_PATH = os.path.join(OUTPUT_DIR, os.getenv("CLASS_REPORT")) 
VECTORIZER_PATH = os.path.join(OUTPUT_DIR, os.getenv("TFIDF_VECTORIZER")) 
PRODUCT_DICTIONARY_PATH = os.path.join(OUTPUT_DIR, os.getenv("PRODUCT_DICTIONARY")) 
PARAM_CONFIG_PATH = os.getenv("PARAM_CONFIG")
RUN_ID_PATH = os.path.join(OUTPUT_DIR, os.getenv("CURRENT_RUN_ID")) 

def validate_environment():
    """Validate that all required environment variables are set."""
    required_vars = [
        "DATA_PROCESSED_DIR", "MODEL_DIR",
        "X_TRAIN_TFIDF", "Y_TRAIN", "X_VALIDATE_TFIDF", "Y_VALIDATE", "MODEL",
        "DAGSHUB_USER_NAME", "DAGSHUB_USER_TOKEN", "DAGSHUB_REPO_OWNER", "DAGSHUB_REPO_NAME",
        "CLASS_REPORT", "TFIDF_VECTORIZER", "PRODUCT_DICTIONARY", "PARAM_CONFIG", "CURRENT_RUN_ID"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {missing_vars}")
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True) 
    print(f"Ensured output directory exists: {OUTPUT_DIR}")

def validate_input_files():
    """Validate that all required input files exist."""
    required_files = {
        "X_train_tfidf": X_TRAIN_TFIDF_PATH,
        "y_train": Y_TRAIN_PATH,
        "X_validate_tfidf": X_VALIDATE_TFIDF_PATH,
        "y_validate": Y_VALIDATE_PATH
    }
    
    missing_files = []
    for name, path in required_files.items():
        if not os.path.exists(path):
            missing_files.append(f"{name}: {path}")
    
    if missing_files:
        raise FileNotFoundError(f"Missing required input files:\n" + "\n".join(missing_files))
    
    print("All required input files found.")

def load_config(filename=PARAM_CONFIG_PATH):
    """Load configuration from YAML file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, filename) 
    
    if not os.path.exists(config_path):
        # Try alternative locations
        alt_paths = [
            filename,  # Current directory
            os.path.join(os.getcwd(), filename),  # Working directory 
            os.path.join(os.getcwd(), "config", filename)  # Config subdirectory 
        ]
        
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                config_path = alt_path
                break
        else:
            raise FileNotFoundError(f"Configuration file not found: {filename}")
    
    print(f"Loading configuration from: {config_path}")
    with open(config_path, "r") as fh:
        return yaml.safe_load(fh)

def setup_mlflow():
    """Setup MLflow tracking."""
    tracking_uri = (
        f"https://{DAGSHUB_USER_NAME}:{DAGSHUB_USER_TOKEN}"
        f"@dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}.mlflow"
    )
    
    print(f"Setting up MLflow tracking...")
    print(f"Tracking URI: https://{DAGSHUB_USER_NAME}:***@dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}.mlflow")
    
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("rakuten_final_model")
    
    return tracking_uri

def main():
    print("=" * 60)
    print("STARTING MODEL TRAINING PIPELINE")
    print("=" * 60)
    
    try:
        # Validate environment and inputs
        validate_environment()
        validate_input_files()
        
        print(f"Input directory: {INPUT_DIR}")
        print(f"Output directory: {OUTPUT_DIR}")
        print(f"Loading X_train_tfidf from: {X_TRAIN_TFIDF_PATH}")
        print(f"Loading y_train from: {Y_TRAIN_PATH}")
        print(f"Loading X_validate_tfidf from: {X_VALIDATE_TFIDF_PATH}")
        print(f"Loading y_validate from: {Y_VALIDATE_PATH}")

        # 1. Load training and validation data
        print("\n1. Loading training data...")
        X_train_tfidf, y_train = load_train_data(X_TRAIN_TFIDF_PATH, Y_TRAIN_PATH)
        print(f"Training data shape: {X_train_tfidf.shape}")
        print(f"Training labels shape: {y_train.shape}")
        print(f"Training labels distribution:\n{y_train.value_counts().head()}")
        
        print("\n   Loading validation data...")
        X_validate_tfidf, y_validate = load_train_data(X_VALIDATE_TFIDF_PATH, Y_VALIDATE_PATH)
        print(f"Validation data shape: {X_validate_tfidf.shape}")
        print(f"Validation labels shape: {y_validate.shape}")

        # 2. Calculate class weights
        print("\n2. Calculating class weights...")
        custom_class_weights = calculate_class_weights(y_train)
        print(f"Calculated class weights for {len(custom_class_weights)} classes")

        # 3. Load configuration
        print("\n3. Loading model configuration...")
        config_param = load_config(PARAM_CONFIG_PATH)
        model_params = config_param.get("model", {}).get("params", {})
        print(f"Model parameters: {model_params}")

        # 4. Setup MLflow
        print("\n4. Setting up MLflow tracking...")
        tracking_uri = setup_mlflow()

        # 5. Train model with MLflow tracking
        print("\n5. Training model...")
        
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            print(f"MLflow run ID: {run_id}")
            
            # Train the model
            model = train_final_model(custom_class_weights, X_train_tfidf, y_train)
            print("Model training completed")
            
            # Validate the model
            print("Validating model on validation set...")
            y_pred = model.predict(X_validate_tfidf)
            f1_weighted = f1_score(y_validate, y_pred, average="weighted")
            print(f"Validation F1 score (weighted): {f1_weighted:.4f}")
            
            # Generate classification report
            report = classification_report(y_validate, y_pred)
            with open(CLASS_REPORT_PATH, "w") as f:
                f.write(report)
            print(f"Classification report saved to: {CLASS_REPORT_PATH}")
            
            # Log parameters and metrics to MLflow
            mlflow.log_param("loss", model_params.get("loss", "unknown"))
            mlflow.log_param("alpha", model_params.get("alpha", "unknown"))
            mlflow.log_param("max_iter", model_params.get("max_iter", "unknown"))
            mlflow.log_param("class_weight", "balanced_custom")
            mlflow.log_param("training_samples", X_train_tfidf.shape[0])
            mlflow.log_param("validation_samples", X_validate_tfidf.shape[0])
            mlflow.log_param("feature_count", X_train_tfidf.shape[1])
            
            mlflow.log_metric("f1_weighted", f1_weighted)
            
            # Log model to MLflow
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name="SGDClassifier_Model"
            )
            
            # Log classification report as artifact
            mlflow.log_artifact(CLASS_REPORT_PATH, artifact_path="classification_report")
            mlflow.log_artifact(VECTORIZER_PATH, artifact_path="vectorizer")
            mlflow.log_artifact(PRODUCT_DICTIONARY_PATH, artifact_path="product_dictionary")
            
            # Save run ID for later use
            with open(RUN_ID_PATH, "w") as f:
                f.write(run_id)
            print(f"Run ID saved to: {RUN_ID_PATH}")

        ## 6. Save model locally
        print("\n6. Saving model locally...")
        joblib.dump(model, MODEL_PATH)
        print(f"Model saved to: {MODEL_PATH}")
        
        # Summary
        print("\n" + "=" * 60)
        print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Model type: {type(model).__name__}")
        print(f"Training samples: {X_train_tfidf.shape[0]}")
        print(f"Features: {X_train_tfidf.shape[1]}")
        print(f"Validation F1 score: {f1_weighted:.4f}")
        print(f"MLflow run ID: {run_id}")

    except Exception as e:
        print(f"\nERROR in model training pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
    
    # Track trained model with DVC
    print("\n" + "=" * 60)
    print("TRACKING TRAINED MODEL WITH DVC")
    print("=" * 60)
    
    success = track_and_push_with_retry(
        description="track trained model and artifacts", 
        max_retries=3,
        force_all=False
    )
    
    if success:
        print("Successfully tracked and pushed trained model to DVC")
    else:
        print("Warning: DVC tracking failed, but model training completed successfully")