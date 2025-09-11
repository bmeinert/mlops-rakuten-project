"""
run_preprocessing.py

This script orchestrates the full data preprocessing pipeline for our machine learning project.
It performs the following steps:

1. Data Loading and Combining:
   - Loads feature and label data from two separate CSV files using the function `load_combined_data`.
   - Combines these datasets into a single DataFrame and optionally saves the combined raw data as a CSV.

2. Text Cleaning:
   - Applies a custom text cleaning function `clean_text` to the "description" column.
   - The cleaning function removes special characters (keeping letters, numbers, and spaces), converts text to lowercase,
     tokenizes, lemmatizes, and removes stopwords (from English, French, and custom-defined lists).

3. Data Splitting:
   - Splits the combined DataFrame into training, validation, and test sets using the function `split_dataset`.
   - The splitting is stratified based on the target column "prdtypecode" to preserve class distributions.

4. TF-IDF Vectorization:
   - Loads the cleaned text from the training, validation, and test sets.
   - Applies TF-IDF vectorization with custom parameters (ngram_range, max_df, min_df, max_features) via `apply_tfidf`.
   - The vectorizer is fitted on the training data and then used to transform the validation and test data.
   - Optionally, the TF-IDF matrices are saved as pickle files.

5. Saving Processed Data:
   - Saves the training, validation, and test sets (both features and targets) as CSV files in the processed data directory.

Usage:
    To run the preprocessing pipeline, execute this script from the command line:
    
        python run_preprocessing.py

All steps are executed in sequence, ensuring that the raw data is processed and saved for further modeling tasks.
"""

import sys
import os
import pandas as pd
from dvc_push_manager import track_and_push_with_retry
from combine_xy import load_combined_data
from combine_feedback_raw import combine_feedback_raw
from split_data import split_dataset
from tfidf_transform import apply_tfidf
from preprocessing_core import ProductTypePredictorMLflow # replaces step02_text_cleaning



# ---------- ENV-Variablen ----------
RAW_DIR  = os.getenv("DATA_RAW_DIR")
PROC_DIR = os.getenv("DATA_PROCESSED_DIR")
MODEL_DIR = os.getenv("MODEL_DIR")
FEEDBACK_DIR = os.getenv("DATA_FEEDBACK_DIR")
X_RAW_PATH            = os.path.join(RAW_DIR,  os.getenv("X_RAW")) 
Y_RAW_PATH            = os.path.join(RAW_DIR,  os.getenv("Y_RAW")) 
X_Y_RAW_PATH          = os.path.join(RAW_DIR,  os.getenv("X_Y_RAW")) 
X_TRAIN_TFIDF_PATH    = os.path.join(PROC_DIR, os.getenv("X_TRAIN_TFIDF")) 
X_VALIDATE_TFIDF_PATH = os.path.join(PROC_DIR, os.getenv("X_VALIDATE_TFIDF")) 
X_TEST_TFIDF_PATH     = os.path.join(PROC_DIR, os.getenv("X_TEST_TFIDF")) 
TFIDF_VECTORIZER_PATH = os.path.join(MODEL_DIR, os.getenv("TFIDF_VECTORIZER")) 
Y_TRAIN_PATH          = os.path.join(PROC_DIR, os.getenv("Y_TRAIN")) 
Y_VALIDATE_PATH       = os.path.join(PROC_DIR, os.getenv("Y_VALIDATE")) 
Y_TEST_PATH           = os.path.join(PROC_DIR, os.getenv("Y_TEST")) 
FEEDBACK_PATH         = os.path.join(FEEDBACK_DIR, os.getenv("FEEDBACK_CSV")) 
RETRAIN_RAW_PATH      = os.path.join(RAW_DIR, os.getenv("RETRAIN_RAW")) 

def validate_environment():
    """Validate that all required environment variables are set."""
    required_vars = [
        "DATA_RAW_DIR", "DATA_PROCESSED_DIR", "MODEL_DIR", "DATA_FEEDBACK_DIR",
        "X_RAW", "Y_RAW", "X_Y_RAW",
        "X_TRAIN_TFIDF", "X_VALIDATE_TFIDF", "X_TEST_TFIDF",
        "TFIDF_VECTORIZER", "Y_TRAIN", "Y_VALIDATE", "Y_TEST"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {missing_vars}")
    
    # Create directories if they don't exist
    for directory in [RAW_DIR, PROC_DIR, MODEL_DIR]:
        os.makedirs(directory, exist_ok=True)
        print(f"Ensured directory exists: {directory}")

def main():
    print("=" * 60)
    print("STARTING PREPROCESSING PIPELINE")
    print("=" * 60)
    
    try:
        # Validate environment
        validate_environment()
        print(f"Raw data directory: {RAW_DIR}")
        print(f"Processed data directory: {PROC_DIR}")
        print(f"Model directory: {MODEL_DIR}")
        print(f"Feedback directory: {FEEDBACK_DIR}")

        # 1.a Load and combine raw data
        if not os.path.exists(X_Y_RAW_PATH):
            print(f"Combined raw data file not found at {X_Y_RAW_PATH}. Combining XY raw data...")
            print("\n1. Loading and combining raw data...")
            df = load_combined_data(
                x_path=X_RAW_PATH,
                y_path=Y_RAW_PATH,
                save_path=X_Y_RAW_PATH
            )
            print(f"Combined dataset shape: {df.shape}")
            print("First few rows:")
            print(df.head())
        else:
            df = pd.read_csv(X_Y_RAW_PATH)
            df["productid"] = df["productid"].astype("str")
            df["imageid"] = df["imageid"].astype("str")
            df["prdtypecode"] = df["prdtypecode"].astype("str")
            print(f"Loaded existing combined raw data from {X_Y_RAW_PATH} with shape: {df.shape}")

        # 1.b Combine with feedback data if available
        df = combine_feedback_raw(FEEDBACK_PATH, df_raw=df, save_path=RETRAIN_RAW_PATH)
        print(f"Combined dataset shape after feedback: {df.shape}")
        print(df.info())

        # 2. Clean text
        print("\n2. Cleaning text data...")
        if "designation" not in df.columns:
            raise KeyError("'designation' column not found in dataset")
        
        df["cleaned_text"] = df.apply(lambda row: ProductTypePredictorMLflow.clean_text_static(row["designation"], row["description"]), axis=1)   
        print(f"Text cleaning completed. Sample cleaned text: {df['cleaned_text'].iloc[0][:100]}...")
        print(df.head())
        print(df['prdtypecode'].value_counts(ascending=True))

        # 3. Train/Validate/Test Split
        print("\n3. Splitting dataset...")
        if "prdtypecode" not in df.columns:
            raise KeyError("'prdtypecode' column not found in dataset")
        
        X_train, X_validate, X_test, y_train, y_validate, y_test = split_dataset(
            df, target_column="prdtypecode"
        )
        print(f"Training set size: {len(X_train)}")
        print(f"Validation set size: {len(X_validate)}")
        print(f"Test set size: {len(X_test)}")

        # 4. TF-IDF Transformation
        print("\n4. Applying TF-IDF transformation...")
        tfidf_paths = {
            "train":      X_TRAIN_TFIDF_PATH,
            "validate":   X_VALIDATE_TFIDF_PATH,
            "test":       X_TEST_TFIDF_PATH,
            "vectorizer": TFIDF_VECTORIZER_PATH
        }
        
        X_train_tfidf, X_validate_tfidf, X_test_tfidf, vectorizer = apply_tfidf(
            X_train["cleaned_text"],
            X_validate["cleaned_text"],
            X_test["cleaned_text"],
            save_paths=tfidf_paths
        )
        print(f"TF-IDF transformation completed.")
        print(f"Training TF-IDF shape: {X_train_tfidf.shape}")
        print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

        # 5. Save processed data
        print("\n5. Saving processed data...")
        data_mapping = {
            "X_train": X_train,
            "X_validate": X_validate,
            "X_test": X_test,
            "y_train": y_train,
            "y_validate": y_validate,
            "y_test": y_test
        }
        
        for name, df_item in data_mapping.items():
            csv_path = os.path.join(PROC_DIR, f"{name}.csv")
            df_item.to_csv(csv_path, index=False)
            print(f"Saved {name} to {csv_path}")

        # Save target variables as pickle files
        y_train.to_pickle(Y_TRAIN_PATH)
        y_validate.to_pickle(Y_VALIDATE_PATH)
        y_test.to_pickle(Y_TEST_PATH)
        print("Saved target variables as pickle files")

        # Summary
        print("\n" + "=" * 60)
        print("PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"TF-IDF Shape (Train): {X_train_tfidf.shape}")
        print(f"TF-IDF Shape (Validate): {X_validate_tfidf.shape}")
        print(f"TF-IDF Shape (Test): {X_test_tfidf.shape}")
        print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
        print(f"Class distribution in training set:")
        print(y_train.value_counts().head())

    except Exception as e:
        print(f"\nERROR in preprocessing pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
    
    # Track processed data with DVC
    print("\n" + "=" * 60)
    print("TRACKING PROCESSED DATA WITH DVC")
    print("=" * 60)
    
    success = track_and_push_with_retry(
        description="track processed data after preprocessing", 
        max_retries=3,
        force_all=False
    )
    
    if success:
        print("Successfully tracked and pushed processed data to DVC")
    else:
        print("Warning: DVC tracking failed, but preprocessing completed successfully")
