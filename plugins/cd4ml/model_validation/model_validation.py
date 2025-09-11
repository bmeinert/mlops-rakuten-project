import pandas as pd
from sklearn.metrics import classification_report, f1_score, accuracy_score
import joblib
from sklearn.base import BaseEstimator

def load_model(output_dir: str, model_file: str) -> BaseEstimator:
    """
    Load a scikit-learn model from a specified directory and file.
    """
    model_path = f"{output_dir}/{model_file}"
    model = joblib.load(model_path)

    return model

def prediction_and_metrics(X_validate_tfidf: pd.DataFrame, y_validate: pd.DataFrame, model: BaseEstimator):
    """
    Generate predictions and compute evaluation metrics on validation data.

    Args:
        X_validate_tfidf (pd.DataFrame): TF-IDF transformed validation features.
        y_validate (pd.DataFrame): True labels for the validation set.
        model (BaseEstimator): Trained scikit-learn model.

    Returns:
        tuple:
            - val_accuracy (float): Accuracy score.
            - val_f1 (float): Weighted F1 score.
            - class_report (str): Detailed classification report.
            - y_pred_validate (np.ndarray): Model predictions.
    """
    y_pred_validate = model.predict(X_validate_tfidf)
    val_accuracy = accuracy_score(y_validate, y_pred_validate)
    val_f1 = f1_score(y_validate, y_pred_validate, average='weighted')
    class_report = classification_report(y_validate, y_pred_validate)

    return val_accuracy, val_f1, class_report, y_pred_validate

def save_txt_file(output_dir: str, file_name: str, content: str):
    """
    Save content as a .txt file in the specified directory.
    """
    report_path = f"{output_dir}/{file_name}.txt"
    with open(report_path, "w") as f:
        f.write(content)