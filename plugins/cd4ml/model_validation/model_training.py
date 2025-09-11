import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.base import BaseEstimator

def load_train_data(X_train_tfidf_path: str, y_train_path: str) -> pd.DataFrame:
    """
    Loads dataframes for 
        - X_train data, modified with tfidf
        - y_train data

    Args:
        X_train_path (str): Path to dataframe (.pkl) containing tfidf modified X data
        y_train_path (str): Path to dataframe (.pkl) containg target data

    Returns:
        pd.DataFrame: X_train_tfidf dataframe
        pd.DataFrame: y_train
    """

    X_train_tfidf = pd.read_pickle(X_train_tfidf_path)
    y_train = pd.read_pickle(y_train_path)

    return X_train_tfidf, y_train

def calculate_class_weights(y_train: pd.DataFrame) -> dict:
    """
    Calculates class weights for model training

    Args:
        y_train (pd.DataFrame): DataFrame with target data

    Returns:
        dictionary: custom_class_weights containing class_labels and class_weights
    """

    class_labels = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=class_labels, y=y_train)
    custom_class_weights = dict(zip(class_labels, class_weights))

    return custom_class_weights


def train_final_model(custom_class_weights: dict, X: pd.DataFrame, y: pd.DataFrame) -> BaseEstimator:
    """
    Defines model and trains with tfidf modified X data and y data.

    Args:
        custom_class_weight: dictionary with adjusted weights
        X: X_train_tfidf (pd.DataFrame): X_train modified with tfidf
        y: y_train (pd.DataFrame): y_train/ target

    Returns: 
        BaseEstimator: trained scitkit-learn model
    """

    model = SGDClassifier(
    loss='log_loss',
    alpha=1.1616550847757421e-06,
    eta0=0.04,
    l1_ratio=0.0,
    learning_rate='optimal',
    penalty='elasticnet',
    random_state=36,
    class_weight=custom_class_weights
)

    model.fit(X, y)

    return model

