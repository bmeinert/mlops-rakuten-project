import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from plugins.cd4ml.data_processing.preprocessing_core import ProductTypePredictorMLflow
import pickle

def apply_tfidf(
    train_series: pd.Series,
    validate_series: pd.Series,
    test_series: pd.Series,
    ngram_range: tuple = (1, 3),
    max_df: float = 0.9,
    min_df: int = 2,
    max_features: int = 5000,
    save_paths: dict = None
):
    """
    Applies TF-IDF vectorization to separate training, validation, and test text data.

    The vectorizer is configured with additional parameters (ngram_range, max_df, min_df, max_features).
    It is fitted on the training series, then used to transform the validation and test series.

    Optionally, the resulting TF-IDF matrices can be saved to pickle files if a dictionary of file paths is provided.
    Expected keys in save_paths: 'train', 'validate', 'test'.

    Args:
        train_series (pd.Series): Pandas Series containing training text data.
        validate_series (pd.Series): Pandas Series containing validation text data.
        test_series (pd.Series): Pandas Series containing test text data.
        ngram_range (tuple): The lower and upper boundary of the range of n-values for different n-grams to be extracted. Defaults to (1, 3).
        max_df (float): When building the vocabulary, ignore terms that have a document frequency strictly higher than this threshold. Defaults to 0.9.
        min_df (int): When building the vocabulary, ignore terms that have a document frequency strictly lower than this threshold. Defaults to 2.
        max_features (int): Maximum number of features for the vectorizer. Defaults to 5000.
        save_paths (dict, optional): Dictionary containing file paths to save the resulting TF-IDF matrices.
                                     Expected keys are 'train', 'validate', 'test'. If provided, each matrix is saved as a pickle file.

    Returns:
        tuple: (X_train_tfidf, X_validate_tfidf, X_test_tfidf, vectorizer)
    """
    # Create the TF-IDF vectorizer with the additional parameters.
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_df=max_df,
        min_df=min_df,
        max_features=max_features
    )
    
    # Fit the vectorizer on the training data.
    X_train_tfidf = vectorizer.fit_transform(train_series)
    
    # Transform the validation and test data.
    X_validate_tfidf = ProductTypePredictorMLflow.vectorizer_transform(validate_series, vectorizer)
    X_test_tfidf = ProductTypePredictorMLflow.vectorizer_transform(test_series, vectorizer)
    
    # Optionally, save the TF-IDF matrices to files if save_paths is provided.
    if save_paths is not None:
        if 'train' in save_paths:
            with open(save_paths['train'], 'wb') as f:
                pickle.dump(X_train_tfidf, f)
        if 'validate' in save_paths:
            with open(save_paths['validate'], 'wb') as f:
                pickle.dump(X_validate_tfidf, f)
        if 'test' in save_paths:
            with open(save_paths['test'], 'wb') as f:
                pickle.dump(X_test_tfidf, f)
        if 'vectorizer' in save_paths:
            with open(save_paths['vectorizer'], 'wb') as f:
                pickle.dump(vectorizer, f)
    
    return X_train_tfidf, X_validate_tfidf, X_test_tfidf, vectorizer
