import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.1,
    val_size: float = 0.2222,
    random_state: int = 36
):
    """
    Splits a DataFrame into training, validation, and test sets using scikit-learn.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing features and the target.
        target_column (str): Name of the target (label) column.
        test_size (float): Proportion of the dataset to be used as the test set. Defaults to 0.1.
        val_size (float): Proportion of the remaining data (after the test split) to be used as the validation set. Defaults to 0.2222.
        random_state (int): Random seed for reproducibility. Defaults to 36.
    
    Returns:
        tuple: (X_train, X_validate, X_test, y_train, y_validate, y_test)
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # First split off the test set with stratification.
    X_train_validate, X_test, y_train_validate, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Split the remaining data into training and validation sets with stratification.
    X_train, X_validate, y_train, y_validate = train_test_split(
        X_train_validate, y_train_validate, test_size=val_size, random_state=random_state, stratify=y_train_validate
    )
    
    return X_train, X_validate, X_test, y_train, y_validate, y_test
