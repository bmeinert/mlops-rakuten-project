import pandas as pd

def load_combined_data(x_path: str, y_path: str, save_path: str = None) -> pd.DataFrame:
    """
    Loads feature and target data from separate CSV files, combines them into a single DataFrame,
    converts the target data to string, and optionally saves the combined DataFrame to a CSV file.

    Args:
        x_path (str): Path to the CSV file containing features.
        y_path (str): Path to the CSV file containing target labels.
        save_path (str, optional): If provided, the path where the combined DataFrame will be saved.
    
    Returns:
        pd.DataFrame: Combined DataFrame with features and target.
    """
    df_x = pd.read_csv(x_path, index_col=0)
    df_y = pd.read_csv(y_path, index_col=0)
    
    # Convert target labels to string
    df_y = df_y.astype("str")
    
    df_combined = pd.concat([df_x, df_y], axis=1)
    df_combined["productid"] = df_combined["productid"].astype("str")
    df_combined["imageid"] = df_combined["imageid"].astype("str") 
    
    if save_path is not None:
        df_combined.to_csv(save_path, index=False)
    
    return df_combined
