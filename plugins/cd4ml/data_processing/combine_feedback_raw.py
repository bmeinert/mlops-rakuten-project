import pandas as pd
from typing import Optional

def combine_feedback_raw(feedback_path: str, 
                         df_raw: Optional[pd.DataFrame] = None, 
                         xy_raw_path: Optional[str] = None,
                         save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Combine the raw XY data with feedback data.
    
    Args:
        feedback_path (str): Path to the feedback data file.
        df_raw (pd.DataFrame, optional): In-memory raw data, if provided
        xy_raw_path (str): Path to the raw XY data file, if df_raw is not provided.
        save_path (str, optional): Path to save the combined DataFrame. If None, the file is not saved.
        
    
    Returns:
        pd.DataFrame: Combined DataFrame containing both XY and feedback data.
    """
    # Load the raw XY data
    if df_raw is None:
        if xy_raw_path is None:
            raise ValueError("Either df_raw or xy_raw_path must be provided.")
        df_raw = pd.read_csv(xy_raw_path)
        df_raw["productid"] = df_raw["productid"].astype("str")
        df_raw["imageid"] = df_raw["imageid"].astype("str")
        df_raw.to_csv("/app/shared_volume/data/raw/combined_x_y.csv", index=False)
    
    # Load the feedback data
    try:
        df_feedback = pd.read_csv(feedback_path, dtype={"correct_code": "Int64"})
        df_feedback = df_feedback[["designation", "description", "correct_code", "productid", "imageid"]]
        df_feedback = df_feedback.rename(columns={'correct_code': 'prdtypecode'})
        df_feedback = df_feedback.drop_duplicates(subset=['designation', 'description', 'prdtypecode', 'productid', 'imageid'])
        df_feedback = df_feedback.dropna(subset=["prdtypecode"])
        df_feedback['prdtypecode'] = df_feedback['prdtypecode'].astype(int).astype(str)   
    except FileNotFoundError:
        print(f"Feedback file not found at {feedback_path}. Returning raw data only.")
        return df_raw
    
    # Combine the two DataFrames
    df_retrain = pd.concat([df_raw, df_feedback], axis=0, ignore_index=True)
    if save_path is not None:
        df_retrain.to_csv(save_path, index=False)
    
    return df_retrain