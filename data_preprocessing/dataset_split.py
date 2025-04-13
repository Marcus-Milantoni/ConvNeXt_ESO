import pandas as pd
import os


patient_data_path = r""

def stratify_dataset(df, stratify_col, test_size=0.2, random_state=42):
    """
    Splits the dataset into training and testing sets while preserving the distribution of a specified column.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    stratify_col (str): The column name to stratify by.
    test_size (float): The proportion of the dataset to include in the test split.
    random_state (int): Random seed for reproducibility.

    Returns:
    pd.DataFrame: Training set.
    pd.DataFrame: Testing set.
    """
    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[stratify_col],
        random_state=random_state
    )
    
    return train_df, test_df



def main():
    for patient_id in patient_data_path:
        