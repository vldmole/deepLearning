import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class DataManager:
  
  #-----------------------------------------------------------------------------
  @staticmethod
  def read_data_set(path: str):
    if os.path.exists(path):
        df = pd.read_csv(path)
        print("✅ Arquivo carregado!")
        print(df.head())
        return df
    else:
        print(f"{path} Arquivo não encontrado!")
        return None

  #-----------------------------------------------------------------------------
  @staticmethod
  def read_data_set_with_mask(path: str, mask: list):
    """
    Reads a CSV file using a binary mask to select columns.
    mask: List of integers (1 to include, 0 to exclude)
    """
    if not os.path.exists(path):
        print(f"❌ Error: File not found at {path}")
        return None

    full_columns = pd.read_csv(path, nrows=0).columns.tolist()
    if len(mask) != len(full_columns):
        print(f"❌ Error: Mask length ({len(mask)}) differs from columns ({len(full_columns)})")
        return None

    selected_cols = [col for col, bit in zip(full_columns, mask) if bit == 1]
    df = pd.read_csv(path, usecols=selected_cols)
    
    print(f"✅ Loaded with mask! Columns selected: {len(selected_cols)}")
    return df

  #-----------------------------------------------------------------------------
  @staticmethod
  def split_features_target_by_mask(df, mask):
    """
    Splits the DataFrame into features (X) and labels (y) using a binary mask.
    
    Parameters:
    - df: The input pandas DataFrame.
    - mask: List of integers. 1 for features (X), 0 for labels (y).
    
    Returns:
    - X: DataFrame containing only feature columns.
    - y: DataFrame (or Series) containing only label columns.
    """
    if len(mask) != len(df.columns):
        print(f"❌ Error: Mask length ({len(mask)}) differs from DataFrame columns ({len(df.columns)})")
        return None, None

    feature_indices = [i for i, m in enumerate(mask) if m == 1]
    label_indices = [i for i, m in enumerate(mask) if m == 0]

    X = df.iloc[:, feature_indices]
    y = df.iloc[:, label_indices]

    print(f"✅ Split complete! Features (X): {X.shape}, Labels (y): {y.shape}")
    return X, y

  #-----------------------------------------------------------------------------
  @staticmethod
  def normalize_dataframe(df):
    """
    Normalizes numeric columns of a DataFrame to a [0, 1] range.
    Returns the normalized DataFrame and the scaler object used.
    """
    df_normalized = df.copy()
    numeric_cols = df_normalized.select_dtypes(include=['float64', 'int64']).columns
    scaler = MinMaxScaler()
    df_normalized[numeric_cols] = scaler.fit_transform(df_normalized[numeric_cols])
    
    return df_normalized

  #-----------------------------------------------------------------------------
  @staticmethod
  def binarize_labels(df, label_column, positive_label):
    """
    Converts a label column to 0 and 1.
    pos_label: The value that should become 1. Others become 0.
    """
    df_copy = df.copy()

    df_copy[label_column] = (df_copy[label_column] == positive_label).astype(int)
    
    return df_copy

  #-----------------------------------------------------------------------------
  @staticmethod
  def prepare_dataset(file_path, columns_to_read_mask, label_column, positive_label):

    print(f"\n--- Preparing Dataset: {file_path} ---")
    
    df = DataManager.read_data_set_with_mask(file_path, columns_to_read_mask)
    if df is None: return
    
    df_binaryLabel = DataManager.binarize_labels(df, label_column, positive_label)

    features_label_mask = mask = [1 if col != label_column else 0 for col in df.columns]
    
    df_x, df_y = DataManager.split_features_target_by_mask(df_binaryLabel, features_label_mask )
    df_x_normalized = DataManager.normalize_dataframe(df_x)

    return df_x_normalized.values, df_y.values.flatten()

  #-----------------------------------------------------------------------------