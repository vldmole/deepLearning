import os
import pandas as pd
import numpy as np

class DataManager:
  
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

  @staticmethod
  def prepare_data(df):
    """
    Slices the DataFrame:
    - X: Skips the first column (Bias) and the last column (Target).
    - y: Takes only the last column and normalizes labels to 0 and 1.
    """
    # .iloc[:, 1:-1] -> Skips first and last columns
    X = df.iloc[:, 1:-1].values
    
    # .iloc[:, -1] -> Takes only the last column
    y = df.iloc[:, -1].values
    
    # Normalizes targets to 0 and 1
    y = np.where(y <= 0, 0, 1)
    
    return X, y