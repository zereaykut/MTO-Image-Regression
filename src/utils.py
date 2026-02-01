import os
import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class Config:
    """Central configuration for file paths and model parameters."""
    # Paths
    MAIN_DATA_PATH = "/home/spidy/Projects/Data/model_data"
    PLOT_OUTPATH = "../data/plots" # Assumed based on context
    
    # Data Split Dates
    TRAIN_END = "2024-12-31"
    VAL_START = "2024-06-01"
    VAL_END = "2024-12-31"
    TEST_START = "2025-01-01"
    TEST_END = "2025-10-07"
    
    # Model Params
    BATCH_SIZE = 32
    EPOCHS = 100
    VAL_SPLIT_RATIO = 0.2
    LEARNING_RATE = 0.001
    MAX_POWER = 50

    # Input Features List
    PARAMS_LIST = [
        "u1000hPa", "u950hPa", 
        "v1000hPa", "v950hPa", 
        "t1000hPa", "t950hPa", 
        "ws1000hPa", "ws950hPa",
        "wd1000hPa", "wd950hPa",
        "t1000hPa_latitude_differentiate", "t1000hPa_longitude_differentiate",
        "t950hPa_latitude_differentiate", "t950hPa_longitude_differentiate",
    ]

class DataLoader:
    """Handles loading .npz files and CSV targets."""
    
    def __init__(self, config: Config):
        self.cfg = config
        
    def load_features(self):
        """Loads and stacks feature arrays from .npz files."""
        X = []
        for param in self.cfg.PARAMS_LIST:
            file_path = f"{self.cfg.MAIN_DATA_PATH}/{param}.npz"
            try:
                data = np.load(file_path)
                data = data[param]
                X.append(data)
                print(f"Append Completed: {param}")
            except FileNotFoundError:
                print(f"Error: File not found {file_path}")
                
        # Stack and Transpose: (N, C, H, W) -> (N, H, W, C)
        X = np.stack(X, axis=1).transpose(0, 2, 3, 1)
        return X

    def load_targets(self):
        """Loads the target CSV and aligns time index."""
        path = f"{self.cfg.MAIN_DATA_PATH}/df_grt_aligned.csv"
        df = pd.read_csv(path)
        df["valid_time"] = pd.to_datetime(df["valid_time"])
        df = df.set_index("valid_time")
        return df

class DataProcessor:
    """Handles data splitting and normalization."""
    
    def __init__(self, config: Config):
        self.cfg = config

    def split_data(self, X, df_targets):
        """Splits data into Train, Validation, and Test sets based on dates."""
        
        # Slicing targets based on date ranges
        y_train = df_targets["generation_mw"].loc[:self.cfg.TRAIN_END].values.astype("float32")
        y_val = df_targets["generation_mw"].loc[self.cfg.VAL_START:self.cfg.VAL_END].values.astype("float32")
        y_test = df_targets["generation_mw"].loc[self.cfg.TEST_START:self.cfg.TEST_END].values.astype("float32")
        
        # Slicing features based on target lengths
        len_train = len(y_train)
        len_val = len(y_val)
        len_test = len(y_test)
        
        X_train = X[:len_train]
        # Validation overlaps or follows train depending on logic in original nb
        X_val = X[len_train - len_val : len_train] 
        X_test = X[len_train : len_train + len_test]
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def normalize(self, X_train, X_val, X_test):
        """Normalizes data using statistics from X_train."""
        mean = np.mean(X_train, axis=(0, 2, 3), keepdims=True)
        std = np.std(X_train, axis=(0, 2, 3), keepdims=True)
        epsilon = 1e-7

        X_train_norm = (X_train - mean) / (std + epsilon)
        X_val_norm = (X_val - mean) / (std + epsilon)
        X_test_norm = (X_test - mean) / (std + epsilon)
        
        return X_train_norm, X_val_norm, X_test_norm