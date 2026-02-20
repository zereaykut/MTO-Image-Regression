import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime

# --- 1. Reusable Logging Mechanism ---
def get_logger(name="project_logger", log_dir="./logs"):
    """
    Creates or retrieves a logger that writes to both file and console.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent adding handlers multiple times
    if not logger.handlers:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"run_{timestamp}.log")

        # File Handler
        f_handler = logging.FileHandler(log_file)
        f_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(f_handler)

        # Stream (Console) Handler
        s_handler = logging.StreamHandler()
        s_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(s_handler)
    
    return logger

# --- 2. Configuration ---
class Config:
    """Central configuration for file paths and parameters."""
    
    # Paths
    MAIN_DATA_PATH = "/home/spidy/Projects/Data/model_data"
    OUTPUT_DIR = "./outputs"
    LOG_DIR = "./logs"
    CHECKPOINT_DIR = "../data/models" # Directory for saving weights
    
    # Data Split Dates
    TRAIN_END = "2024-12-31"
    VAL_START = "2024-06-01"
    VAL_END = "2024-12-31"
    TEST_START = "2025-01-01"
    TEST_END = "2025-10-07"
    
    # Model Params
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4 # Added as per snippet
    
    # Feature List
    PARAMS_LIST = [
        "u1000hPa", "u950hPa", 
        "v1000hPa", "v950hPa", 
        "t1000hPa", "t950hPa", 
        "ws1000hPa", "ws950hPa",
        "wd1000hPa", "wd950hPa",
        "t1000hPa_latitude_differentiate", "t1000hPa_longitude_differentiate",
        "t950hPa_latitude_differentiate", "t950hPa_longitude_differentiate",
    ]

# --- 3. Data Loading ---
class DataLoader:
    def __init__(self, config):
        self.cfg = config
        self.logger = get_logger()

    def load_features(self):
        X = []
        self.logger.info(f"Loading features from {self.cfg.MAIN_DATA_PATH}...")
        
        for param in self.cfg.PARAMS_LIST:
            file_path = os.path.join(self.cfg.MAIN_DATA_PATH, f"{param}.npz")
            try:
                data = np.load(file_path)[param]
                X.append(data)
                self.logger.info(f"Append Completed: {param}")
            except Exception as e:
                self.logger.error(f"Error loading {param}: {e}")
                raise

        # Stack: (Features, Samples, H, W) -> Transpose to (Samples, H, W, Features)
        X = np.stack(X, axis=1).transpose(0, 2, 3, 1)
        self.logger.info(f"Feature loading complete. Raw Shape: {X.shape}")
        return X

    def load_targets(self):
        csv_path = os.path.join(self.cfg.MAIN_DATA_PATH, "df_grt_aligned.csv")
        self.logger.info(f"Loading targets from {csv_path}...")
        df = pd.read_csv(csv_path)
        df["valid_time"] = pd.to_datetime(df["valid_time"])
        df = df.set_index("valid_time")
        return df

# --- 4. Data Processing ---
class DataProcessor:
    def __init__(self, config):
        self.cfg = config
        self.logger = get_logger()

    def split_data(self, X, df_targets):
        # 1. Split Targets
        y_train = df_targets["generation_mw"].loc[:self.cfg.TRAIN_END].values.astype("float32")
        y_val = df_targets["generation_mw"].loc[self.cfg.VAL_START:self.cfg.VAL_END].values.astype("float32")
        y_test = df_targets["generation_mw"].loc[self.cfg.TEST_START:self.cfg.TEST_END].values.astype("float32")

        # 2. Split Features based on target lengths
        len_train = len(y_train)
        len_val = len(y_val)
        len_test = len(y_test)

        X_train = X[:len_train]
        X_val = X[len_train - len_val : len_train] # Overlap logic
        X_test = X[len_train : len_train + len_test]

        self.logger.info(f"Split Complete. Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

        self.logger.info("Scaling features...")
        
        # 1. Reshape to (Samples * H * W, Features) to scale channel-wise
        n_train, h, w, c = X_train.shape
        X_train_reshaped = X_train.reshape(-1, c)
        
        # 2. Fit Scaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_reshaped)
        
        # 3. Transform Val and Test using the TRAIN scaler
        X_val_scaled = scaler.transform(X_val.reshape(-1, c))
        X_test_scaled = scaler.transform(X_test.reshape(-1, c))
        
        # 4. Reshape back to (Samples, H, W, C)
        X_train = X_train_scaled.reshape(n_train, h, w, c)
        X_val = X_val_scaled.reshape(X_val.shape[0], h, w, c)
        X_test = X_test_scaled.reshape(X_test.shape[0], h, w, c)
        
        # OPTIONAL: Scale Targets (y) if values are large (e.g., > 100 MW)
        # y_scaler = StandardScaler()
        # y_train = y_scaler.fit_transform(y_train.reshape(-1, 1))
        # y_val = y_scaler.transform(y_val.reshape(-1, 1))
        # y_test = y_scaler.transform(y_test.reshape(-1, 1))
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# --- 5. Results & Metrics ---
class ResultManager:
    def __init__(self, config):
        self.cfg = config
        if not os.path.exists(self.cfg.OUTPUT_DIR):
            os.makedirs(self.cfg.OUTPUT_DIR)

    def save_predictions(self, y_true, y_pred):
        df = pd.DataFrame({'Real': y_true.flatten(), 'Predicted': y_pred.flatten()})
        path = os.path.join(self.cfg.OUTPUT_DIR, "predictions.csv")
        df.to_csv(path, index=False)
        return path

    def save_metrics(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            mape = 0.0 if np.isnan(mape) or np.isinf(mape) else mape

        metrics = {'RMSE': [rmse], 'MAE': [mae], 'R2': [r2], 'MAPE': [mape]}
        pd.DataFrame(metrics).to_csv(os.path.join(self.cfg.OUTPUT_DIR, "metrics.csv"), index=False)
        return metrics

    def plot_scatter(self, y_true, y_pred):
        plt.figure(figsize=(10, 8))
        plt.scatter(y_true, y_pred, alpha=0.5, color='blue')
        
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        plt.title('Prediction vs Real')
        plt.xlabel('Real Values')
        plt.ylabel('Predicted Values')
        plt.grid(True)
        
        path = os.path.join(self.cfg.OUTPUT_DIR, "scatter_plot.png")
        plt.savefig(path)
        plt.close()
        return path

    def save_history(self, history):
        """Saves training history to CSV and plots."""
        hist_df = pd.DataFrame(history.history)
        hist_df['epoch'] = history.epoch
        csv_path = os.path.join(self.cfg.OUTPUT_DIR, "training_history.csv")
        hist_df.to_csv(csv_path, index=False)
        
        # Plot Loss
        plt.figure(figsize=(10, 6))
        plt.plot(hist_df['epoch'], hist_df['loss'], label='Train Loss')
        plt.plot(hist_df['epoch'], hist_df['val_loss'], label='Val Loss')
        plt.title('Model Loss (MSE)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.cfg.OUTPUT_DIR, "history_loss.png"))
        plt.close()
        
        return csv_path