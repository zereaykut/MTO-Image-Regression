import os
import gc
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import Sequence

from src.utils import Config, DataLoader, DataProcessor, get_logger, ResultManager
from src.models import (
    ConvModel, ConvTransformerModel, ConvGATModel,
    ConvGCNModel, ConvGRNModel, ConvSpatialGNNModel, AIFSForecasterModel
)

# =========================================================
# MEMORY-EFFICIENT DATA GENERATOR FOR PHASE 1
# =========================================================
class Phase1DataGenerator(Sequence):
    """
    Generates batches of (X_t, X_{t+1}) on the fly to prevent Out-Of-Memory (OOM) crashes.
    This stops Keras from trying to duplicate massive 4D weather arrays into GPU memory all at once.
    """
    def __init__(self, data, batch_size, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.data) - 1)
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil((len(self.data) - 1) / self.batch_size))

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.indices))
        batch_indices = self.indices[start_idx:end_idx]

        X_batch = self.data[batch_indices]
        Y_batch = self.data[batch_indices + 1]

        return X_batch, Y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

# =========================================================
# SEQUENCE & METADATA GENERATOR FOR PHASE 2 (MULTI-MODAL)
# =========================================================
class Phase2DataGenerator(Sequence):
    """
    Yields historical sequences of weather arrays AND current-hour metadata 
    to map dynamic weather systems to energy output.
    """
    def __init__(self, X, Meta, Y, batch_size, sequence_length=3, shuffle=True):
        self.X = X
        self.Meta = Meta
        self.Y = Y
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.shuffle = shuffle
        
        # Start indices from sequence_length - 1 so we have enough history for the first batch
        self.indices = np.arange(self.sequence_length - 1, len(self.X))
        if self.shuffle: 
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.indices))
        batch_indices = self.indices[start_idx:end_idx]

        X_seq_batch, Meta_batch, Y_batch = [], [], []
        for i in batch_indices:
            # Get the past N frames including current
            X_seq_batch.append(self.X[i - self.sequence_length + 1 : i + 1])
            Meta_batch.append(self.Meta[i])
            Y_batch.append(self.Y[i])

        # FIX: Return inputs as a Tuple, not a List
        return (np.array(X_seq_batch), np.array(Meta_batch)), np.array(Y_batch)

    def on_epoch_end(self):
        if self.shuffle: 
            np.random.shuffle(self.indices)

def main():
    cfg = Config()
    logger = get_logger("Run_All_Models")
    
    # 1. Load & Process Data
    loader = DataLoader(cfg)
    X_raw = loader.load_features()
    df_targets = loader.load_targets()

    processor = DataProcessor(cfg)
    # Updated to receive Metadata from split_data
    (X_train, meta_train, y_train), (X_val, meta_val, y_val), (X_test, meta_test, y_test) = processor.split_data(X_raw, df_targets)

    input_shape = X_train.shape[1:] 

    # 2. Define Registry
    models_to_run = {
        "AIFS_Hybrid": AIFSForecasterModel,
        # "ConvGAT": ConvGATModel,
        # "ConvBaseline": ConvModel,
        # "ConvTransformer": ConvTransformerModel,
        # "ConvGCN": ConvGCNModel,
        # "ConvGRN": ConvGRNModel,
        # "ConvSpatialGNN": ConvSpatialGNNModel,
    }

    base_out_dir = cfg.OUTPUT_DIR
    os.makedirs(base_out_dir, exist_ok=True)
    
    summary_metrics = {}
    all_monthly_metrics = {} 

    # 3. Master Execution Loop
    for model_name, ModelClass in models_to_run.items():
        logger.info(f"\n{'='*50}\n STARTING MODEL: {model_name}\n{'='*50}")

        model_out_dir = os.path.join(base_out_dir, model_name)
        model_ckpt_dir = os.path.join(cfg.CHECKPOINT_DIR, model_name)
        os.makedirs(model_out_dir, exist_ok=True)
        os.makedirs(model_ckpt_dir, exist_ok=True)
        
        cfg.OUTPUT_DIR = model_out_dir 

        tf.keras.backend.clear_session()
        gc.collect()

        try:
            model_wrapper = ModelClass(input_shape=input_shape)
            
            # --- CUSTOM LOGIC FOR AIFS ---
            if model_name == "AIFS_Hybrid":
                logger.info(f"[{model_name}] Starting Phase 1 Pre-training...")
                model_wrapper.compile_pretrain(learning_rate=0.001)
                
                pretrain_ckpt = os.path.join(model_ckpt_dir, "phase1_pretrain.weights.h5")
                cb_pre = [
                    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
                    tf.keras.callbacks.ModelCheckpoint(pretrain_ckpt, save_best_only=True, save_weights_only=True)
                ]
                
                train_gen_p1 = Phase1DataGenerator(X_train, cfg.BATCH_SIZE, shuffle=True)
                val_gen_p1 = Phase1DataGenerator(X_val, cfg.BATCH_SIZE, shuffle=False)
                
                model_wrapper.pretrain_model.fit(
                    train_gen_p1,
                    validation_data=val_gen_p1,
                    epochs=20, callbacks=cb_pre, verbose=1
                )
                
                logger.info(f"[{model_name}] Starting Phase 2 Spatio-Temporal Fine-tuning...")
                
                # Fetch sequence dimensions
                seq_len = 3
                meta_dim = meta_train.shape[1]
                
                finetune_model = model_wrapper.build_finetune_model_multimodal(
                    sequence_length=seq_len, meta_dim=meta_dim, freeze_encoder=False
                )
                
                # Pass custom asymmetric loss penalties
                model_wrapper.compile_finetune(learning_rate=0.0005, under_penalty=1.5, over_penalty=1.0)
                
                finetune_ckpt = os.path.join(model_ckpt_dir, "best_model.weights.h5")
                cb_fine = [
                    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
                    tf.keras.callbacks.ModelCheckpoint(finetune_ckpt, save_best_only=True, save_weights_only=True)
                ]
                
                train_gen_p2 = Phase2DataGenerator(X_train, meta_train, y_train, cfg.BATCH_SIZE, sequence_length=seq_len, shuffle=True)
                val_gen_p2 = Phase2DataGenerator(X_val, meta_val, y_val, cfg.BATCH_SIZE, sequence_length=seq_len, shuffle=False)
                
                history = finetune_model.fit(
                    train_gen_p2, validation_data=val_gen_p2,
                    epochs=cfg.EPOCHS, callbacks=cb_fine, verbose=1
                )
                best_model = finetune_model

            # --- STANDARD LOGIC FOR ALL OTHER MODELS ---
            else:
                model_wrapper.compile(learning_rate=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
                history = model_wrapper.train(
                    X_train, y_train, X_val, y_val, 
                    epochs=cfg.EPOCHS, batch_size=cfg.BATCH_SIZE, checkpoint_dir=model_ckpt_dir
                )
                best_model = model_wrapper.model

            # --- STANDARDIZED EVALUATION ---
            if len(X_test) > 0:
                logger.info(f"[{model_name}] Generating predictions and metrics...")
                
                if model_name == "AIFS_Hybrid":
                    test_gen_p2 = Phase2DataGenerator(X_test, meta_test, y_test, cfg.BATCH_SIZE, sequence_length=seq_len, shuffle=False)
                    y_pred = best_model.predict(test_gen_p2).flatten()
                    
                    # Align test targets and dates by clipping the skipped sequence buffer
                    y_test_flat = y_test[seq_len - 1:].flatten()
                    test_dates = df_targets.loc[cfg.TEST_START:cfg.TEST_END].index[seq_len - 1:]
                else:
                    y_pred = best_model.predict(X_test).flatten()
                    y_test_flat = y_test.flatten()
                    test_dates = df_targets.loc[cfg.TEST_START:cfg.TEST_END].index

                results = ResultManager(cfg)
                
                try:
                    results.save_history(history)
                except Exception as e:
                    logger.warning(f"Could not save training history: {e}")
                    
                results.save_predictions(y_test_flat, y_pred)
                metrics = results.save_metrics(y_test_flat, y_pred)
                results.plot_scatter(y_test_flat, y_pred)
                
                monthly_metrics = results.save_monthly_metrics(y_test_flat, y_pred, test_dates)
                results.plot_hourly_by_month(y_test_flat, y_pred, test_dates)
                
                summary_metrics[model_name] = {
                    "RMSE": metrics["RMSE"][0],
                    "MAE": metrics["MAE"][0],
                    "R2": metrics["R2"][0]
                }
                all_monthly_metrics[model_name] = monthly_metrics
                
                logger.info(f"[{model_name}] Final Test Metrics: {metrics}")
                logger.info(f"[{model_name}] Results saved to: {model_out_dir}")

        except Exception as e:
            logger.error(f"Failed to execute {model_name}. Error: {e}", exc_info=True)
            continue
            
        logger.info(f"\n{'='*50}\n FINISHED MODEL: {model_name}\n{'='*50}")

    # =========================================================
    # 4. GENERATE MASTER COMPARISON PLOTS
    # =========================================================
    if summary_metrics:
        logger.info("Generating Master Comparison Plots...")
        cfg.OUTPUT_DIR = base_out_dir 
        
        # ---------------------------------------------------------
        # Plot 1: Overall Summary Bar Charts
        # ---------------------------------------------------------
        df_summary = pd.DataFrame.from_dict(summary_metrics, orient="index")
        df_summary.index.name = "Model"
        df_summary.reset_index(inplace=True)
        df_summary = df_summary.sort_values(by="RMSE")
        
        csv_path = os.path.join(cfg.OUTPUT_DIR, "all_models_summary.csv")
        df_summary.to_csv(csv_path, index=False)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        x = np.arange(len(df_summary["Model"]))
        
        metrics_to_plot = [
            ("RMSE", "#1f77b4", "Error (MW)", "Lower is Better"),
            ("MAE", "#ff7f0e", "Error (MW)", "Lower is Better"),
            ("R2", "#2ca02c", "Score", "Higher is Better")
        ]
        
        for i, (metric, color, ylabel, goodness) in enumerate(metrics_to_plot):
            ax = axes[i]
            bars = ax.bar(x, df_summary[metric], color=color, width=0.6)
            ax.set_title(f"{metric}\n({goodness})", fontsize=14, pad=10)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(df_summary["Model"], rotation=40, ha="right", fontsize=10)
            ax.grid(axis="y", linestyle="--", alpha=0.7)
            
            for bar in bars:
                height = bar.get_height()
                val_str = f"{height:.3f}" if metric == "R2" else f"{height:.2f}"
                y_offset = 3 if height >= 0 else -12
                va_align = "bottom" if height >= 0 else "top"
                
                ax.annotate(val_str,
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, y_offset),  
                            textcoords="offset points",
                            ha="center", va=va_align, fontsize=10)
        
        plt.suptitle("Benchmarking All Architectures (Overall)", fontsize=18, fontweight="bold", y=1.05)
        plt.tight_layout()
        plot_path = os.path.join(cfg.OUTPUT_DIR, "all_models_overall_comparison.png")
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()

        # ---------------------------------------------------------
        # Plot 2: Monthly Trend Line Plots
        # ---------------------------------------------------------
        if all_monthly_metrics:
            logger.info("Generating Monthly Comparison Line Plots...")
            fig, axes = plt.subplots(3, 1, figsize=(14, 15), sharex=True)
            
            first_model = list(all_monthly_metrics.keys())[0]
            x_labels = [row["Date_Label"] for row in all_monthly_metrics[first_model]]
            
            monthly_metrics_to_plot = [
                ("RMSE", "Error (MW)", "Lower is Better"),
                ("MAE", "Error (MW)", "Lower is Better"),
                ("R2", "Score", "Higher is Better")
            ]
            
            for i, (metric, ylabel, goodness) in enumerate(monthly_metrics_to_plot):
                ax = axes[i]
                
                for model_name, m_data in all_monthly_metrics.items():
                    y_values = [row[metric] for row in m_data]
                    ax.plot(x_labels, y_values, marker="o", label=model_name, linewidth=2, markersize=6)
                
                ax.set_title(f"Monthly {metric} Trend ({goodness})", fontsize=14)
                ax.set_ylabel(ylabel, fontsize=12)
                ax.grid(True, linestyle="--", alpha=0.6)
                ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10)
            
            axes[-1].set_xlabel("Month", fontsize=12)
            plt.xticks(rotation=45, ha="right")
            
            plt.suptitle("Monthly Performance Trends Across Models", fontsize=18, fontweight="bold", y=0.98)
            plt.tight_layout()
            
            monthly_plot_path = os.path.join(cfg.OUTPUT_DIR, "all_models_monthly_trends.png")
            plt.savefig(monthly_plot_path, bbox_inches="tight")
            plt.close()

            logger.info(f"Master overall plot saved to {plot_path}")
            logger.info(f"Master monthly trends plot saved to {monthly_plot_path}")

if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices("GPU")
    if physical_devices:
        try:
            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
            
    main()