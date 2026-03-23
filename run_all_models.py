import os
import gc
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from src.utils import Config, DataLoader, DataProcessor, get_logger, ResultManager
from src.models import (
    ConvModel, ConvTransformerModel, ConvGATModel,
    ConvGCNModel, ConvGRNModel, ConvSpatialGNNModel, AIFSForecasterModel
)

def main():
    cfg = Config()
    logger = get_logger("Run_All_Models")
    
    # 1. Load & Process Data
    loader = DataLoader(cfg)
    X_raw = loader.load_features()
    df_targets = loader.load_targets()

    processor = DataProcessor(cfg)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = processor.split_data(X_raw, df_targets)

    input_shape = X_train.shape[1:] 

    # 2. Define Registry
    models_to_run = {
        "ConvBaseline": ConvModel,
        "ConvTransformer": ConvTransformerModel,
        # "ConvGCN": ConvGCNModel,
        # "ConvGAT": ConvGATModel,
        # "ConvGRN": ConvGRNModel,
        # "ConvSpatialGNN": ConvSpatialGNNModel,
        # "AIFS_Hybrid": AIFSForecasterModel
    }

    # Store base output dir so we can restore it for the final summary
    base_out_dir = cfg.OUTPUT_DIR
    os.makedirs(base_out_dir, exist_ok=True)
    
    # Dictionary to aggregate metrics for the final comparison plot
    summary_metrics = {}

    # 3. Master Execution Loop
    for model_name, ModelClass in models_to_run.items():
        logger.info(f"\n{'='*50}\n STARTING MODEL: {model_name}\n{'='*50}")

        # Setup isolated directories for the OLD output system
        model_out_dir = os.path.join(base_out_dir, model_name)
        model_ckpt_dir = os.path.join(cfg.CHECKPOINT_DIR, model_name)
        os.makedirs(model_out_dir, exist_ok=True)
        os.makedirs(model_ckpt_dir, exist_ok=True)
        
        # Override output config dynamically so ResultManager saves to the specific folder
        cfg.OUTPUT_DIR = model_out_dir 

        # Clear GPU/RAM memory to prevent OOM errors between runs
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
                
                model_wrapper.pretrain_model.fit(
                    X_train[:-1], X_train[1:], 
                    validation_data=(X_val[:-1], X_val[1:]),
                    epochs=20, batch_size=cfg.BATCH_SIZE, callbacks=cb_pre, shuffle=True, verbose=1
                )
                
                logger.info(f"[{model_name}] Starting Phase 2 Fine-tuning...")
                finetune_model = model_wrapper.build_finetune_model(freeze_encoder=False)
                model_wrapper.compile_finetune(learning_rate=0.0005)
                
                finetune_ckpt = os.path.join(model_ckpt_dir, "best_model.weights.h5")
                cb_fine = [
                    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
                    tf.keras.callbacks.ModelCheckpoint(finetune_ckpt, save_best_only=True, save_weights_only=True)
                ]
                
                history = finetune_model.fit(
                    X_train, y_train, validation_data=(X_val, y_val),
                    epochs=cfg.EPOCHS, batch_size=cfg.BATCH_SIZE, callbacks=cb_fine, shuffle=False, verbose=1
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

            # --- STANDARDIZED EVALUATION (OLD SYSTEM) ---
            if len(X_test) > 0:
                logger.info(f"[{model_name}] Generating predictions and metrics...")
                y_pred = best_model.predict(X_test).flatten()
                y_test_flat = y_test.flatten()
                
                # Extract dates exactly as app.py does
                test_dates = df_targets.loc[cfg.TEST_START:cfg.TEST_END].index

                # Initialize the ResultManager (it will write to cfg.OUTPUT_DIR which is currently the model folder)
                results = ResultManager(cfg)
                
                # Save Individual Model Results exactly like app.py
                try:
                    results.save_history(history)
                except Exception as e:
                    logger.warning(f"Could not save training history: {e}")
                    
                results.save_predictions(y_test_flat, y_pred)
                metrics = results.save_metrics(y_test_flat, y_pred)
                plot_path = results.plot_scatter(y_test_flat, y_pred)
                monthly_metrics = results.save_monthly_metrics(y_test_flat, y_pred, test_dates)
                hourly_plots_dir = results.plot_hourly_by_month(y_test_flat, y_pred, test_dates)
                
                # Extract RMSE, MAE, and R2 for the Master Summary (NEW SYSTEM)
                summary_metrics[model_name] = {
                    "RMSE": metrics["RMSE"][0],
                    "MAE": metrics["MAE"][0],
                    "R2": metrics["R2"][0]
                }
                
                logger.info(f"[{model_name}] Final Test Metrics: {metrics}")
                logger.info(f"[{model_name}] Results saved to: {model_out_dir}")

        except Exception as e:
            logger.error(f"Failed to execute {model_name}. Error: {e}", exc_info=True)
            continue
            
        logger.info(f"\n{'='*50}\n FINISHED MODEL: {model_name}\n{'='*50}")

    # 4. GENERATE MASTER COMPARISON PLOT (NEW SYSTEM)
    if summary_metrics:
        logger.info("Generating Master Comparison Plot...")
        
        # Restore base output directory for the summary (back to root ./outputs/)
        cfg.OUTPUT_DIR = base_out_dir 
        
        # Create a DataFrame from the collected metrics
        df_summary = pd.DataFrame.from_dict(summary_metrics, orient="index")
        df_summary.index.name = "Model"
        df_summary.reset_index(inplace=True)
        
        # Sort by RMSE (lowest error first) so the best model is always on the left
        df_summary = df_summary.sort_values(by="RMSE")
        
        # Save CSV
        csv_path = os.path.join(cfg.OUTPUT_DIR, "all_models_summary.csv")
        df_summary.to_csv(csv_path, index=False)
        
        # ---------------------------------------------------------
        # SPLIT PLOT: 1 Row, 3 Columns (RMSE, MAE, R2)
        # ---------------------------------------------------------
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
            
            # Add exact values on top of the bars
            for bar in bars:
                height = bar.get_height()
                # Format R2 to 3 decimals, errors to 2
                val_str = f"{height:.3f}" if metric == "R2" else f"{height:.2f}"
                
                # Handle potential negative R2 values for text placement
                y_offset = 3 if height >= 0 else -12
                va_align = "bottom" if height >= 0 else "top"
                
                ax.annotate(val_str,
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, y_offset),  
                            textcoords="offset points",
                            ha="center", va=va_align, fontsize=10)
        
        plt.suptitle("Benchmarking All Architectures", fontsize=18, fontweight="bold", y=1.05)
        plt.tight_layout()
        
        # Save the master plot
        plot_path = os.path.join(cfg.OUTPUT_DIR, "all_models_comparison.png")
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Master summary saved to {csv_path}")
        logger.info(f"Master plot saved to {plot_path}")

if __name__ == "__main__":
    # Ensure TF doesn't gobble all VRAM, allowing smooth iteration
    physical_devices = tf.config.list_physical_devices("GPU")
    if physical_devices:
        try:
            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
            
    main()