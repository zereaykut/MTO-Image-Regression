import os
import gc
import numpy as np
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

    # Input shape naturally adapts to include the new time channels (e.g., 6 features + 4 time = 10)
    input_shape = X_train.shape[1:] 

    # 2. Define Registry
    models_to_run = {
        "ConvBaseline": ConvModel,
        "ConvTransformer": ConvTransformerModel,
        "ConvGCN": ConvGCNModel,
        "ConvGAT": ConvGATModel,
        "ConvGRN": ConvGRNModel,
        "ConvSpatialGNN": ConvSpatialGNNModel,
        "AIFS_Hybrid": AIFSForecasterModel
    }

    # 3. Master Execution Loop
    for model_name, ModelClass in models_to_run.items():
        logger.info(f"\n{'='*50}\n STARTING MODEL: {model_name}\n{'='*50}")

        # Setup isolated directories
        model_out_dir = os.path.join(cfg.OUTPUT_DIR, model_name)
        model_ckpt_dir = os.path.join(cfg.CHECKPOINT_DIR, model_name)
        os.makedirs(model_out_dir, exist_ok=True)
        os.makedirs(model_ckpt_dir, exist_ok=True)
        
        # Override output config dynamically for ResultManager
        cfg.OUTPUT_DIR = model_out_dir 

        # Clear GPU/RAM memory to prevent OOM errors between runs
        tf.keras.backend.clear_session()
        gc.collect()

        try:
            model_wrapper = ModelClass(input_shape=input_shape)
            
            # --- CUSTOM LOGIC FOR AIFS ---
            if model_name == "AIFS_Hybrid":
                # Phase 1
                logger.info(f"[{model_name}] Starting Phase 1 Pre-training...")
                model_wrapper.compile_pretrain(learning_rate=0.001)
                
                # Pretrain callbacks
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
                
                # Phase 2
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

            # --- STANDARDIZED EVALUATION ---
            if len(X_test) > 0:
                logger.info(f"[{model_name}] Generating predictions and metrics...")
                y_pred = best_model.predict(X_test).flatten()
                y_test_flat = y_test.flatten()
                test_dates = df_targets.loc[cfg.TEST_START:cfg.TEST_END].index

                results = ResultManager(cfg)
                results.save_history(history)
                results.save_predictions(y_test_flat, y_pred)
                metrics = results.save_metrics(y_test_flat, y_pred)
                results.plot_scatter(y_test_flat, y_pred)
                results.save_monthly_metrics(y_test_flat, y_pred, test_dates)
                results.plot_hourly_by_month(y_test_flat, y_pred, test_dates)
                
                logger.info(f"[{model_name}] Final Test Metrics: {metrics}")

        except Exception as e:
            logger.error(f"Failed to execute {model_name}. Error: {e}", exc_info=True)
            continue
            
        logger.info(f"\n{'='*50}\n FINISHED MODEL: {model_name}\n{'='*50}")

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