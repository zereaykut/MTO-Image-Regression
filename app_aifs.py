import sys
import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.utils import Config, DataLoader, DataProcessor, get_logger, ResultManager
from src.models import AIFSForecasterModel

def main():
    # 1. Setup
    cfg = Config()
    logger = get_logger("AIFS_Pipeline")
    logger.info("Configuration loaded. Starting AIFS Pre-training and Fine-tuning pipeline.")

    try:
        # 2. Load Data
        loader = DataLoader(cfg)
        X_raw = loader.load_features()
        df_targets = loader.load_targets()
        
        # 3. Process Data
        processor = DataProcessor(cfg)
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = processor.split_data(X_raw, df_targets)

        # =========================================================
        # DATA PREP FOR PHASE 1: Create shifted targets (t to t+1)
        # =========================================================
        logger.info("Preparing shifted data for Phase 1 (Next-Frame Weather Prediction)...")
        X_train_pre = X_train[:-1]
        Y_train_pre = X_train[1:]

        X_val_pre = X_val[:-1]
        Y_val_pre = X_val[1:]
        
        logger.info(f"Phase 1 Train Shape: X={X_train_pre.shape}, Y={Y_train_pre.shape}")

        # 4. Initialize AIFS Model Wrapper
        input_shape = X_train.shape[1:] 
        logger.info(f"Initializing AIFSForecasterModel with input shape: {input_shape}")
        
        model_wrapper = AIFSForecasterModel(input_shape=input_shape)
        
        # =========================================================
        # PHASE 1: PRE-TRAINING (Learning Meteorological Physics)
        # =========================================================
        logger.info("--------------------------------------------------")
        logger.info("    PHASE 1: WEATHER PHYSICS PRE-TRAINING         ")
        logger.info("--------------------------------------------------")
        
        model_wrapper.compile_pretrain(learning_rate=0.001)
        model_wrapper.pretrain_model.summary(print_fn=logger.info)

        # Optional: Save pre-training weights just in case
        if not os.path.exists(cfg.CHECKPOINT_DIR):
            os.makedirs(cfg.CHECKPOINT_DIR)
        pretrain_checkpoint_path = os.path.join(cfg.CHECKPOINT_DIR, "phase1_pretrain.weights.h5")

        pretrain_callbacks = [
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
            ModelCheckpoint(filepath=pretrain_checkpoint_path, monitor="val_loss", save_best_only=True, save_weights_only=True, verbose=1)
        ]

        logger.info("Starting Phase 1 Training...")
        model_wrapper.pretrain_model.fit(
            X_train_pre, Y_train_pre,
            validation_data=(X_val_pre, Y_val_pre),
            epochs=25,        # 20-30 epochs is usually sufficient for the physics autoencoder
            batch_size=cfg.BATCH_SIZE,
            callbacks=pretrain_callbacks,
            shuffle=True,     # We can shuffle here because we are mapping purely spatial X -> Y pairs
            verbose=1
        )
        logger.info("Phase 1 Complete. Model has learned underlying spatial weather dynamics.")

        # =========================================================
        # PHASE 2: FINE-TUNING (Megawatt Regression)
        # =========================================================
        logger.info("--------------------------------------------------")
        logger.info("    PHASE 2: MEGAWATT FINE-TUNING                 ")
        logger.info("--------------------------------------------------")
        
        # Build the fine-tuning head (freeze_encoder=False allows the whole network to adjust to MW)
        finetune_model = model_wrapper.build_finetune_model(freeze_encoder=False)
        model_wrapper.compile_finetune(learning_rate=0.0005) # Lower LR so we don't wreck the physics weights
        finetune_model.summary(print_fn=logger.info)

        finetune_checkpoint_path = os.path.join(cfg.CHECKPOINT_DIR, "best_aifs_finetune_model.weights.h5")
        
        finetune_callbacks = [
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
            ModelCheckpoint(filepath=finetune_checkpoint_path, monitor="val_loss", save_best_only=True, save_weights_only=True, verbose=1)
        ]

        logger.info("Starting Phase 2 Training...")
        # Note: Using the original unshifted data (X_train, y_train)
        history = finetune_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=cfg.EPOCHS,
            batch_size=cfg.BATCH_SIZE,
            callbacks=finetune_callbacks,
            shuffle=False, # MUST be False for time-series evaluation
            verbose=1
        )

        # =========================================================
        # 5. EVALUATION AND PLOTTING
        # =========================================================
        logger.info("Loading best fine-tuned weights for final evaluation...")
        finetune_model.load_weights(finetune_checkpoint_path)

        if len(X_test) > 0:
            logger.info("Generating predictions on Test Set...")
            y_pred = finetune_model.predict(X_test)
            
            # Flatten outputs
            y_test = y_test.flatten()
            y_pred = y_pred.flatten()

            # Initialize Result Manager
            results = ResultManager(cfg)

            # Extract Dates for Monthly Aggregation
            test_dates = df_targets.loc[cfg.TEST_START:cfg.TEST_END].index

            # Save Results
            results.save_history(history)
            results.save_predictions(y_test, y_pred)
            metrics = results.save_metrics(y_test, y_pred)
            plot_path = results.plot_scatter(y_test, y_pred)
            
            # Save Monthly Results
            monthly_metrics = results.save_monthly_metrics(y_test, y_pred, test_dates)
            hourly_plots_dir = results.plot_hourly_by_month(y_test, y_pred, test_dates)
            
            logger.info(f"Evaluation Complete. Overall Metrics: {metrics}")
            logger.info(f"Monthly Metrics have been saved.")
            logger.info(f"Hourly plots for each month saved to: {hourly_plots_dir}")
        else:
            logger.warning("Test set empty. Skipping evaluation.")

    except Exception as e:
        logger.exception("An error occurred during execution:")
        sys.exit(1)

if __name__ == "__main__":
    main()