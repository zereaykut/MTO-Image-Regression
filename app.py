import sys
import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from src.utils import Config, DataLoader, DataProcessor, get_logger, ResultManager
from src.models import ConvModel, ConvTransformerModel

def main():
    # 1. Setup
    cfg = Config()
    logger = get_logger()
    logger.info("Configuration loaded.")

    try:
        # 2. Load Data
        loader = DataLoader(cfg)
        X_raw = loader.load_features()
        df_targets = loader.load_targets()
        
        # 3. Process Data
        processor = DataProcessor(cfg)
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = processor.split_data(X_raw, df_targets)

        # 4. Initialize Model
        input_shape = X_train.shape[1:] 
        logger.info(f"Initializing ConvModel with input shape: {input_shape}")
        
        # model_wrapper = ConvModel(input_shape=input_shape)
        model_wrapper = ConvTransformerModel(input_shape=input_shape)
        model_wrapper.model.summary(print_fn=logger.info)
        
        # Compile with weight decay
        model_wrapper.compile(learning_rate=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)

        # 5. Define Callbacks (Matching the provided snippet)
        if not os.path.exists(cfg.CHECKPOINT_DIR):
            os.makedirs(cfg.CHECKPOINT_DIR)
        
        checkpoint_path = os.path.join(cfg.CHECKPOINT_DIR, "cnn_checkpoint.weights.h5")
        log_dir = os.path.join(cfg.LOG_DIR, datetime.now().strftime("%Y%m%d-%H%M%S"))

        callbacks_list = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path, 
                monitor="val_loss", 
                save_best_only=True, 
                save_weights_only=True
            ),
            tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        ]

        # 6. Train
        history = model_wrapper.train(
            X_train, y_train, 
            X_val, y_val, 
            epochs=cfg.EPOCHS, 
            batch_size=cfg.BATCH_SIZE,
            callbacks=callbacks_list
        )
        logger.info("Training completed.")

        # Save History
        results = ResultManager(cfg)
        results.save_history(history)

        # 7. Evaluation (Load best weights first)
        logger.info("Loading best weights for evaluation...")
        model_wrapper.model.load_weights(checkpoint_path)

        if len(X_test) > 0:
            logger.info("Generating predictions on Test Set...")
            y_pred = model_wrapper.model.predict(X_test)
            
            # Flatten
            y_test = y_test.flatten()
            y_pred = y_pred.flatten()

            # Save Results
            results.save_predictions(y_test, y_pred)
            metrics = results.save_metrics(y_test, y_pred)
            plot_path = results.plot_scatter(y_test, y_pred)
            
            logger.info(f"Evaluation Complete. Metrics: {metrics}")
        else:
            logger.warning("Test set empty. Skipping evaluation.")

    except Exception as e:
        logger.exception("An error occurred during execution:")
        sys.exit(1)

if __name__ == "__main__":
    main()