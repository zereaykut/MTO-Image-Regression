import sys
import numpy as np
import tensorflow as tf
from src.utils import Config, DataLoader, DataProcessor
from src.models import ConvModel

def main():
    # 1. Initialize Config
    cfg = Config()
    print("Configuration loaded.")

    # 2. Load Data
    loader = DataLoader(cfg)
    print("Loading features...")
    X_raw = loader.load_features()
    print(f"Features loaded. Shape: {X_raw.shape}")

    print("Loading targets...")
    df_targets = loader.load_targets()
    
    # 3. Process Data (Split & Normalize)
    processor = DataProcessor(cfg)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = processor.split_data(X_raw, df_targets)
    
    print(f"Train Shape: {X_train.shape}, {y_train.shape}")
    print(f"Val Shape:   {X_val.shape}, {y_val.shape}")
    print(f"Test Shape:  {X_test.shape}, {y_test.shape}")

    # Optional: Normalize
    # X_train, X_val, X_test = processor.normalize(X_train, X_val, X_test)

    # 4. Initialize Model
    input_shape = X_train.shape[1:] # (H, W, C)
    print(f"Initializing ConvModel with input shape: {input_shape}")
    
    model_wrapper = ConvModel(input_shape=input_shape)
    model_wrapper.summary()
    model_wrapper.compile(learning_rate=cfg.LEARNING_RATE)

    # 5. Train
    print("Starting training...")
    # Define callbacks if needed
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10),
        tf.keras.callbacks.ModelCheckpoint(
            filepath="../data/models/cnn_checkpoint.weights.h5", 
            save_best_only=True,
            save_weights_only=True
        )
    ]
    
    history = model_wrapper.train(
        X_train, y_train, 
        X_val, y_val, 
        epochs=cfg.EPOCHS, 
        batch_size=cfg.BATCH_SIZE,
        callbacks=callbacks
    )
    
    print("Training finished.")

if __name__ == "__main__":
    main()