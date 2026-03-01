import os
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.utils import get_logger

class ConvModel:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = self._build_model()
        self.logger = get_logger()

    def _build_model(self):
        """Defines the CNN architecture."""
        model = models.Sequential([
            layers.Input(shape=self.input_shape),
            layers.Conv2D(32, 3, activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),    
            layers.Conv2D(64, 3, activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, 3, activation='relu'),
            layers.Dropout(0.1),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(1, activation='linear'),
        ])
        return model

    def compile(self, learning_rate=0.001, weight_decay=1e-4):
        """Compiles the model with AdamW (Adam + Weight Decay) and MSE loss."""
        self.logger.info(f"Compiling model with LR={learning_rate}, Weight Decay={weight_decay}")
        
        # Optimizer with weight decay
        optimizer_fn = optimizers.Adam(learning_rate=learning_rate, weight_decay=weight_decay)
        
        # Loss function
        loss_fn = losses.MeanSquaredError()
        
        self.model.compile(
            optimizer=optimizer_fn, 
            loss=loss_fn, 
            metrics=['mae', 'mse']
        )

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, callbacks=None, checkpoint_dir="../data/models"):
        """Trains the model with shuffle=False and default callbacks."""
        self.logger.info("Starting model training...")
        
        # Default callbacks if none are provided
        if callbacks is None:
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
                
            checkpoint_path = os.path.join(checkpoint_dir, "best_conv_model.weights.h5")
            
            callbacks = [
                EarlyStopping(
                    monitor='val_loss', 
                    patience=10, 
                    restore_best_weights=True,
                    verbose=1
                ),
                ModelCheckpoint(
                    filepath=checkpoint_path, 
                    monitor='val_loss', 
                    save_best_only=True, 
                    save_weights_only=True,
                    verbose=1
                )
            ]
            self.logger.info(f"Using default callbacks. Checkpoints will be saved to: {checkpoint_path}")

        return self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            shuffle=False, # Explicitly set to False as per snippet
            verbose=1
        )