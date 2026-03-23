import os
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from src.utils import get_logger

def build_grid_adjacency(h, w):
    """Creates a normalized adjacency matrix for a 2D grid."""
    n = h * w
    A = np.zeros((n, n), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            idx = i * w + j
            if i > 0: A[idx, (i - 1) * w + j] = 1.0 # Top
            if i < h - 1: A[idx, (i + 1) * w + j] = 1.0 # Bottom
            if j > 0: A[idx, i * w + (j - 1)] = 1.0 # Left
            if j < w - 1: A[idx, i * w + (j + 1)] = 1.0 # Right
            A[idx, idx] = 1.0 # Self-loop
    
    # Normalize: D^(-1/2) * A * D^(-1/2)
    D = np.diag(np.sum(A, axis=1))
    D_inv_sqrt = np.linalg.inv(np.sqrt(D))
    A_norm = np.dot(np.dot(D_inv_sqrt, A), D_inv_sqrt)
    return tf.constant(A_norm, dtype=tf.float32)

class GCNLayer(layers.Layer):
    def __init__(self, units, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = layers.Activation(activation)
        
    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], self.units), initializer="glorot_uniform", trainable=True)

    def call(self, inputs, A):
        # inputs: (Batch, Nodes, Features)
        # A: (Nodes, Nodes)
        # 1. Feature Transformation: X * W
        h = tf.matmul(inputs, self.W) 
        # 2. Neighborhood Aggregation: A * (X * W)
        output = tf.einsum("vw,bwc->bvc", A, h) 
        return self.activation(output)

class ConvGCNModel:
    """
    Graph Convolutional Network
    """
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.logger = get_logger()
        self.model = self._build_model()

    def _build_model(self):
        inputs = layers.Input(shape=self.input_shape)
        
        # CNN Stem (40x40 -> 10x10)
        x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        H, W, C = x.shape[1], x.shape[2], x.shape[3]
        num_nodes = H * W
        
        # Grid to Graph
        x = layers.Reshape((num_nodes, C))(x)
        A = build_grid_adjacency(H, W)
        
        # GCN Blocks
        x = GCNLayer(64)(x, A)
        x = GCNLayer(64)(x, A)
        
        # Output Head
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(32, activation="relu")(x)
        outputs = layers.Dense(1, activation="relu")(x)
        
        return models.Model(inputs, outputs)

    def compile(self, learning_rate=0.001, weight_decay=1e-4):
        """Compiles the model with AdamW to match your app.py interface."""
        self.logger.info(f"Compiling ConvGCNModel with LR={learning_rate}")
        
        optimizer_fn = optimizers.AdamW(
            learning_rate=learning_rate, 
            weight_decay=weight_decay
        )
        
        self.model.compile(
            optimizer=optimizer_fn, 
            loss=losses.MeanSquaredError(), 
            metrics=["mae", "mse"]
        )

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, callbacks=None, checkpoint_dir="../data/models"):
        """Trains the model with shuffle=False and default callbacks."""
        self.logger.info("Starting ConvGCNModel training...")
        
        # Default callbacks if none are provided
        if callbacks is None:
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
                
            checkpoint_path = os.path.join(checkpoint_dir, "best_conv_gcn_model.weights.h5")
            
            callbacks = [
                EarlyStopping(
                    monitor="val_loss", 
                    patience=10, 
                    restore_best_weights=True,
                    verbose=1
                ),
                ModelCheckpoint(
                    filepath=checkpoint_path, 
                    monitor="val_loss", 
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
            shuffle=False, 
            verbose=1
        )