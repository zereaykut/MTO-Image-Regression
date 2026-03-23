import os
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.utils import get_logger
from .conv_gcn import build_grid_adjacency

class GATLayer(layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W = layers.Dense(units, use_bias=False)
        self.a = layers.Dense(1, use_bias=False)

    def call(self, inputs, A):
        # 1. Linear Transformation
        h = self.W(inputs) # (Batch, N, Units)
        
        # 2. Compute Attention Scores (Dense implementation for small grids)
        N = tf.shape(h)[1]
        h_repeated = tf.repeat(tf.expand_dims(h, 2), N, axis=2) # (B, N, N, Units)
        h_tiled = tf.tile(tf.expand_dims(h, 1), [1, N, 1, 1])   # (B, N, N, Units)
        
        # Concatenate node features with neighbor features
        a_input = tf.concat([h_repeated, h_tiled], axis=-1)
        e = layers.LeakyReLU(0.2)(self.a(a_input)) # (B, N, N, 1)
        e = tf.squeeze(e, axis=-1)
        
        # Mask out non-neighbors using Adjacency matrix A
        mask = -10e9 * (1.0 - A)
        attention = tf.nn.softmax(e + mask, axis=-1) # (B, N, N)
        
        # 3. Aggregate
        output = tf.matmul(attention, h)
        return tf.nn.elu(output)

class ConvGATModel:
    """
    Graph Attention Network
    """
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.logger = get_logger()
        self.model = self._build_model()

    def _build_model(self):
        inputs = layers.Input(shape=self.input_shape)
        
        x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
        x = layers.MaxPooling2D((4, 4))(x) # 40x40 -> 10x10
        
        H, W, C = x.shape[1], x.shape[2], x.shape[3]
        x = layers.Reshape((H * W, C))(x)
        
        # We use a binary adjacency matrix for GAT (unnormalized)
        A = tf.cast(build_grid_adjacency(H, W) > 0, tf.float32)
        
        x = GATLayer(64)(x, A)
        x = GATLayer(64)(x, A)
        
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(32, activation="relu")(x)
        outputs = layers.Dense(1, activation="relu")(x)
        return models.Model(inputs, outputs)

    def compile(self, learning_rate=0.001, weight_decay=1e-4):
        """Compiles the model with AdamW (Adam + Weight Decay) and MSE loss."""
        self.logger.info(f"Compiling ConvGATModel with LR={learning_rate}, Weight Decay={weight_decay}")
        
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
        self.logger.info("Starting ConvGATModel training...")
        
        # Default callbacks if none are provided
        if callbacks is None:
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
                
            checkpoint_path = os.path.join(checkpoint_dir, "best_conv_gat_model.weights.h5")
            
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