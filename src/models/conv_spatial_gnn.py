import os
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.utils import get_logger
from .conv_gcn import build_grid_adjacency

class GraphSAGELayer(layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.concat_dense = layers.Dense(units, activation="relu")

    def call(self, inputs, A):
        # 1. Aggregate neighbors (Mean aggregation)
        # We calculate the degree (number of neighbors) to average the features
        degree = tf.reduce_sum(A, axis=1, keepdims=True)
        A_mean = A / tf.maximum(degree, 1e-9)
        
        h_neighbors = tf.einsum('vw,bwc->bvc', A_mean, inputs)
        
        # 2. Concatenate Node's own features with Aggregated Neighbor features
        h_combined = tf.concat([inputs, h_neighbors], axis=-1)
        
        # 3. Apply dense transformation
        return self.concat_dense(h_combined)

class ConvSpatialGNNModel:
    """
    Spatial GNN / GraphSAGE
    """
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.logger = get_logger()
        self.model = self._build_model()

    def _build_model(self):
        inputs = layers.Input(shape=self.input_shape)
        
        x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
        x = layers.MaxPooling2D((4, 4))(x)
        
        H, W, C = x.shape[1], x.shape[2], x.shape[3]
        x = layers.Reshape((H * W, C))(x)
        
        # Binary adjacency matrix, we handle normalization in the layer
        A = tf.cast(build_grid_adjacency(H, W) > 0, tf.float32)
        
        x = GraphSAGELayer(64)(x, A)
        x = GraphSAGELayer(64)(x, A)
        
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(32, activation="relu")(x)
        outputs = layers.Dense(1, activation="relu")(x)
        return models.Model(inputs, outputs)

    def compile(self, learning_rate=0.001, weight_decay=1e-4):
        self.logger.info(f"Compiling ConvSpatialGNNModel with LR={learning_rate}, Weight Decay={weight_decay}")
        optimizer_fn = optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
        self.model.compile(optimizer=optimizer_fn, loss=losses.MeanSquaredError(), metrics=["mae", "mse"])

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, callbacks=None, checkpoint_dir="../data/models"):
        self.logger.info("Starting ConvSpatialGNNModel training...")
        if callbacks is None:
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            checkpoint_path = os.path.join(checkpoint_dir, "best_conv_spatial_gnn_model.weights.h5")
            callbacks = [
                EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
                ModelCheckpoint(filepath=checkpoint_path, monitor="val_loss", save_best_only=True, save_weights_only=True, verbose=1)
            ]
            self.logger.info(f"Using default callbacks. Checkpoints will be saved to: {checkpoint_path}")

        return self.model.fit(
            X_train, y_train, validation_data=(X_val, y_val),
            epochs=epochs, batch_size=batch_size, callbacks=callbacks,
            shuffle=False, verbose=1
        )