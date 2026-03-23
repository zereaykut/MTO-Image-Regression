import os
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.utils import get_logger
from .conv_gcn import build_grid_adjacency

class GraphRecurrentLayer(layers.Layer):
    def __init__(self, units, steps=3, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.steps = steps
        self.gru_cell = layers.GRUCell(units)
        
    def build(self, input_shape):
        self.W_in = layers.Dense(self.units)
        self.W_message = layers.Dense(self.units)

    def call(self, inputs, A):
        # Initial node states
        h = self.W_in(inputs) 
        
        # Message passing steps
        for _ in range(self.steps):
            # 1. Aggregate messages from neighbors
            m = tf.einsum('vw,bwc->bvc', A, h)
            m = self.W_message(m)
            
            # 2. Update node states using GRU
            # GRU expects (Batch*Nodes, Features)
            B = tf.shape(h)[0]
            N = tf.shape(h)[1]
            h_flat = tf.reshape(h, (-1, self.units))
            m_flat = tf.reshape(m, (-1, self.units))
            
            h_new_flat, _ = self.gru_cell(m_flat, [h_flat])
            h = tf.reshape(h_new_flat, (B, N, self.units))
            
        return h

class ConvGRNModel:
    """
    Graph Recurrent Network
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
        A = build_grid_adjacency(H, W)
        
        # The layer itself loops 3 times (steps=3) to pass messages further across the map
        x = GraphRecurrentLayer(64, steps=3)(x, A)
        
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(32, activation="relu")(x)
        outputs = layers.Dense(1, activation="relu")(x)
        return models.Model(inputs, outputs)

    def compile(self, learning_rate=0.001, weight_decay=1e-4):
        self.logger.info(f"Compiling ConvGRNModel with LR={learning_rate}, Weight Decay={weight_decay}")
        optimizer_fn = optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
        self.model.compile(optimizer=optimizer_fn, loss=losses.MeanSquaredError(), metrics=["mae", "mse"])

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, callbacks=None, checkpoint_dir="../data/models"):
        self.logger.info("Starting ConvGRNModel training...")
        if callbacks is None:
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            checkpoint_path = os.path.join(checkpoint_dir, "best_conv_grn_model.weights.h5")
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